import logging
import math
import random
import os
import json
import numpy as np
import torch
import torch.nn as nn
import wandb
from transformers import AutoTokenizer, get_linear_schedule_with_warmup
from tqdm import tqdm
from scipy.stats import pearsonr, spearmanr, kendalltau
from sklearn.metrics import mean_squared_error

from transquest.algo.sentence_level.siamesetransquest.model_args import SiameseTransQuestArgs
from geoopt.optim import RiemannianAdam
from .hyperbolic_net import HyperbolicInteractionNet
from .losses.ranking_losses import CircleLoss, ListMLELoss, PolyLoss, HyperbolicAngleLoss, EntailmentConeLoss, PearsonLoss
from examples.sentence_level.logger_utils import ExperimentLogger, ConsoleFileHandler

logger = logging.getLogger(__name__)

class HyperTransQuestModel:
    def __init__(self, model_name: str = None, args=None, curvature=1.0, dropout=0.1, num_langs=20, 
                 use_film=False, manifold_type="poincare", proj_type="mlp", learn_curvature=True,
                 scale_factor=0.3, use_interaction=True):
        self.args = args
        if isinstance(args, dict):
            self.args = SiameseTransQuestArgs()
            self.args.update_from_dict(args)
        elif isinstance(args, SiameseTransQuestArgs):
            self.args = args

        if torch.cuda.device_count() > 1 and self.args.n_gpu > 1:
            self.device = torch.device("cuda")
            self.n_gpu = torch.cuda.device_count()
            logger.info(f"🔥 Activated {self.n_gpu} GPUs for HIN-v3!")
        else:
            self.device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
            self.n_gpu = 1
        
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = HyperbolicInteractionNet(
            model_name, curvature=curvature, dropout=dropout, num_langs=num_langs,
            use_film=use_film, manifold_type=manifold_type, proj_type=proj_type,
            learn_curvature=learn_curvature, scale_factor=scale_factor, use_interaction=use_interaction
        )
        
        if self.n_gpu > 1: self.model = nn.DataParallel(self.model)
        self.model.to(self.device)
        
        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.n_gpu > 0: torch.cuda.manual_seed_all(self.args.manual_seed)

    def predict(self, to_predict, lang_ids=None, verbose=True):
        self.model.eval()
        scores = []
        batch_size = self.args.eval_batch_size * max(1, self.n_gpu)
        if lang_ids is None: lang_ids = [0] * len(to_predict)
        data = list(zip(to_predict, lang_ids))

        def chunker(seq, size): return (seq[pos:pos + size] for pos in range(0, len(seq), size))
        iterator = chunker(data, batch_size)
        if verbose: iterator = tqdm(iterator, desc="Predicting", total=math.ceil(len(data)/batch_size))

        with torch.no_grad():
            for batch in iterator:
                s1 = [x[0][0] for x in batch]
                s2 = [x[0][1] for x in batch]
                lid = torch.tensor([x[1] for x in batch]).to(self.device)
                
                inp_src = self.tokenizer(s1, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
                inp_tgt = self.tokenizer(s2, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
                
                src_ids = inp_src['input_ids'].to(self.device)
                src_mask = inp_src['attention_mask'].to(self.device)
                tgt_ids = inp_tgt['input_ids'].to(self.device)
                tgt_mask = inp_tgt['attention_mask'].to(self.device)

                if isinstance(self.model, nn.DataParallel):
                    batch_scores, _, _ = self.model.module.forward(src_ids, src_mask, tgt_ids, tgt_mask, lid)
                else:
                    batch_scores, _, _ = self.model.forward(src_ids, src_mask, tgt_ids, tgt_mask, lid)
                scores.extend(batch_scores.cpu().numpy().tolist())
        return scores

    def calculate_spa(self, preds, labels):
        preds = np.array(preds)
        labels = np.array(labels)
        n = len(preds)
        if n > 2000: 
            indices = np.random.choice(n, 2000, replace=False)
            preds = preds[indices]
            labels = labels[indices]
            n = 2000
        diff_h = labels[:, None] - labels[None, :]
        diff_m = preds[:, None] - preds[None, :]
        p_h = np.zeros_like(diff_h)
        p_h[diff_h > 0] = 1.0
        p_h[diff_h == 0] = 0.5
        p_m = 1 / (1 + np.exp(-10.0 * diff_m))
        return (1.0 - np.abs(p_h - p_m))[np.triu_indices(n, k=1)].mean()

    def log_snapshot(self, src_ids, tgt_ids, scores, labels, step, curvature):
        scores_list = scores.detach().cpu().numpy()
        idx = np.argmax(scores_list)
        s_text = self.tokenizer.decode(src_ids[idx], skip_special_tokens=True)[:50]
        t_text = self.tokenizer.decode(tgt_ids[idx], skip_special_tokens=True)[:50]
        
        # If curvature is a tensor (1-D), take item
        c_val = curvature.item() if torch.is_tensor(curvature) else curvature
        
        logger.info(f"\n=== STEP {step} | C={c_val:.4f} ===")
        logger.info(f"Stats | Mean: {scores_list.mean():.3f} | Std: {scores_list.std():.3f}")
        logger.info(f"[BEST] P: {scores_list[idx]:.4f} | L: {labels[idx].item():.4f}")
        logger.info(f"   S: {s_text}")
        logger.info(f"   T: {t_text}")

    def train_model(self, train_df, eval_df, args=None, output_dir=None, exp_config=None):
        if args: self.args.update_from_dict(args)
        if output_dir and not os.path.exists(output_dir): os.makedirs(output_dir)
        
        with ConsoleFileHandler(os.path.join(output_dir, "console.log")):
            self.args.save(output_dir)
            with open(os.path.join(output_dir, "exp_config.json"), 'w') as f: json.dump(exp_config, f, indent=4)

            train_src = train_df['text_a'].tolist()
            train_tgt = train_df['text_b'].tolist()
            train_labels = train_df['labels'].tolist()
            unique_langs = sorted(list(set(train_df['lang_pair'])))
            lang_map = {l: i for i, l in enumerate(unique_langs)}
            train_lang_ids = [lang_map[l] for l in train_df['lang_pair']]
            logger.info(f"Language Map: {lang_map}")
            
            train_weights = torch.ones(len(train_src))
            if exp_config.get("use_uncertainty", True) and 'z_scores' in train_df.columns:
                try:
                    import ast
                    z_list = [ast.literal_eval(x) for x in train_df['z_scores']]
                    variances = [np.var(z) for z in z_list]
                    weights = torch.tensor([1.0 / (v + 0.2) for v in variances])
                    train_weights = (weights / weights.mean()).to(self.device)
                except: pass

            loss_type = exp_config.get("loss_type", "mnrl")
            if loss_type == "circle": ranking_loss = CircleLoss(m=0.25, gamma=64).to(self.device)
            elif loss_type == "listmle": ranking_loss = ListMLELoss().to(self.device)
            elif loss_type == "poly": ranking_loss = PolyLoss().to(self.device)
            elif loss_type == "angle": ranking_loss = HyperbolicAngleLoss(curvature=1.0).to(self.device)
            elif loss_type == "entailment": ranking_loss = EntailmentConeLoss(K=0.1).to(self.device)
            elif loss_type == "pearson": ranking_loss = PearsonLoss().to(self.device)
            else: ranking_loss = nn.CrossEntropyLoss().to(self.device)
            
            mse_loss = nn.MSELoss(reduction='none')
            angle_loss_fn = HyperbolicAngleLoss(curvature=1.0).to(self.device)

            core_model = self.model.module if hasattr(self.model, "module") else self.model
            
            # Params for Optimizer
            backbone_params = set(core_model.backbone.parameters())
            head_params = list(core_model.hyp_proj.parameters()) + \
                          list(core_model.euc_proj.parameters()) + \
                          list(core_model.interaction.parameters()) + \
                          [core_model.alpha, core_model.beta]
            
            # EXPLICITLY Add C Param if Learnable
            c_params = []
            if core_model.learn_curvature:
                c_params = [core_model.c_param]
            
            if core_model.use_film: head_params += list(core_model.film.parameters())
            backbone_params = list(backbone_params - set(head_params))

            optimizer = RiemannianAdam([
                {'params': backbone_params, 'lr': self.args.learning_rate},
                {'params': head_params, 'lr': 5e-4},
                {'params': c_params, 'lr': 1e-2}  # High LR for C
            ], stabilize=10)
            
            eff_batch = self.args.train_batch_size * max(1, self.n_gpu)
            steps = math.ceil(len(train_src) / eff_batch) * self.args.num_train_epochs
            scheduler = get_linear_schedule_with_warmup(optimizer, num_warmup_steps=int(steps*0.1), num_training_steps=steps)

            self.model.train()
            global_step = 0
            best_spa = -1.0

            for epoch in range(self.args.num_train_epochs):
                combined = list(zip(train_src, train_tgt, train_labels, train_lang_ids, train_weights.tolist()))
                random.shuffle(combined)
                b_src_all, b_tgt_all, b_lbl_all, b_lid_all, b_w_all = zip(*combined)

                batch_iterator = tqdm(range(0, len(train_src), eff_batch), desc=f"Epoch {epoch+1}")

                for i in batch_iterator:
                    b_src = list(b_src_all[i : i+eff_batch])
                    b_tgt = list(b_tgt_all[i : i+eff_batch])
                    b_lbl = torch.tensor(b_lbl_all[i : i+eff_batch], dtype=torch.float).to(self.device)
                    b_lid = torch.tensor(b_lid_all[i : i+eff_batch], dtype=torch.long).to(self.device)
                    b_w = torch.tensor(b_w_all[i : i+eff_batch], dtype=torch.float).to(self.device)

                    inputs_src = self.tokenizer(b_src, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
                    inputs_tgt = self.tokenizer(b_tgt, return_tensors="pt", padding=True, truncation=True, max_length=self.args.max_seq_length)
                    src_ids, src_mask = inputs_src['input_ids'].to(self.device), inputs_src['attention_mask'].to(self.device)
                    tgt_ids, tgt_mask = inputs_tgt['input_ids'].to(self.device), inputs_tgt['attention_mask'].to(self.device)

                    optimizer.zero_grad()
                    
                    if isinstance(self.model, nn.DataParallel):
                        scores_real, u_hyp, v_hyp = self.model(src_ids, src_mask, tgt_ids, tgt_mask, b_lid)
                    else:
                        scores_real, u_hyp, v_hyp = self.model.forward(src_ids, src_mask, tgt_ids, tgt_mask, b_lid)

                    l_mse = (mse_loss(scores_real.view(-1), b_lbl.view(-1)) * b_w).mean()
                    
                    # Use Manual Distance Calculation (Graph Safe)
                    l_rank = 0
                    if loss_type in ["mnrl", "poly", "circle"]:
                        # Calculate matrix manually using the core model's method
                        dists = core_model.manual_dist(u_hyp.unsqueeze(1), v_hyp.unsqueeze(0))
                        scores_mat = torch.exp(-core_model.alpha * dists + core_model.beta)
                        
                        if loss_type == "circle":
                            sp = torch.diagonal(scores_mat)
                            mask = ~torch.eye(scores_mat.size(0), dtype=torch.bool, device=self.device)
                            sn = scores_mat[mask]
                            l_rank = ranking_loss(sp, sn)
                        else:
                            labels_idx = torch.arange(scores_mat.size(0)).to(self.device)
                            l_rank = ranking_loss(scores_mat * 20.0, labels_idx)
                            
                    elif loss_type == "listmle": l_rank = ranking_loss(scores_real, b_lbl)
                    elif loss_type == "angle": l_rank = ranking_loss(u_hyp, v_hyp)
                    elif loss_type == "entailment": l_rank = ranking_loss(u_hyp, v_hyp)
                    elif loss_type == "pearson": l_rank = ranking_loss(scores_real, b_lbl)

                    l_hard = 0
                    if exp_config.get("use_hard_negatives", True) and loss_type != "listmle":
                        perm = torch.randperm(tgt_ids.size(1)).to(self.device)
                        if isinstance(self.model, nn.DataParallel):
                            scores_neg, _, _ = self.model(src_ids, src_mask, tgt_ids[:, perm], tgt_mask, b_lid)
                        else:
                            scores_neg, _, _ = self.model.forward(src_ids, src_mask, tgt_ids[:, perm], tgt_mask, b_lid)
                        l_hard = torch.mean(torch.relu(scores_neg - scores_real + 0.1))

                    loss = 0.4*l_mse + 0.4*l_rank + 0.2*l_hard
                    if exp_config["exp_id"] == "19": 
                        loss += 0.1 * angle_loss_fn(u_hyp, v_hyp)

                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(self.model.parameters(), self.args.max_grad_norm)
                    optimizer.step()
                    scheduler.step()

                    if global_step % 10 == 0:
                        # Check gradients
                        c_grad = 0.0
                        if core_model.learn_curvature and core_model.c_param.grad is not None:
                            c_grad = core_model.c_param.grad.norm().item()
                            
                        wandb.log({
                            "train/loss_total": loss.item(),
                            "train/loss_mse": l_mse.item(),
                            "train/loss_rank": l_rank.item() if torch.is_tensor(l_rank) else l_rank,
                            "train/curv": core_model.c.item(),
                            "train/grad_c": c_grad # Monitor this!
                        }, step=global_step)
                    
                    if global_step % 200 == 0:
                        self.log_snapshot(src_ids, tgt_ids, scores_real, b_lbl, global_step, core_model.c)
                        
                    global_step += 1

                if eval_df is not None:
                    spa, s, p, k, m = self.evaluate(eval_df, lang_map=lang_map)
                    wandb.log({"dev/spa": spa, "dev/spearman": s, "epoch": epoch+1}, step=global_step)
                    if spa > best_spa:
                        best_spa = spa
                        torch.save(core_model.state_dict(), f"{output_dir}/best_model.bin")
                        self.best_metrics = {"spa": spa, "spearman": s, "pearson": p, "kendall": k, "mse": m}
                        self.args.final_curvature = core_model.c.item()

            torch.save(core_model.state_dict(), f"{output_dir}/final_model.bin")
            if output_dir and hasattr(self, 'best_metrics'):
                 experiment_logger.log_run(exp_config, self.args.__dict__, self.best_metrics)
        
        return best_spa

    def evaluate(self, eval_df, name="DEV", lang_map=None):
        eval_src = eval_df['text_a'].tolist()
        eval_tgt = eval_df['text_b'].tolist()
        labels = eval_df['labels'].tolist()
        if lang_map: lang_ids = [lang_map.get(l, 0) for l in eval_df['lang_pair']]
        else: lang_ids = [0] * len(eval_df)
        
        preds = self.predict(list(zip(eval_src, eval_tgt)), lang_ids=lang_ids, verbose=False)
        p = pearsonr(preds, labels)[0]
        s = spearmanr(preds, labels)[0]
        k = kendalltau(preds, labels)[0]
        m = mean_squared_error(labels, preds)
        spa = self.calculate_spa(preds, labels)
        
        logger.info(f"{name} | SPA: {spa:.4f} | S: {s:.4f}")
        return spa, s, p, k, m