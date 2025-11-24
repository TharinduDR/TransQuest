import os
import pandas as pd
import torch
import wandb
import logging
from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel
from examples.sentence_level.logger_utils import ExperimentLogger
from examples.sentence_level.visualizer import visualize_embeddings

logging.basicConfig(level=logging.INFO)

# --- GLOBAL CONFIG ---
WANDB_PROJECT = "hyperqe-ablation-final"
BASE_PATH = "examples/sentence_level/wmt_2020"
OUTPUT_ROOT = "outputs/ablation_study_final"
LANG_PAIRS = ["en_de", "en_zh", "et_en", "ne_en", "ro_en", "ru_en", "si_en"]

# --- EXPERIMENT LIST ---
EXPERIMENTS = [
    # 1-5: Baselines
    {"exp_id": "1", "architecture": "HIN-v2", "loss_type": "mnrl", "use_film": False, "use_hard_negatives": True, "use_uncertainty": True, "desc": "Baseline"},
    {"exp_id": "2", "architecture": "HIN-v3", "loss_type": "circle", "use_film": False, "use_hard_negatives": True, "use_uncertainty": True, "desc": "Circle Loss"},
    {"exp_id": "3", "architecture": "HIN-v3", "loss_type": "poly", "use_film": False, "use_hard_negatives": True, "use_uncertainty": True, "desc": "PolyLoss"},
    {"exp_id": "4", "architecture": "HIN-v3", "loss_type": "listmle", "use_film": False, "use_hard_negatives": True, "use_uncertainty": True, "desc": "ListMLE"},
    {"exp_id": "5", "architecture": "HIN-v3", "loss_type": "angle", "use_film": False, "use_hard_negatives": True, "use_uncertainty": True, "desc": "Angle Loss"},
    
    # 6-10: HIN Features
    {"exp_id": "6", "architecture": "HIN-v3", "loss_type": "mnrl", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + MNRL"},
    {"exp_id": "7", "architecture": "HIN-v3", "loss_type": "circle", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + Circle"},
    {"exp_id": "8", "architecture": "HIN-v3", "loss_type": "listmle", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + ListMLE"},
    {"exp_id": "9", "architecture": "HIN-v3", "loss_type": "angle", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + Angle"},
    {"exp_id": "10", "architecture": "HIN-v4", "loss_type": "circle", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "Grand Unified"},
    
    # 11-13: Geometric Constraints
    {"exp_id": "11", "architecture": "HIN-v3", "loss_type": "entailment", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + EntailmentCone"},
    {"exp_id": "12", "architecture": "HIN-v3", "loss_type": "circle", "manifold_type": "lorentz", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + Circle + Lorentz"},
    {"exp_id": "13", "architecture": "HIN-v3", "loss_type": "circle", "learn_curvature": False, "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + Circle + FixedCurv"},
    
    # 14-16: Architectures
    {"exp_id": "14", "architecture": "HIN-v3", "loss_type": "circle", "proj_type": "grn", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + Circle + GRN"},
    {"exp_id": "15", "architecture": "Siamese", "loss_type": "circle", "use_film": True, "use_interaction": False, "use_hard_negatives": True, "use_uncertainty": True, "desc": "No Interaction"},
    {"exp_id": "16", "architecture": "HIN-v3", "loss_type": "circle", "proj_type": "linear", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + Circle + LinearProj"},
    
    # 17-19: Advanced Losses
    {"exp_id": "17", "architecture": "HIN-v3", "loss_type": "angle", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + AngleLoss"},
    {"exp_id": "18", "architecture": "HIN-v3", "loss_type": "pearson", "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "FiLM + PearsonLoss"},
    {"exp_id": "19", "architecture": "HIN-v4", "loss_type": "circle", "manifold_type": "lorentz", "use_hard_negatives": True, "use_film": True, "use_uncertainty": True, "desc": "HIN-v4 (Lorentz Grand)"},
    
    # 20-23: Scaling & Data
    {"exp_id": "20", "architecture": "HIN-v3", "loss_type": "circle", "scale_factor": 0.1, "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "Scale 0.1"},
    {"exp_id": "21", "architecture": "HIN-v3", "loss_type": "circle", "scale_factor": 0.5, "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "Scale 0.5"},
    {"exp_id": "22", "architecture": "HIN-v3", "loss_type": "circle", "scale_factor": 1.0, "use_film": True, "use_hard_negatives": True, "use_uncertainty": True, "desc": "Scale 1.0"},
    {"exp_id": "23", "architecture": "HIN-v3", "loss_type": "circle", "use_hard_negatives": False, "use_film": True, "use_uncertainty": True, "desc": "No Hard Negatives"},
]

def load_all_data():
    train_dfs, dev_dfs, test_dfs = [], [], []
    for pair in LANG_PAIRS:
        folder = pair.replace('_', '-')
        file_flat = pair.replace('_', '')
        base = os.path.join(BASE_PATH, pair, "data", folder)
        if not os.path.exists(base): base = os.path.join(BASE_PATH, pair, "data")
        
        def get_df(prefix):
            p1 = os.path.join(base, f"{prefix}.{folder}.df.short.tsv")
            p2 = os.path.join(base, f"{prefix}.{file_flat}.df.short.tsv")
            path = p1 if os.path.exists(p1) else (p2 if os.path.exists(p2) else None)
            if path:
                try:
                    df = pd.read_csv(path, sep='\t', quoting=3, on_bad_lines='skip')
                    df = df.rename(columns={'original': 'text_a', 'translation': 'text_b', 'mean': 'labels', 'z_scores': 'z_scores'})
                    df = df.dropna(subset=['text_a', 'text_b', 'labels'])
                    if df['labels'].max() > 1.0: df['labels'] = df['labels'] / 100.0
                    df['lang_pair'] = pair
                    cols = ['text_a', 'text_b', 'labels', 'lang_pair']
                    if 'z_scores' in df.columns: cols.append('z_scores')
                    else: df['z_scores'] = None
                    return df[cols]
                except: return None
            return None

        tr = get_df("train")
        de = get_df("dev")
        
        # Explicitly handle DataFrame to avoid ambiguous truth value error
        te = get_df("test20")
        if te is None: 
            te = get_df("test")
            
        if tr is not None: train_dfs.append(tr)
        if de is not None: dev_dfs.append(de)
        if te is not None: test_dfs.append(te)

    if not train_dfs: raise ValueError("No Data")
    return pd.concat(train_dfs).sample(frac=1, random_state=42).reset_index(drop=True), \
           pd.concat(dev_dfs).reset_index(drop=True) if dev_dfs else None, \
           pd.concat(test_dfs).reset_index(drop=True) if test_dfs else None

def run_experiment(exp_config, train_df, eval_df, test_df, lang_map):
    exp_id = exp_config["exp_id"]
    run_name = f"Exp{exp_id}_{exp_config['desc'].replace(' ', '_')}"
    output_dir = os.path.join(OUTPUT_ROOT, run_name)
    
    # --- RESTART LOGIC: Force restart for 1-10 ---
    force_restart = int(exp_id) <= 10
    if not force_restart and os.path.exists(os.path.join(output_dir, "final_model.bin")):
        print(f"⏩ Skipping {run_name} (Completed)")
        return

    print(f"\n\n{'='*30}\nSTARTING {run_name}\n{'='*30}")
    wandb.init(project=WANDB_PROJECT, name=run_name, config=exp_config, reinit=True)
    
    model_args = {
        "num_train_epochs": 10, # Fast ablation schedule
        "learning_rate": 1e-5,
        "n_gpu": 2,
        "train_batch_size": 32,
        "eval_batch_size": 64,
        "max_seq_length": 256,
        "evaluate_during_training_steps": 0,
        "overwrite_output_dir": True,
        "manual_seed": 42,
        "loss_type": exp_config.get("loss_type", "mnrl")
    }
    
    model = HyperTransQuestModel(
        "xlm-roberta-base", 
        args=model_args, 
        curvature=1.0, 
        dropout=0.1,
        num_langs=len(lang_map),
        use_film=exp_config.get("use_film", False),
        manifold_type=exp_config.get("manifold_type", "poincare"),
        proj_type=exp_config.get("proj_type", "mlp"),
        learn_curvature=exp_config.get("learn_curvature", True),
        scale_factor=exp_config.get("scale_factor", 0.3),
        use_interaction=exp_config.get("use_interaction", True)
    )
    
    model.train_model(train_df, eval_df, output_dir=output_dir, exp_config=exp_config)
    
    # Final Eval using BEST model
    core_model = model.model.module if hasattr(model.model, "module") else model.model
    core_model.load_state_dict(torch.load(f"{output_dir}/best_model.bin"))
    
    test_metrics = {}
    if test_df is not None:
        for pair in LANG_PAIRS:
            sub_test = test_df[test_df['lang_pair'] == pair]
            if len(sub_test) > 0:
                t_spa, t_s, _, _, _ = model.evaluate(sub_test, name=f"TEST-{pair}", lang_map=lang_map)
                test_metrics[pair] = {"spa": t_spa, "spearman": t_s}
        
        visualize_embeddings(model, test_df, f"{output_dir}/viz_test.png", title=f"{run_name} Test")
        wandb.log({"Viz": wandb.Image(f"{output_dir}/viz_test.png")})

    logger = ExperimentLogger(OUTPUT_ROOT)
    logger.log_run(exp_config, model_args, model.best_metrics, test_metrics)
    wandb.finish()

if __name__ == "__main__":
    full_train, full_dev, full_test = load_all_data()
    unique_langs = sorted(list(set(full_train['lang_pair'])))
    lang_map = {l: i for i, l in enumerate(unique_langs)}
    for exp in EXPERIMENTS:
        run_experiment(exp, full_train, full_dev, full_test, lang_map)