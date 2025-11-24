import os
import pandas as pd
import logging
import torch
import wandb
from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel
from examples.sentence_level.visualizer import visualize_embeddings
from examples.sentence_level.logger_utils import ExperimentLogger

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
BASE_PATH = "examples/sentence_level/wmt_2020"
OUTPUT_DIR = "outputs/hyper_multilingual_v3"
LANG_PAIRS = ["en_de", "en_zh", "et_en", "ne_en", "ro_en", "ru_en", "si_en"]

# --- WANDB SETUP ---
WANDB_KEY = "f9e72a1d0a9c43c835e12effae4461cb85504042"
WANDB_ENTITY = "dipteshkanojia"
WANDB_PROJECT = "hyperqe"

def setup_wandb(config):
    """Authenticates and initializes WandB run."""
    os.environ["WANDB_API_KEY"] = WANDB_KEY
    wandb.login(key=WANDB_KEY)
    
    wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=f"HIN-v3-{config['loss_type']}-FiLM",
        config=config,
        reinit=True
    )

def load_lang_pair(pair_name):
    folder_name = pair_name.replace('_', '-')
    file_code_flat = pair_name.replace('_', '')
    base_dir = os.path.join(BASE_PATH, pair_name, "data", folder_name)
    
    if not os.path.exists(base_dir):
        base_dir_alt = os.path.join(BASE_PATH, pair_name, "data")
        if os.path.exists(os.path.join(base_dir_alt, f"train.{folder_name}.df.short.tsv")):
            base_dir = base_dir_alt
        else: return None, None, None

    def find_file(prefix):
        f1 = os.path.join(base_dir, f"{prefix}.{folder_name}.df.short.tsv")
        if os.path.exists(f1): return f1
        f2 = os.path.join(base_dir, f"{prefix}.{file_code_flat}.df.short.tsv")
        if os.path.exists(f2): return f2
        return None

    def read_tsv(path):
        if path is None: return None
        try:
            df = pd.read_csv(path, sep='\t', quoting=3, on_bad_lines='skip')
            df = df.rename(columns={'original': 'text_a', 'translation': 'text_b', 'mean': 'labels', 'z_scores': 'z_scores'})
            df = df.dropna(subset=['text_a', 'text_b', 'labels'])
            if df['labels'].max() > 1.0: df['labels'] = df['labels'] / 100.0
            df['lang_pair'] = pair_name
            cols = ['text_a', 'text_b', 'labels', 'lang_pair']
            if 'z_scores' in df.columns: cols.append('z_scores')
            return df[cols]
        except: return None

    return read_tsv(find_file("train")), read_tsv(find_file("dev")), read_tsv(find_file("test20") or find_file("test"))

def main():
    logger.info("--- Loading Data ---")
    train_dfs, dev_dfs, test_dfs = [], [], []
    for pair in LANG_PAIRS:
        tr, de, te = load_lang_pair(pair)
        if tr is not None: train_dfs.append(tr)
        if de is not None: dev_dfs.append(de)
        if te is not None: test_dfs.append(te)
    
    if not train_dfs: raise ValueError("No Data.")
    full_train = pd.concat(train_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
    full_dev = pd.concat(dev_dfs).reset_index(drop=True)
    full_test = pd.concat(test_dfs).reset_index(drop=True) if test_dfs else None

    unique_langs = sorted(list(set(full_train['lang_pair'])))
    
    # HIN-v3 Configuration
    model_args = {
        "model_name": "HIN-v3",
        "num_train_epochs": 20,
        "learning_rate": 1e-5,
        "n_gpu": 2,
        "train_batch_size": 32,    # Effective 64
        "eval_batch_size": 64,
        "max_seq_length": 256,
        "evaluate_during_training_steps": 0,
        "overwrite_output_dir": True,
        "manual_seed": 42,
        "loss_type": "circle",     # SOTA Metric Learning Loss
        "architecture": "FiLM + Interaction + Hyperbolic"
    }

    # Initialize WandB
    setup_wandb(model_args)

    model = HyperTransQuestModel(
        model_name="xlm-roberta-base", 
        args=model_args, 
        curvature=1.0, 
        dropout=0.1,
        num_langs=len(unique_langs)
    )

    logger.info("\n--- Starting HIN-v3 Training ---")
    model.train_model(full_train, full_dev, output_dir=OUTPUT_DIR)

    logger.info("\n--- Eval ---")
    core_model = model.model.module if hasattr(model.model, "module") else model.model
    core_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/pytorch_model.bin"))
    
    # Create Lang Map for Eval
    lang_map = {l: i for i, l in enumerate(unique_langs)}

    logger.info("Global Dev Eval:")
    dev_metrics = model.evaluate(full_dev, name="MULTI-DEV", lang_map=lang_map)
    
    # Log Dev Metrics to WandB Summary
    wandb.run.summary["best_dev_spa"] = dev_metrics[0]
    wandb.run.summary["best_dev_spearman"] = dev_metrics[1]

    if full_test is not None:
        logger.info("Global Test Eval:")
        model.evaluate(full_test, name="MULTI-TEST", lang_map=lang_map)
        visualize_embeddings(model, full_test, "viz_hin_v3_test.png", title="HIN-v3 Test Space")
        wandb.log({"Test Visualization": wandb.Image("viz_hin_v3_test.png")})
        
        # Breakdown
        for pair in LANG_PAIRS:
            sub_test = full_test[full_test['lang_pair'] == pair]
            if len(sub_test) > 0:
                spa, s, _, _, _ = model.evaluate(sub_test, name=f"TEST-{pair}", lang_map=lang_map)
                wandb.log({f"test/{pair}_spa": spa, f"test/{pair}_spearman": s})

    wandb.finish()

if __name__ == "__main__":
    main()