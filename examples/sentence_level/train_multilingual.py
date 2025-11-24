import os
import pandas as pd
import logging
import torch
from sklearn.model_selection import train_test_split
from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel
from examples.sentence_level.visualizer import visualize_embeddings

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIGURATION ---
BASE_PATH = "examples/sentence_level/wmt_2020"
OUTPUT_DIR = "outputs/hyper_multilingual"

# Pairs to train on
LANG_PAIRS = [
    "en_de", 
    "en_zh", 
    "et_en", 
    "ne_en", 
    "ro_en", 
    "ru_en", 
    "si_en"
]

def load_lang_pair(pair_name):
    """
    Robustly loads data for a language pair, handling naming inconsistencies.
    """
    # Folder is usually hyphenated: "en-de"
    folder_name = pair_name.replace('_', '-') 
    # Filename code might be "ende" OR "en-de"
    file_code_hyphen = folder_name
    file_code_flat = pair_name.replace('_', '') # "ende"
    
    # Path to the specific data directory
    # Structure: wmt_2020 / en_de / data / en-de /
    base_dir = os.path.join(BASE_PATH, pair_name, "data", folder_name)
    
    if not os.path.exists(base_dir):
        logger.warning(f"Directory not found: {base_dir}")
        return None, None, None

    def find_file(prefix):
        # Try Flat "train.ende..."
        path_flat = os.path.join(base_dir, f"{prefix}.{file_code_flat}.df.short.tsv")
        if os.path.exists(path_flat): return path_flat
        
        # Try Hyphen "train.en-de..."
        path_hyphen = os.path.join(base_dir, f"{prefix}.{file_code_hyphen}.df.short.tsv")
        if os.path.exists(path_hyphen): return path_hyphen
        
        return None

    # Find Paths
    train_path = find_file("train")
    dev_path = find_file("dev")
    test_path = find_file("test20") # WMT20 usually uses 'test20'

    def read_tsv(path):
        if path is None: return None
        try:
            df = pd.read_csv(path, sep='\t', quoting=3, on_bad_lines='skip')
            df = df.rename(columns={
                'original': 'text_a', 
                'translation': 'text_b', 
                'mean': 'labels',
                'z_scores': 'z_scores'
            })
            df = df.dropna(subset=['text_a', 'text_b', 'labels'])
            if df['labels'].max() > 1.0: df['labels'] = df['labels'] / 100.0
            df['lang_pair'] = pair_name # Tag it
            return df
        except Exception as e:
            logger.warning(f"Failed to read {path}: {e}")
            return None

    logger.info(f"[{pair_name}] Found: Train={bool(train_path)}, Dev={bool(dev_path)}, Test={bool(test_path)}")
    
    return read_tsv(train_path), read_tsv(dev_path), read_tsv(test_path)

# --- LOAD & COMBINE ---
train_dfs, dev_dfs, test_dfs = [], [], []

logger.info("--- Loading Multilingual Data ---")
for pair in LANG_PAIRS:
    tr, de, te = load_lang_pair(pair)
    if tr is not None: 
        train_dfs.append(tr)
    if de is not None: 
        dev_dfs.append(de)
    if te is not None: 
        test_dfs.append(te)

if not train_dfs:
    raise ValueError(f"No training data found in {BASE_PATH}! Please check folder structure.")

# Concatenate
logger.info("Concatenating datasets...")
full_train = pd.concat(train_dfs).sample(frac=1, random_state=42).reset_index(drop=True)
full_dev = pd.concat(dev_dfs).reset_index(drop=True)
full_test = pd.concat(test_dfs).reset_index(drop=True) if test_dfs else None

logger.info(f"TOTAL SAMPLES: Train={len(full_train)}, Dev={len(full_dev)}, Test={len(full_test) if full_test is not None else 0}")

# --- MODEL HYPERPARAMS ---
model_args = {
    "num_train_epochs": 5, # 5 Epochs is plenty for this volume of data
    "learning_rate": 1e-5,
    "n_gpu": 2,
    "train_batch_size": 32, # 64 Global
    "eval_batch_size": 64,
    "max_seq_length": 128,
    "evaluate_during_training_steps": 0, 
    "overwrite_output_dir": True,
    "manual_seed": 42,
}

# Initialize HIN
model = HyperTransQuestModel(model_name="xlm-roberta-base", args=model_args, curvature=1.0, dropout=0.1)

# Train
logger.info("Starting Multilingual HIN Training...")
model.train_model(full_train, full_dev, output_dir=OUTPUT_DIR)

# Evaluate
logger.info("Loading Best Model...")
core_model = model.model.module if hasattr(model.model, "module") else model.model
core_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/pytorch_model.bin"))

logger.info("Evaluating on Combined Dev...")
model.evaluate(full_dev, name="MULTI-DEV")
visualize_embeddings(model, full_dev, "viz_multi_dev.png", title="Multilingual Dev Space")

if full_test is not None:
    logger.info("Evaluating on Combined Test...")
    model.evaluate(full_test, name="MULTI-TEST")
    visualize_embeddings(model, full_test, "viz_multi_test.png", title="Multilingual Test Space")