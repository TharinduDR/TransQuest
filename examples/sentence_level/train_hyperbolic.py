import os
import pandas as pd
import logging
import torch
from sklearn.model_selection import train_test_split
from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel
from examples.sentence_level.visualizer import visualize_embeddings

logging.basicConfig(level=logging.INFO)

def load_and_prep_data(filepath):
    if not os.path.exists(filepath): return None
    print(f"Loading {filepath}...")
    try:
        df = pd.read_csv(filepath, sep='\t', quoting=3, on_bad_lines='skip')
    except: return None
    
    # MAP COLUMNS
    df = df.rename(columns={'original': 'text_a', 'translation': 'text_b', 'mean': 'labels', 'z_scores': 'z_scores'})
    df = df.dropna(subset=['text_a', 'text_b', 'labels'])
    
    if df['labels'].max() > 1.0: 
        df['labels'] = df['labels'] / 100.0
    
    # Keep z_scores if available
    cols = ['text_a', 'text_b', 'labels']
    if 'z_scores' in df.columns: cols.append('z_scores')
        
    return df[cols]

TRAIN_PATH = "examples/sentence_level/wmt_2020/en_de/data/en-de/train.ende.df.short.tsv"
DEV_PATH = "examples/sentence_level/wmt_2020/en_de/data/en-de/dev.ende.df.short.tsv"
TEST_PATH = "examples/sentence_level/wmt_2020/en_de/data/en-de/test20.ende.df.short.tsv"
OUTPUT_DIR = "outputs/hyper_en_de"

train_df = load_and_prep_data(TRAIN_PATH)
eval_df = load_and_prep_data(DEV_PATH)
test_df = load_and_prep_data(TEST_PATH)

# 2-GPU Config
model_args = {
    "num_train_epochs": 15,
    "learning_rate": 1e-5,
    "n_gpu": 2,
    "train_batch_size": 32, 
    "eval_batch_size": 64,
    "max_seq_length": 128,
    "evaluate_during_training_steps": 0, 
    "overwrite_output_dir": True,
    "manual_seed": 42,
}

model = HyperTransQuestModel(model_name="xlm-roberta-base", args=model_args, curvature=1.0, dropout=0.1)

print("\n--- Starting Deep HIN Training ---")
model.train_model(train_df, eval_df, output_dir=OUTPUT_DIR)

print("\n--- Final Evaluation ---")
core_model = model.model.module if hasattr(model.model, "module") else model.model
core_model.load_state_dict(torch.load(f"{OUTPUT_DIR}/pytorch_model.bin"))

print("\n[DEV SET]")
model.evaluate(eval_df, name="DEV")
visualize_embeddings(model, eval_df, "viz_dev_hin_v2.png", title="Deep HIN Dev")

if test_df is not None:
    print("\n[TEST SET]")
    model.evaluate(test_df, name="TEST")
    visualize_embeddings(model, test_df, "viz_test_hin_v2.png", title="Deep HIN Test")