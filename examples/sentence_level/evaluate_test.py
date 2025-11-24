import os
import pandas as pd
import torch
import logging
from scipy.stats import pearsonr, spearmanr
from sklearn.metrics import mean_squared_error

# Import your model wrapper
from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel

# Configure Logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG ---
TEST_DATA_PATH = "examples/sentence_level/wmt_2020/en_de/data/en-de/test20.ende.df.short.tsv"
MODEL_PATH = "outputs/hyper_en_de/pytorch_model.bin" # Path to the best saved checkpoint
MODEL_NAME = "xlm-roberta-base"

def load_test_data(filepath):
    if not os.path.exists(filepath):
        print(f"File not found: {filepath}")
        return None
    
    # Robust loading (same as training)
    try:
        df = pd.read_csv(filepath, sep='\t', quoting=3, on_bad_lines='skip')
    except Exception as e:
        print(f"Error reading CSV: {e}")
        return None

    rename_map = {'original': 'text_a', 'translation': 'text_b', 'mean': 'labels'}
    df = df.rename(columns=rename_map)
    
    # Normalize Labels if needed (0-100 -> 0-1)
    if 'labels' in df.columns and df['labels'].max() > 1.0:
        print("Normalizing Test labels (0-100 -> 0-1)...")
        df['labels'] = df['labels'] / 100.0
        
    return df.dropna()

def main():
    # 1. Load Data
    logger.info(f"Loading Test Data from {TEST_DATA_PATH}...")
    test_df = load_test_data(TEST_DATA_PATH)
    
    # 2. Initialize Model Structure
    logger.info("Initializing Hyperbolic Architecture...")
    
    # --- ARGS FIX ---
    # We must pass 'n_gpu' and batch size so the model knows how to run
    eval_args = {
        "n_gpu": 1,
        "eval_batch_size": 64,  # Faster inference
        "max_seq_length": 128,
        "manual_seed": 42
    }

    model_wrapper = HyperTransQuestModel(
        model_name=MODEL_NAME,
        args=eval_args,   # <--- Fixed: Passing args dictionary
        curvature=1.0, 
        dropout=0.1
    )
    
    # 3. Load Trained Weights
    logger.info(f"Loading Weights from {MODEL_PATH}...")
    if torch.cuda.is_available():
        state_dict = torch.load(MODEL_PATH)
    else:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
    # Load the state dict into the internal HyperbolicSiameseNet
    model_wrapper.model.load_state_dict(state_dict)
    model_wrapper.model.eval()
    
    # 4. Predict
    logger.info("Running Prediction on Test Set...")
    input_data = list(zip(test_df['text_a'].tolist(), test_df['text_b'].tolist()))
    
    predictions = model_wrapper.predict(input_data)
    
    # 5. Calculate Metrics (If labels exist)
    if 'labels' in test_df.columns:
        labels = test_df['labels'].tolist()
        pearson = pearsonr(predictions, labels)[0]
        spearman = spearmanr(predictions, labels)[0]
        mse = mean_squared_error(labels, predictions)
        
        print("\n" + "="*30)
        print(f"TEST SET RESULTS ({len(test_df)} samples)")
        print("="*30)
        print(f"Pearson (Linear):   {pearson:.4f}")
        print(f"Spearman (Rank):    {spearman:.4f}  <-- KEY METRIC")
        print(f"MSE:                {mse:.4f}")
        print("="*30 + "\n")
        
        # 6. Qualitative Analysis (Show Top/Bottom)
        test_df['predicted'] = predictions
        test_df['diff'] = abs(test_df['predicted'] - test_df['labels'])
        
        print("--- Best Predictions (Lowest Error) ---")
        print(test_df.sort_values('diff').head(3)[['text_a', 'text_b', 'labels', 'predicted']])
        
        print("\n--- Worst Predictions (Highest Error) ---")
        print(test_df.sort_values('diff', ascending=False).head(3)[['text_a', 'text_b', 'labels', 'predicted']])

if __name__ == "__main__":
    main()