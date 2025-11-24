import pandas as pd
import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import logging

# Import your model wrapper
from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel

# Configure Logging
logging.basicConfig(format='%(asctime)s - %(message)s', level=logging.INFO)
logger = logging.getLogger(__name__)

# --- CONFIG ---
# Use DEV data because it has score labels (needed for coloring points)
DATA_PATH = "examples/sentence_level/wmt_2020/en_de/data/en-de/dev.ende.df.short.tsv" 
MODEL_PATH = "outputs/hyper_en_de/pytorch_model.bin"
MODEL_NAME = "xlm-roberta-base"

def main():
    # 1. Load Data
    logger.info(f"Loading data from {DATA_PATH}...")
    try:
        df = pd.read_csv(DATA_PATH, sep='\t', quoting=3, on_bad_lines='skip')
    except:
        logger.error("Could not read file. Check path.")
        return

    df = df.rename(columns={'original': 'text_a', 'translation': 'text_b', 'mean': 'labels'}).dropna()
    
    # Filter: Take 50 Best (>80) and 50 Worst (<40) for clear contrast
    df['labels'] = df['labels'].astype(float)
    good_df = df[df['labels'] > 80].head(50)
    bad_df = df[df['labels'] < 40].head(50)
    
    if len(bad_df) < 5:
        logger.warning("Not enough 'Bad' examples found. Taking lowest 50 available.")
        bad_df = df.sort_values('labels').head(50)
        
    viz_df = pd.concat([good_df, bad_df])
    logger.info(f"Visualizing {len(viz_df)} pairs ({len(good_df)} Good, {len(bad_df)} Bad)...")
    
    # 2. Load Model
    logger.info("Initializing Model...")
    
    # --- FIX: Define minimal args ---
    viz_args = {
        "n_gpu": 1,
        "eval_batch_size": 32,
        "max_seq_length": 128,
        "manual_seed": 42
    }

    model_wrapper = HyperTransQuestModel(
        MODEL_NAME, 
        args=viz_args,  # <--- PASSED HERE
        curvature=1.0, 
        dropout=0.1
    )
    
    # Load Weights
    if torch.cuda.is_available():
        state_dict = torch.load(MODEL_PATH)
    else:
        state_dict = torch.load(MODEL_PATH, map_location=torch.device('cpu'))
        
    model_wrapper.model.load_state_dict(state_dict)
    model_wrapper.model.eval()
    
    # 3. Extract Embeddings
    logger.info("Encoding sentences...")
    src_texts = viz_df['text_a'].tolist()
    tgt_texts = viz_df['text_b'].tolist()
    
    # Tokenize
    inputs_src = model_wrapper.tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs_tgt = model_wrapper.tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    # Move inputs to the same device as model
    device = model_wrapper.device
    inputs_src = {k: v.to(device) for k, v in inputs_src.items()}
    inputs_tgt = {k: v.to(device) for k, v in inputs_tgt.items()}
    
    with torch.no_grad():
        # Get raw hyperbolic vectors [Batch, Hidden]
        u_src = model_wrapper.model.encode_hyperbolic(inputs_src['input_ids'], inputs_src['attention_mask'])
        v_tgt = model_wrapper.model.encode_hyperbolic(inputs_tgt['input_ids'], inputs_tgt['attention_mask'])
        
        # Move to CPU for plotting
        u_src = u_src.cpu()
        v_tgt = v_tgt.cpu()
        
    # 4. Project to 2D (PCA)
    logger.info("Projecting to 2D...")
    
    # Combine Source and Target to learn a shared 2D space
    combined = torch.cat([u_src, v_tgt], dim=0).numpy()
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    
    # Split back
    n = len(src_texts)
    src_2d = combined_2d[:n]
    tgt_2d = combined_2d[n:]
    
    # 5. Plotting
    plt.figure(figsize=(12, 12))
    ax = plt.gca()
    
    # Draw approximate boundary (max radius seen)
    max_r = np.linalg.norm(combined_2d, axis=1).max()
    circle = plt.Circle((0, 0), max_r * 1.1, color='black', fill=False, linestyle='--', label='Manifold Boundary (Approx)')
    ax.add_artist(circle)
    
    logger.info("Generating Plot...")
    
    # Plot Connections
    # Green lines = Good Translations (Should be short)
    # Red lines = Bad Translations (Should be long)
    for i in range(n):
        score = viz_df.iloc[i]['labels']
        is_good = score > 60
        
        color = 'green' if is_good else 'red'
        alpha = 0.6 if is_good else 0.3
        width = 1.5 if is_good else 0.8
        
        # Draw line
        plt.plot([src_2d[i, 0], tgt_2d[i, 0]], [src_2d[i, 1], tgt_2d[i, 1]], color=color, alpha=alpha, linewidth=width)
        
        # Plot Target Points
        # Only label the first occurance for the legend
        label = None
        if i == 0: label = "Good Translation (Short Distance)"
        if i == len(good_df): label = "Bad Translation (Long Distance)"
        
        plt.scatter(tgt_2d[i, 0], tgt_2d[i, 1], color=color, s=20, label=label)

    # Plot Source Points (Blue)
    plt.scatter(src_2d[:, 0], src_2d[:, 1], color='blue', alpha=0.5, s=30, marker='^', label='Source Sentence')

    plt.title("Hyperbolic Quality Estimation: Geometric Distance = Translation Quality")
    plt.legend(loc='upper right')
    plt.grid(True, linestyle=':', alpha=0.6)
    
    # Save
    save_path = "hyperbolic_viz.png"
    plt.savefig(save_path, dpi=300)
    logger.info(f"SUCCESS: Visualization saved to {save_path}")

if __name__ == "__main__":
    main()