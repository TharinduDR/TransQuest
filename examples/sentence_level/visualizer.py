import torch
import matplotlib.pyplot as plt
import numpy as np
from sklearn.decomposition import PCA
import pandas as pd

def visualize_embeddings(model_wrapper, df, output_path, title="Hyperbolic Viz"):
    print(f"Generating Visualization: {output_path}...")
    
    df['labels'] = df['labels'].astype(float)
    # Filter based on normalized labels (0-1)
    good_df = df[df['labels'] > 0.8].head(50)
    bad_df = df[df['labels'] < 0.4].head(50)
    viz_df = pd.concat([good_df, bad_df])
    
    if len(viz_df) < 10: return 

    src_texts = viz_df['text_a'].tolist()
    tgt_texts = viz_df['text_b'].tolist()
    
    inputs_src = model_wrapper.tokenizer(src_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    inputs_tgt = model_wrapper.tokenizer(tgt_texts, return_tensors="pt", padding=True, truncation=True, max_length=128)
    
    device = model_wrapper.device
    inputs_src = {k: v.to(device) for k, v in inputs_src.items()}
    inputs_tgt = {k: v.to(device) for k, v in inputs_tgt.items()}
    
    # Unwrap DataParallel
    core_model = model_wrapper.model.module if hasattr(model_wrapper.model, "module") else model_wrapper.model
    core_model.eval()
    
    with torch.no_grad():
        u_src = core_model.encode_hyperbolic(inputs_src['input_ids'], inputs_src['attention_mask']).cpu()
        v_tgt = core_model.encode_hyperbolic(inputs_tgt['input_ids'], inputs_tgt['attention_mask']).cpu()
        
    combined = torch.cat([u_src, v_tgt], dim=0).numpy()
    pca = PCA(n_components=2)
    combined_2d = pca.fit_transform(combined)
    n = len(src_texts)
    src_2d, tgt_2d = combined_2d[:n], combined_2d[n:]
    
    plt.figure(figsize=(10, 10))
    ax = plt.gca()
    # Draw boundary based on data range
    max_r = np.abs(combined_2d).max() * 1.1
    circle = plt.Circle((0, 0), max_r, color='black', fill=False, linestyle='--')
    ax.add_artist(circle)
    
    for i in range(n):
        is_good = viz_df.iloc[i]['labels'] > 0.6
        color = 'green' if is_good else 'red'
        plt.plot([src_2d[i, 0], tgt_2d[i, 0]], [src_2d[i, 1], tgt_2d[i, 1]], color=color, alpha=0.6 if is_good else 0.3)
        plt.scatter(tgt_2d[i, 0], tgt_2d[i, 1], color=color, s=20)

    plt.scatter(src_2d[:, 0], src_2d[:, 1], color='blue', alpha=0.5, s=30, marker='^', label='Source')
    plt.title(title)
    plt.legend()
    plt.xlim(-max_r, max_r)
    plt.ylim(-max_r, max_r)
    plt.savefig(output_path)
    plt.close()
    print(f"Saved to {output_path}")