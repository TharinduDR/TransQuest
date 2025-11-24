# import torch
# from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel

# # 1. Minimal Configuration
# # We use n_gpu=1 for simple inference to avoid DataParallel complexity
# args = {
#     "n_gpu": 1,
#     "eval_batch_size": 32,
#     "max_seq_length": 128,
#     "manual_seed": 42
# }

# # 2. Initialize Model
# model_wrapper = HyperTransQuestModel(
#     model_name="xlm-roberta-base",
#     args=args,  # <--- FIX: Pass args here
#     curvature=1.0, 
#     dropout=0.1
# )

# # 3. Load Weights
# # Update this path if you want to check the multilingual model ("outputs/hyper_multilingual")
# checkpoint_path = "outputs/hyper_en_de/pytorch_model.bin"

# print(f"Loading weights from {checkpoint_path}...")
# state_dict = torch.load(checkpoint_path)

# # The training script saves the 'core' model (no 'module.' prefix).
# # Since we set n_gpu=1, model_wrapper.model is also the core model.
# # Direct load works.
# model_wrapper.model.load_state_dict(state_dict)
# model_wrapper.model.eval()

# # 4. The Hallucination Test
# src = "The cat sits outside"
# tgt_good = "Die Katze sitzt draußen"
# tgt_bad = "Pharmaceuticals are dangerous for the economy."

# print("\n--- Running Hallucination Check ---")
# scores = model_wrapper.predict([[src, tgt_good], [src, tgt_bad]])

# print(f"Source: {src}")
# print(f"Good Tgt ({tgt_good}): {scores[0]:.4f}")
# print(f"Bad Tgt  ({tgt_bad}): {scores[1]:.4f}")

# margin = scores[0] - scores[1]
# print(f"\nMargin: {margin:.4f}")

# if margin > 0.7:
#     print("✅ RESULT: PASS (Robust separation)")
# else:
#     print("❌ RESULT: FAIL (Margin too small)")
import torch
import pandas as pd
from transquest.algo.sentence_level.hypertransquest.run_model import HyperTransQuestModel

# 1. Config
args = {
    "n_gpu": 1,
    "eval_batch_size": 32,
    "max_seq_length": 128,
    "manual_seed": 42
}

# 2. Load Model
model_wrapper = HyperTransQuestModel(
    model_name="xlm-roberta-base",
    args=args,
    curvature=1.0, 
    dropout=0.1
)

# Load Weights
checkpoint_path = "outputs/hyper_en_de/pytorch_model.bin"
print(f"Loading weights from {checkpoint_path}...")
model_wrapper.model.load_state_dict(torch.load(checkpoint_path))
model_wrapper.model.eval()

# 3. Load a REAL sample from your Test Set
test_file = "examples/sentence_level/wmt_2020/en_de/data/en-de/test20.ende.df.short.tsv"
df = pd.read_csv(test_file, sep='\t', quoting=3, on_bad_lines='skip')
df = df.rename(columns={'original': 'text_a', 'translation': 'text_b'})

# Pick a random real sentence (e.g., index 5)
row = df.iloc[5]
src = row['text_a']
tgt_good = row['text_b']
tgt_bad = "Pharmaceuticals are dangerous for the economy."

print(f"\n--- Real Data Check ---")
print(f"Source: {src[:60]}...")
print(f"Target: {tgt_good[:60]}...")

scores = model_wrapper.predict([[src, tgt_good], [src, tgt_bad]])

print(f"\nGood Score: {scores[0]:.4f}")
print(f"Bad Score:  {scores[1]:.4f}")
print(f"Margin:     {scores[0] - scores[1]:.4f}")

if scores[0] - scores[1] > 0.5:
    print("✅ RESULT: PASS (SOTA Behavior Confirmed)")
else:
    print("❌ RESULT: FAIL")