import csv
import os
import datetime

# The data you just pasted
run_data = {
    "Timestamp": datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S"),
    "Model": "HIN-v3 (Multilingual FiLM)",
    "Loss": "Circle Loss",
    "Epochs": 20,
    "Batch": 64,
    "SeqLen": 256,
    "LR": "1e-5",
    "Curvature": "Learnable (Init 1.0)",
    # Dev Metrics
    "Dev_SPA": 0.7140,
    "Dev_Spearman": 0.6708,
    "Dev_Kendall": 0.4839,
    "Dev_MSE": 0.0393,
    # Test Metrics
    "Test_Global_SPA": 0.7159,
    "Test_Global_Spearman": 0.6880,
    "Test_en_de_Spearman": 0.1528,
    "Test_en_zh_Spearman": 0.2584,
    "Test_et_en_Spearman": 0.5407,
    "Test_ne_en_Spearman": 0.4529,
    "Test_ro_en_Spearman": 0.6392,
    "Test_ru_en_Spearman": 0.5909,
    "Test_si_en_Spearman": 0.4778
}

output_file = "outputs/hyper_multilingual_v3/experiment_history.csv"
os.makedirs(os.path.dirname(output_file), exist_ok=True)

file_exists = os.path.exists(output_file)
fieldnames = list(run_data.keys())

with open(output_file, 'a', newline='') as f:
    writer = csv.DictWriter(f, fieldnames=fieldnames)
    if not file_exists:
        writer.writeheader()
    writer.writerow(run_data)

print(f"✅ Recovered run data saved to {output_file}")