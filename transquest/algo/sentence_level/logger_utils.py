import os
import csv
import datetime
import logging
import sys
import pandas as pd

logger = logging.getLogger(__name__)

class ConsoleFileHandler:
    """
    Context manager to redirect all logging and stdout/stderr to a file
    for the duration of a training run.
    """
    def __init__(self, log_file):
        self.log_file = log_file
        self.file_handler = None
        self.root_logger = logging.getLogger()

    def __enter__(self):
        # Create handler
        self.file_handler = logging.FileHandler(self.log_file, mode='w')
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.root_logger.addHandler(self.file_handler)
        
        # Redirect print() to logger
        self.original_stdout = sys.stdout
        sys.stdout = self
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        # Cleanup
        sys.stdout = self.original_stdout
        self.root_logger.removeHandler(self.file_handler)
        self.file_handler.close()

    def write(self, message):
        if message.strip():
            # Log to file and console
            self.root_logger.info(message.strip())
        # Also write to original stdout so user sees it
        self.original_stdout.write(message)

    def flush(self):
        self.original_stdout.flush()

class ExperimentLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self.log_file = os.path.join(output_dir, "experiment_history.csv")
        
    def log_run(self, config, hyperparams, dev_metrics, test_metrics_map=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        
        # Base Row
        row = {
            "Timestamp": timestamp,
            "Exp_ID": config.get("exp_id", "0"),
            "Model": config.get("model_name", "unknown"),
            "Loss": config.get("loss_type", "unknown"),
            "Use_FiLM": config.get("use_film", False),
            "Curvature_C": f"{hyperparams.get('final_curvature', 0.0):.3f}",
            "Epochs": hyperparams.get("num_train_epochs", 0),
            "Batch": hyperparams.get("train_batch_size", 0),
            "LR": hyperparams.get("learning_rate", 0),
            "Dev_SPA": f"{dev_metrics.get('spa', 0):.4f}",
            "Dev_Spearman": f"{dev_metrics.get('spearman', 0):.4f}",
            "Dev_Pearson": f"{dev_metrics.get('pearson', 0):.4f}",
            "Dev_Kendall": f"{dev_metrics.get('kendall', 0):.4f}",
            "Dev_MSE": f"{dev_metrics.get('mse', 0):.4f}"
        }
        
        # Dynamic Test Columns
        if test_metrics_map:
            for lang, metrics in test_metrics_map.items():
                row[f"Test_{lang}_SPA"] = f"{metrics.get('spa', 0):.4f}"
                row[f"Test_{lang}_Spearman"] = f"{metrics.get('spearman', 0):.4f}"

        # Header Management
        new_keys = list(row.keys())
        if os.path.exists(self.log_file):
            try:
                df = pd.read_csv(self.log_file)
                existing = list(df.columns)
                missing = [k for k in new_keys if k not in existing]
                if missing:
                    for k in missing: df[k] = ""
                    df.to_csv(self.log_file, index=False)
            except: pass

        # Append
        with open(self.log_file, 'a', newline='') as f:
            is_empty = os.path.getsize(self.log_file) == 0
            if not is_empty:
                 with open(self.log_file, 'r') as r:
                     reader = csv.reader(r)
                     fieldnames = next(reader)
            else: fieldnames = new_keys

            writer = csv.DictWriter(f, fieldnames=fieldnames, extrasaction='ignore')
            if is_empty: writer.writeheader()
            writer.writerow(row)