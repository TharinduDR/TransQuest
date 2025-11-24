import os
import csv
import datetime
import logging
import sys
import pandas as pd

logger = logging.getLogger(__name__)

class ConsoleFileHandler:
    def __init__(self, log_file):
        self.log_file = log_file
        self.file_handler = None
        self.root_logger = logging.getLogger()
        self.original_stdout = sys.stdout
        self.original_stderr = sys.stderr

    def __enter__(self):
        self.file_handler = logging.FileHandler(self.log_file, mode='w')
        self.file_handler.setFormatter(logging.Formatter('%(asctime)s - %(message)s'))
        self.root_logger.addHandler(self.file_handler)
        sys.stdout = LoggerWriter(self.original_stdout, self.root_logger, logging.INFO)
        sys.stderr = LoggerWriter(self.original_stderr, self.root_logger, logging.ERROR)
        return self

    def __exit__(self, exc_type, exc_val, exc_tb):
        sys.stdout = self.original_stdout
        sys.stderr = self.original_stderr
        self.root_logger.removeHandler(self.file_handler)
        self.file_handler.close()

class LoggerWriter:
    def __init__(self, writer, logger, level):
        self.writer = writer
        self.logger = logger
        self.level = level
    def write(self, message):
        is_progress = '\r' in message or '%|' in message
        if message.strip() and not is_progress:
            self.logger.log(self.level, message.strip())
        self.writer.write(message)
    def flush(self):
        self.writer.flush()

class ExperimentLogger:
    def __init__(self, output_dir):
        self.output_dir = output_dir
        if not os.path.exists(output_dir): os.makedirs(output_dir)
        self.log_file = os.path.join(output_dir, "experiment_history.csv")
        
    def log_run(self, config, hyperparams, dev_metrics, test_metrics_map=None):
        timestamp = datetime.datetime.now().strftime("%Y-%m-%d %H:%M:%S")
        row = {
            "Timestamp": timestamp,
            "Exp_ID": config.get("exp_id", "0"),
            "Model": config.get("model_name", "unknown"),
            "Loss": config.get("loss_type", "unknown"),
            "Use_FiLM": config.get("use_film", False),
            "Curvature_Final": f"{hyperparams.get('final_curvature', 0.0):.3f}",
            "Epochs": hyperparams.get("num_train_epochs", 0),
            "Batch": hyperparams.get("train_batch_size", 0),
            "LR": hyperparams.get("learning_rate", 0),
            "Dev_SPA": f"{dev_metrics.get('spa', 0):.4f}",
            "Dev_Spearman": f"{dev_metrics.get('spearman', 0):.4f}",
            "Dev_MSE": f"{dev_metrics.get('mse', 0):.4f}"
        }
        
        if test_metrics_map:
            for lang, metrics in test_metrics_map.items():
                row[f"Test_{lang}_SPA"] = f"{metrics.get('spa', 0):.4f}"
                row[f"Test_{lang}_Spearman"] = f"{metrics.get('spearman', 0):.4f}"

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

        with open(self.log_file, 'a', newline='') as f:
            is_empty = os.path.getsize(self.log_file) == 0
            writer = csv.DictWriter(f, fieldnames=new_keys, extrasaction='ignore')
            if is_empty: writer.writeheader()
            writer.writerow(row)
        logger.info(f"✅ Experiment logged to {self.log_file}")