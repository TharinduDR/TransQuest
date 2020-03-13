import logging
import os

import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from algo.evaluation import pearson_corr, spearman_corr
from algo.run_model import QuestModel
from config.global_config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, global_config, SEED, RESULT_FILE
from util.reader import read_files

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)


TRAIN_SRC_FILE = "data/Quality_Estimation/2018/sentence_level/en_lv/train.smt.src"
TRAIN_TARGET_FILE = "data/Quality_Estimation/2018/sentence_level/en_lv/train.smt.mt"
TRAIN_HTER_FILE = "data/Quality_Estimation/2018/sentence_level/en_lv/train.smt.hter"

TEST_SRC_FILE = "data/Quality_Estimation/2018/sentence_level/en_lv/dev.smt.src"
TEST_TARGET_FILE = "data/Quality_Estimation/2018/sentence_level/en_lv/dev.smt.mt"
TEST_HTER_FILE = "data/Quality_Estimation/2018/sentence_level/en_lv/dev.smt.hter"


model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                   args=global_config)

train = read_files(TRAIN_SRC_FILE, TRAIN_TARGET_FILE, TRAIN_HTER_FILE)
test = read_files(TEST_SRC_FILE, TEST_TARGET_FILE, TEST_HTER_FILE)

logging.info("Started Training")

if global_config["evaluate_during_training"]:
    train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
    model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                      mae=mean_absolute_error)

else:
    model.train_model(train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)

logging.info("Finished Training")

# Evaluate the model
result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                            spearman_corr=spearman_corr, mae=mean_absolute_error)

test['predictions'] = model_outputs
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
