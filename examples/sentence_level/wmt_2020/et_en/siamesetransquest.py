import csv
import logging
import math
import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from examples.sentence_level.wmt_2020.common.util.download import download_from_google_drive
from examples.sentence_level.wmt_2020.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2020.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2020.common.util.postprocess import format_submission
from examples.sentence_level.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2020.et_en.siamesetransquest_config import TEMP_DIRECTORY, GOOGLE_DRIVE, DRIVE_FILE_ID, MODEL_NAME, \
    siamesetransquest_config, SEED, RESULT_FILE, RESULT_IMAGE, SUBMISSION_FILE
from transquest.algo.sentence_level.siamesetransquest import LoggingHandler, SentencesDataset, \
    SiameseTransQuestModel
from transquest.algo.sentence_level.siamesetransquest import models, losses
from transquest.algo.sentence_level.siamesetransquest.evaluation import EmbeddingSimilarityEvaluator
from transquest.algo.sentence_level.siamesetransquest.readers import QEDataReader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

TRAIN_FILE = "examples/sentence_level/wmt_2020/et_en/data/et-en/train.eten.df.short.tsv"
DEV_FILE = "examples/sentence_level/wmt_2020/et_en/data/et-en/dev.eten.df.short.tsv"
TEST_FILE = "examples/sentence_level/wmt_2020/et_en/data/et-en/test20.eten.df.short.tsv"

train = read_annotated_file(TRAIN_FILE)
dev = read_annotated_file(DEV_FILE)
test = read_test_file(TEST_FILE)
index = test['index'].to_list()

train = train[['original', 'translation', 'z_mean']]
dev = dev[['original', 'translation', 'z_mean']]
test = test[['original', 'translation']]

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()

train = fit(train, 'labels')
dev = fit(dev, 'labels')

assert (len(index) == 1000)
if siamesetransquest_config["evaluate_during_training"]:
    if siamesetransquest_config["n_fold"] > 0:
        dev_preds = np.zeros((len(dev), siamesetransquest_config["n_fold"]))
        test_preds = np.zeros((len(test), siamesetransquest_config["n_fold"]))
        for i in range(siamesetransquest_config["n_fold"]):

            if os.path.exists(siamesetransquest_config['best_model_dir']) and os.path.isdir(
                    siamesetransquest_config['best_model_dir']):
                shutil.rmtree(siamesetransquest_config['best_model_dir'])

            if os.path.exists(siamesetransquest_config['cache_dir']) and os.path.isdir(
                    siamesetransquest_config['cache_dir']):
                shutil.rmtree(siamesetransquest_config['cache_dir'])

            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            model = SiameseTransQuestModel(MODEL_NAME)
            model.train_model(train_df, eval_df)

            model = SiameseTransQuestModel(siamesetransquest_config['best_model_dir'])
            dev_preds[:, i] = model.predict(dev_sentence_pairs)
            test_preds[:, i] = model.predict(test_sentence_pairs)

        dev['predictions'] = dev_preds.mean(axis=1)
        test['predictions'] = test_preds.mean(axis=1)

dev = un_fit(dev, 'labels')
dev = un_fit(dev, 'predictions')
test = un_fit(test, 'predictions')
dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "Estonian-English")
print_stat(dev, 'labels', 'predictions')
format_submission(df=test, index=index, language_pair="et-en", method="SiameseTransQuest",
                  path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))
