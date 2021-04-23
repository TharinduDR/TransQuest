import csv
import logging
import math
import os
import shutil

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from examples.sentence_level.wmt_2020_task2.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2020_task2.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2020_task2.common.util.postprocess import format_submission
from examples.sentence_level.wmt_2020_task2.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2020_task2.en_de.siamesetransquest_config import TEMP_DIRECTORY, \
    MODEL_NAME, siamesetransquest_config, SEED, RESULT_FILE, SUBMISSION_FILE, RESULT_IMAGE

from transquest.algo.sentence_level.siamesetransquest.logging_handler import LoggingHandler
from transquest.algo.sentence_level.siamesetransquest.run_model import SiameseTransQuestModel

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)


TRAIN_FOLDER = "examples/sentence_level/wmt_2020_task2/en_de/data/en-de/train"
DEV_FOLDER = "examples/sentence_level/wmt_2020_task2/en_de/data/en-de/dev"
TEST_FOLDER = "examples/sentence_level/wmt_2020_task2/en_de/data/en-de/test-blind"

train = read_annotated_file(path=TRAIN_FOLDER, original_file="train.src", translation_file="train.mt", hter_file="train.hter")
dev = read_annotated_file(path=DEV_FOLDER, original_file="dev.src", translation_file="dev.mt", hter_file="dev.hter")
test = read_test_file(path=TEST_FOLDER, original_file="test.src", translation_file="test.mt")

index = test['index'].to_list()

train = train[['original', 'translation', 'hter']]
dev = dev[['original', 'translation', 'hter']]
test = test[['original', 'translation']]

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()

dev_sentence_pairs = list(map(list, zip(dev['text_a'].to_list(), dev['text_b'].to_list())))
test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

train = fit(train, 'labels')
dev = fit(dev, 'labels')

assert(len(index) == 1000)
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
draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "English-German")
print_stat(dev, 'labels', 'predictions')
format_submission(df=test, index=index, language_pair="en-de", method="SiameseTransQuest",
                  path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))
