import csv
import logging
import math
import os
import random
import shutil
import time

import numpy as np
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader

from examples.sentence_level.wmt_2020.common.util.download import download_from_google_drive
from examples.sentence_level.wmt_2020.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2020.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2020.common.util.postprocess import format_submission
from examples.sentence_level.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2020.ro_en.siamesetransquest_config import TEMP_DIRECTORY, GOOGLE_DRIVE, DRIVE_FILE_ID, MODEL_NAME, \
    siamesetransquest_config, SEED, RESULT_FILE, RESULT_IMAGE, SUBMISSION_FILE

from transquest.algo.sentence_level.siamesetransquest.evaluation.embedding_similarity_evaluator import \
    EmbeddingSimilarityEvaluator
from transquest.algo.sentence_level.siamesetransquest.logging_handler import LoggingHandler
from transquest.algo.sentence_level.siamesetransquest.losses.cosine_similarity_loss import CosineSimilarityLoss
from transquest.algo.sentence_level.siamesetransquest.models.pooling import Pooling
from transquest.algo.sentence_level.siamesetransquest.models.transformer import Transformer
from transquest.algo.sentence_level.siamesetransquest.readers.input_example import InputExample
from transquest.algo.sentence_level.siamesetransquest.run_model import SiameseTransQuestModel

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

TRAIN_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/train.roen.df.short.tsv"
DEV_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/dev.roen.df.short.tsv"
TEST_FILE = "examples/sentence_level/wmt_2020/ro_en/data/ro-en/test20.roen.df.short.tsv"

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

            os.makedirs(siamesetransquest_config['cache_dir'])

            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)


            word_embedding_model = Transformer(MODEL_NAME, max_seq_length=siamesetransquest_config[
                'max_seq_length'])

            pooling_model = Pooling(word_embedding_model.get_word_embedding_dimension(),
                                    pooling_mode_mean_tokens=True,
                                    pooling_mode_cls_token=False,
                                    pooling_mode_max_tokens=False)

            model = SiameseTransQuestModel(modules=[word_embedding_model, pooling_model])

            train_samples = []
            eval_samples = []
            dev_samples = []
            test_samples = []

            for index, row in train_df.iterrows():
                score = float(row["labels"])
                inp_example = InputExample(texts=[row['text_a'], row['text_b']], label=score)
                train_samples.append(inp_example)

            for index, row in eval_df.iterrows():
                score = float(row["labels"])
                inp_example = InputExample(texts=[row['text_a'], row['text_b']], label=score)
                eval_samples.append(inp_example)

            train_dataloader = DataLoader(train_samples, shuffle=True, batch_size=siamesetransquest_config['train_batch_size'])
            train_loss = CosineSimilarityLoss(model=model)

            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_samples, name='eval')
            warmup_steps = math.ceil(len(train_dataloader) * siamesetransquest_config["num_train_epochs"] * 0.1)

            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=siamesetransquest_config['num_train_epochs'],
                      evaluation_steps=100,
                      optimizer_params={'lr': siamesetransquest_config["learning_rate"],
                                        'eps': siamesetransquest_config["adam_epsilon"],
                                        'correct_bias': False},
                      warmup_steps=warmup_steps,
                      output_path=siamesetransquest_config['best_model_dir'])

            model = SiameseTransQuestModel(siamesetransquest_config['best_model_dir'])

            for index, row in dev.iterrows():
                score = float(row["labels"])
                inp_example = InputExample(texts=[row['text_a'], row['text_b']], label=score)
                dev_samples.append(inp_example)

            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(dev_samples, name='dev')
            model.evaluate(evaluator,
                           output_path=os.path.join(siamesetransquest_config['cache_dir'], "dev_result.txt"))

            # test_data = SentencesDataset(examples=sts_reader.get_examples("test.tsv", test_file=True), model=model)
            # test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)

            for index, row in test.iterrows():
                score = random.uniform(0, 1)
                inp_example = InputExample(texts=[row['text_a'], row['text_b']], label=score)
                test_samples.append(inp_example)

            evaluator = EmbeddingSimilarityEvaluator.from_input_examples(test_samples, name='dev')

            model.evaluate(evaluator,
                           output_path=os.path.join(siamesetransquest_config['cache_dir'], "test_result.txt"),
                           verbose=False)

            with open(os.path.join(siamesetransquest_config['cache_dir'], "dev_result.txt")) as f:
                dev_preds[:, i] = list(map(float, f.read().splitlines()))

            with open(os.path.join(siamesetransquest_config['cache_dir'], "test_result.txt")) as f:
                test_preds[:, i] = list(map(float, f.read().splitlines()))

        dev['predictions'] = dev_preds.mean(axis=1)
        test['predictions'] = test_preds.mean(axis=1)

dev = un_fit(dev, 'labels')
dev = un_fit(dev, 'predictions')
test = un_fit(test, 'predictions')
dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "Romanian-English")
print_stat(dev, 'labels', 'predictions')
format_submission(df=test, index=index, language_pair="ro-en", method="SiameseTransQuest",
                  path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))
