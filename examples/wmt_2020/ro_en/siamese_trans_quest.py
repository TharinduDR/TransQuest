"""
This examples trains BERT for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.
"""
import shutil
from datetime import datetime

import torch
from sklearn.model_selection import train_test_split
from torch.utils.data import DataLoader
import math
import logging
import os
import numpy as np
import csv

from examples.wmt_2020.common.util.download import download_from_google_drive
from examples.wmt_2020.common.util.draw import draw_scatterplot, print_stat
from examples.wmt_2020.common.util.normalizer import fit, un_fit
from examples.wmt_2020.common.util.postprocess import format_submission
from examples.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from examples.wmt_2020.ro_en.siamese_transformer_config import TEMP_DIRECTORY, GOOGLE_DRIVE, DRIVE_FILE_ID, MODEL_NAME, \
    siamese_transformer_config, SEED, RESULT_FILE, RESULT_IMAGE, SUBMISSION_FILE
from transquest.algo.siamese_transformers import losses, models, LoggingHandler, SentencesDataset, SentenceTransformer
from transquest.algo.siamese_transformers.evaluation.EmbeddingSimilarityEvaluator import EmbeddingSimilarityEvaluator
from transquest.algo.siamese_transformers.predict import sentence_pairs_predict

from transquest.algo.siamese_transformers.readers import STSDataReader

logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

TRAIN_FILE = "examples/wmt_2020/ro_en/data/ro-en/train.roen.df.short.tsv"
DEV_FILE = "examples/wmt_2020/ro_en/data/ro-en/dev.roen.df.short.tsv"
TEST_FILE = "examples/wmt_2020/ro_en/data/ro-en/test20.roen.df.short.tsv"

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

# dev_sentence_pairs = list(map(list, zip(dev['text_a'].to_list(), test['text_b'].to_list())))
# test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

train = fit(train, 'labels')
dev = fit(dev, 'labels')
test["labels"] = 0

assert(len(index) == 1000)
if siamese_transformer_config["evaluate_during_training"]:
    if siamese_transformer_config["n_fold"] > 0:
        dev_preds = np.zeros((len(dev), siamese_transformer_config["n_fold"]))
        test_preds = np.zeros((len(test), siamese_transformer_config["n_fold"]))
        for i in range(siamese_transformer_config["n_fold"]):

            if os.path.exists(siamese_transformer_config['output_dir']) and os.path.isdir(siamese_transformer_config['output_dir']):
                shutil.rmtree(siamese_transformer_config['output_dir'])

            if os.path.exists(siamese_transformer_config['cache_dir']) and os.path.isdir(siamese_transformer_config['cache_dir']):
                shutil.rmtree(siamese_transformer_config['cache_dir'])

            os.makedirs(siamese_transformer_config['cache_dir'])

            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            train_df.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "train.tsv"), header=True, sep='\t',
                         index=False, quoting=csv.QUOTE_NONE)
            eval_df.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "eval_df.tsv"), header=True, sep='\t',
                         index=False, quoting=csv.QUOTE_NONE)
            dev.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "dev.tsv"), header=True, sep='\t',
                           index=False, quoting=csv.QUOTE_NONE)
            test.to_csv(os.path.join(siamese_transformer_config['cache_dir'], "test.tsv"), header=True, sep='\t',
                       index=False, quoting=csv.QUOTE_NONE)

            sts_reader = STSDataReader(siamese_transformer_config['cache_dir'], s1_col_idx=0, s2_col_idx=1, score_col_idx=2,
                                       normalize_scores=False, min_score=0, max_score=1, header=True)

            word_embedding_model = models.Transformer(MODEL_NAME)

            pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                                           pooling_mode_mean_tokens=True,
                                           pooling_mode_cls_token=False,
                                           pooling_mode_max_tokens=False)

            model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
            train_data = SentencesDataset(sts_reader.get_examples('train.tsv'), model)
            train_dataloader = DataLoader(train_data, shuffle=True, batch_size=siamese_transformer_config['train_batch_size'])
            train_loss = losses.CosineSimilarityLoss(model=model)

            eval_data = SentencesDataset(examples=sts_reader.get_examples('eval_df.tsv'), model=model)
            eval_dataloader = DataLoader(eval_data, shuffle=False, batch_size=siamese_transformer_config['train_batch_size'])
            evaluator = EmbeddingSimilarityEvaluator(eval_dataloader)

            warmup_steps = math.ceil(len(train_data) * siamese_transformer_config["num_train_epochs"] / siamese_transformer_config['train_batch_size'] * 0.1)

            model.fit(train_objectives=[(train_dataloader, train_loss)],
                      evaluator=evaluator,
                      epochs=siamese_transformer_config['num_train_epochs'],
                      evaluation_steps=100,
                      optimizer_params={'lr':  siamese_transformer_config["learning_rate"],
                                        'eps':  siamese_transformer_config["adam_epsilon"],
                                        'correct_bias': False},
                      warmup_steps=warmup_steps,
                      output_path=siamese_transformer_config['output_dir'])

            model = SentenceTransformer(siamese_transformer_config['output_dir'])

            dev_data = SentencesDataset(examples=sts_reader.get_examples("dev.tsv"), model=model)
            dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=8)
            evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
            model.evaluate(evaluator, result_path=os.path.join(siamese_transformer_config['cache_dir'], "dev_result.txt"))

            test_data = SentencesDataset(examples=sts_reader.get_examples("test.tsv"), model=model)
            test_dataloader = DataLoader(test_data, shuffle=False, batch_size=8)
            evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
            model.evaluate(evaluator,
                           result_path=os.path.join(siamese_transformer_config['cache_dir'], "test_result.txt"), verbose=False)

            with open(os.path.join(siamese_transformer_config['cache_dir'], "dev_result.txt")) as f:
                dev_preds[:, i] = list(map(float, f.read().splitlines()))

            with open(os.path.join(siamese_transformer_config['cache_dir'], "test_result.txt")) as f:
                test_preds[:, i] = list(map(float, f.read().splitlines()))

            del model
            torch.cuda.empty_cache()

        dev['predictions'] = dev_preds.mean(axis=1)
        test['predictions'] = test_preds.mean(axis=1)


dev = un_fit(dev, 'labels')
dev = un_fit(dev, 'predictions')
test = un_fit(test, 'predictions')
dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "Romanian-English")
print_stat(dev, 'labels', 'predictions')
format_submission(df=test, index=index, language_pair="ro-en", method="TransQuest", path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))

# # Read the dataset
# train_batch_size = 16
# num_epochs = 4
# model_save_path = 'output/training_stsbenchmark_bert-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
# sts_reader = STSDataReader('examples/wmt_2020/ro_en/data/ro-en/', s1_col_idx=1, s2_col_idx=2, score_col_idx=6, normalize_scores=True, header=True)
#
# # Use BERT for mapping tokens to embeddings
# word_embedding_model = models.Transformer('xlm-roberta-base')
#
# # Apply mean pooling to get one fixed sized sentence vector
# pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
#                                pooling_mode_mean_tokens=True,
#                                pooling_mode_cls_token=False,
#                                pooling_mode_max_tokens=False)
#
# model = SentenceTransformer(modules=[word_embedding_model, pooling_model])
#
# # Convert the dataset to a DataLoader ready for training
# logging.info("Read STSbenchmark train dataset")
# train_data = SentencesDataset(sts_reader.get_examples('train.roen.df.short.tsv'), model)
# train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
# train_loss = losses.CosineSimilarityLoss(model=model)
#
#
# logging.info("Read STSbenchmark dev dataset")
# dev_data = SentencesDataset(examples=sts_reader.get_examples('dev.roen.df.short.tsv'), model=model)
# dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
# evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)
#
#
# # Configure the training. We skip evaluation in this example
# warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
# logging.info("Warmup-steps: {}".format(warmup_steps))
#
#
# # Train the model
# model.fit(train_objectives=[(train_dataloader, train_loss)],
#           evaluator=evaluator,
#           epochs=num_epochs,
#           evaluation_steps=1000,
#           warmup_steps=warmup_steps,
#           output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

# model = SentenceTransformer(model_save_path)
# test = read_test_file("examples/wmt_2020/ro_en/data/ro-en/test20.roen.df.short.tsv")
# test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
# predictions = model.predict(test_sentence_pairs)
# test['predictions'] = predictions


# test_data = SentencesDataset(examples=sts_reader.get_examples("test20.roen.df.short.tsv"), model=model)
# test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
# evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
# model.evaluate(evaluator)
