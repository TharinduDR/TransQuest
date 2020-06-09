import csv
import os
import shutil

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from examples.wmt_2020.common.util.download import download_from_google_drive
from examples.wmt_2020.common.util.draw import draw_scatterplot, print_stat
from examples.wmt_2020.common.util.normalizer import fit, un_fit
from examples.wmt_2020.common.util.postprocess import format_submission
from examples.wmt_2020.common.util.reader import read_annotated_file, read_test_file
from examples.wmt_2020.ro_en.transformer_plus_config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, transformer_plus_config, SEED, \
    RESULT_FILE, RESULT_IMAGE, GOOGLE_DRIVE, DRIVE_FILE_ID, SUBMISSION_FILE
from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr
from transquest.algo.transformers.run_model import QuestModel
from transquest.algo.transformers_plus.preprocess import prepare_file, extend_file

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

TRAIN_FILE = "examples/ro_en/data/ro-en/train.roen.df.short.tsv"
DEV_FILE = "examples/ro_en/data/ro-en/dev.roen.df.short.tsv"
TEST_FILE = "examples/ro_en/data/ro-en/test20.roen.df.short.tsv"

train = read_annotated_file(TRAIN_FILE)
dev = read_annotated_file(DEV_FILE)
test = read_test_file(TEST_FILE)

train_embedding = prepare_file(train, "translation")
dev_embedding = prepare_file(dev, "translation")
test_embedding = prepare_file(test, "translation")

train_extended = extend_file(train_embedding, "translation", 'z_mean')
dev_extended = extend_file(dev_embedding, "translation", 'z_mean', train_embedding)
test_extended = extend_file(test_embedding, "translation", 'z_mean', train_embedding)

train_extended.to_csv("examples/ro_en/data/ro-en/train.roen.df.short.extend.tsv", sep='\t', quoting=csv.QUOTE_NONE, encoding="utf-8-sig")
dev_extended.to_csv("examples/ro_en/data/ro-en/dev.roen.df.short.extend.tsv", sep='\t', quoting=csv.QUOTE_NONE, encoding="utf-8-sig")
test_extended.to_csv("examples/ro_en/data/ro-en/test20.roen.df.short.extend.tsv", sep='\t', quoting=csv.QUOTE_NONE, encoding="utf-8-sig")

# train = train[['original', 'translation', 'z_mean']]
# dev = dev[['original', 'translation', 'z_mean']]
# test = test[['index', 'original', 'translation']]
#
# index = test['index'].to_list()
# train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
# dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
# test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()
#
# test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))
#
# train = fit(train, 'labels')
# dev = fit(dev, 'labels')
#
# assert (len(index) == 1000)
# if transformer_config["evaluate_during_training"]:
#     if transformer_config["n_fold"] > 1:
#         dev_preds = np.zeros((len(dev), transformer_config["n_fold"]))
#         test_preds = np.zeros((len(test), transformer_config["n_fold"]))
#         for i in range(transformer_config["n_fold"]):
#
#             if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
#                 shutil.rmtree(transformer_config['output_dir'])
#
#             model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
#                                args=transformer_config)
#             train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
#             model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
#                               mae=mean_absolute_error)
#             model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
#                                use_cuda=torch.cuda.is_available(), args=transformer_config)
#             result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
#                                                                         spearman_corr=spearman_corr,
#                                                                         mae=mean_absolute_error)
#             predictions, raw_outputs = model.predict(test_sentence_pairs)
#             dev_preds[:, i] = model_outputs
#             test_preds[:, i] = predictions
#
#         dev['predictions'] = dev_preds.mean(axis=1)
#         test['predictions'] = test_preds.mean(axis=1)
#
#     else:
#         model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
#                            args=transformer_config)
#         train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
#         model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
#                           mae=mean_absolute_error)
#         model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
#                            use_cuda=torch.cuda.is_available(), args=transformer_config)
#         result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
#                                                                     spearman_corr=spearman_corr,
#                                                                     mae=mean_absolute_error)
#         predictions, raw_outputs = model.predict(test_sentence_pairs)
#         dev['predictions'] = model_outputs
#         test['predictions'] = predictions
#
# else:
#     model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
#                        args=transformer_config)
#     model.train_model(train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
#     result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
#                                                                 spearman_corr=spearman_corr, mae=mean_absolute_error)
#     predictions, raw_outputs = model.predict(test_sentence_pairs)
#     dev['predictions'] = model_outputs
#     test['predictions'] = predictions
#
# dev = un_fit(dev, 'labels')
# dev = un_fit(dev, 'predictions')
# test = un_fit(test, 'predictions')
# dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
# draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "Romanian-English")
# print_stat(dev, 'labels', 'predictions')
# format_submission(df=test, index=index, language_pair="ro-en", method="TransQuest",
#                   path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE))
