import os
import shutil

import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split
import numpy as np

from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr
from examples.common.util.download import download_from_google_drive
from examples.common.util.draw import draw_scatterplot
from examples.common.util.normalizer import fit, un_fit
from examples.ro_en.transformer_config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, transformer_config, SEED, \
    RESULT_FILE, RESULT_IMAGE, GOOGLE_DRIVE, DRIVE_FILE_ID
from transquest.algo.transformers.run_model import QuestModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

TRAIN_FILE = "data/ro-en/train.roen.df.short.tsv"
TEST_FILE = "data/ro-en/dev.roen.df.short.tsv"

train = pd.read_csv(TRAIN_FILE, sep='\t')
test = pd.read_csv(TEST_FILE, sep='\t')

train = train[['original', 'translation', 'z_mean']]
test = test[['original', 'translation', 'z_mean']]

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()

train = fit(train, 'labels')
test = fit(test, 'labels')


if transformer_config["evaluate_during_training"]:
    if transformer_config["n_fold"] > 1:
        test_preds = np.zeros((len(test), transformer_config["n_fold"]))
        for i in range(transformer_config["n_fold"]):

            if os.path.exists(transformer_config['output_dir']) and os.path.isdir(transformer_config['output_dir']):
                shutil.rmtree(transformer_config['output_dir'])

            model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                               args=transformer_config)
            train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED*i)
            model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
            model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1, use_cuda=torch.cuda.is_available(), args=transformer_config)
            result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr,
                                                                        mae=mean_absolute_error)
            test_preds[:, i] = model_outputs

        test['predictions'] = test_preds.mean(axis=1)

    else:
        model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                           args=transformer_config)
        train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
        model = QuestModel(MODEL_TYPE, transformer_config["best_model_dir"], num_labels=1,
                           use_cuda=torch.cuda.is_available(), args=transformer_config)
        result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
        test['predictions'] = model_outputs


else:
    model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                       args=transformer_config)
    model.train_model(train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    result, model_outputs, wrong_predictions = model.eval_model(test, pearson_corr=pearson_corr,
                                                                spearman_corr=spearman_corr, mae=mean_absolute_error)
    test['predictions'] = model_outputs


test = un_fit(test, 'labels')
test = un_fit(test, 'predictions')
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(test, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), MODEL_TYPE + " " + MODEL_NAME)
