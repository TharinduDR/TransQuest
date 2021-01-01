import os
import shutil

import numpy as np
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from examples.sentence_level.wmt_2019.common.util.download import download_from_google_drive
from examples.sentence_level.wmt_2019.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2019.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2019.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2019.en_de.monotransquest_config import TEMP_DIRECTORY, GOOGLE_DRIVE, DRIVE_FILE_ID, MODEL_NAME, \
    monotransquest_config, MODEL_TYPE, SEED, RESULT_FILE, RESULT_IMAGE
from transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

TRAIN_FOLDER = "examples/sentence_level/wmt_2019/en_de/data/en-de/train"
DEV_FOLDER = "examples/sentence_level/wmt_2019/en_de/data/en-de/dev"
TEST_FOLDER = "examples/sentence_level/wmt_2019/en_de/data/en-de/test-blind"
ANNOTATED_TEST_FOLDER = "examples/sentence_level/wmt_2019/en_de/data/en-de/test"

train = read_annotated_file(path=TRAIN_FOLDER, original_file="train.src", translation_file="train.mt", hter_file="train.hter")
dev = read_annotated_file(path=DEV_FOLDER, original_file="dev.src", translation_file="dev.mt", hter_file="dev.hter")
test = read_test_file(path=TEST_FOLDER, original_file="test.src", translation_file="test.mt")
annotated_test = read_annotated_file(path=ANNOTATED_TEST_FOLDER, original_file="test.src", translation_file="test.mt", hter_file="test.hter")

train = train[['original', 'translation', 'hter']]
dev = dev[['original', 'translation', 'hter']]
test = test[['index', 'original', 'translation']]
annotated_test = annotated_test[['original', 'translation', 'hter']]

index = test['index'].to_list()
train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
dev = dev.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()
annotated_test = annotated_test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()

test_sentence_pairs = list(map(list, zip(test['text_a'].to_list(), test['text_b'].to_list())))

train = fit(train, 'labels')
dev = fit(dev, 'labels')

if monotransquest_config["evaluate_during_training"]:
    if monotransquest_config["n_fold"] > 1:
        dev_preds = np.zeros((len(dev), monotransquest_config["n_fold"]))
        test_preds = np.zeros((len(test), monotransquest_config["n_fold"]))
        for i in range(monotransquest_config["n_fold"]):

            if os.path.exists(monotransquest_config['output_dir']) and os.path.isdir(monotransquest_config['output_dir']):
                shutil.rmtree(monotransquest_config['output_dir'])

            model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                        args=monotransquest_config)
            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED*i)
            model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
            model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1, use_cuda=torch.cuda.is_available(), args=monotransquest_config)
            result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr,
                                                                        mae=mean_absolute_error)
            predictions, raw_outputs = model.predict(test_sentence_pairs)
            dev_preds[:, i] = model_outputs
            test_preds[:, i] = predictions

        dev['predictions'] = dev_preds.mean(axis=1)
        test['predictions'] = test_preds.mean(axis=1)

    else:
        model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                    args=monotransquest_config)
        train, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
        model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1,
                                    use_cuda=torch.cuda.is_available(), args=monotransquest_config)
        result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
        predictions, raw_outputs = model.predict(test_sentence_pairs)
        dev['predictions'] = model_outputs
        test['predictions'] = predictions

else:
    model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                args=monotransquest_config)
    model.train_model(train, pearson_corr=pearson_corr, spearman_corr=spearman_corr, mae=mean_absolute_error)
    result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                spearman_corr=spearman_corr, mae=mean_absolute_error)
    predictions, raw_outputs = model.predict(test_sentence_pairs)
    dev['predictions'] = model_outputs
    test['predictions'] = predictions

dev = un_fit(dev, 'labels')
dev = un_fit(dev, 'predictions')
test = un_fit(test, 'predictions')
dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(dev, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), "English-German")
print_stat(dev, 'labels', 'predictions')
annotated_test["predictions"] = test['predictions']
print_stat(annotated_test, 'labels', 'predictions')
