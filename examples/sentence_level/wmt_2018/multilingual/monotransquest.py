import os
import shutil

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from examples.sentence_level.wmt_2018.common.util.download import download_from_google_drive
from examples.sentence_level.wmt_2018.common.util.draw import draw_scatterplot, print_stat
from examples.sentence_level.wmt_2018.common.util.normalizer import fit, un_fit
from examples.sentence_level.wmt_2018.common.util.postprocess import format_submission
from examples.sentence_level.wmt_2018.common.util.reader import read_annotated_file, read_test_file
from examples.sentence_level.wmt_2018.multilingual.monotransquest_config import TEMP_DIRECTORY, GOOGLE_DRIVE, DRIVE_FILE_ID, MODEL_NAME, \
    monotransquest_config, MODEL_TYPE, SEED, RESULT_FILE, SUBMISSION_FILE
from transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr
from transquest.algo.sentence_level.monotransquest.run_model import MonoTransQuestModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

if GOOGLE_DRIVE:
    download_from_google_drive(DRIVE_FILE_ID, MODEL_NAME)

languages = {
    # "DE-EN": ["examples/sentence_level/wmt_2018/de_en/data/de_en/",
    #           "examples/sentence_level/wmt_2018/de_en/data/de_en/",
    #           "examples/sentence_level/wmt_2018/de_en/data/de_en/",
    #           "smt"],

    "EN-ZH": ["examples/sentence_level/wmt_2020_task2/en_zh/data/en-zh/train",
              "examples/sentence_level/wmt_2020_task2/en_zh/data/en-zh/dev",
              "examples/sentence_level/wmt_2020_task2/en_zh/data/en-zh/test-blind",
              ""],

    "EN-CS": ["examples/sentence_level/wmt_2018/en_cs/data/en_cs/",
              "examples/sentence_level/wmt_2018/en_cs/data/en_cs/",
              "examples/sentence_level/wmt_2018/en_cs/data/en_cs/",
              "smt"],

    "EN-DE-NMT": ["examples/sentence_level/wmt_2020_task2/en_de/data/en-de/train",
              "examples/sentence_level/wmt_2020_task2/en_de/data/en-de/dev",
              "examples/sentence_level/wmt_2020_task2/en_de/data/en-de/test-blind",
               ""],

    "EN-DE-SMT": ["examples/sentence_level/wmt_2018/en_de/data/en_de",
              "examples/sentence_level/wmt_2018/en_de/data/en_de",
              "examples/sentence_level/wmt_2018/en_de/data/en_de",
                  "smt"],

    "EN-RU": ["examples/sentence_level/wmt_2019/en_ru/data/en-ru/train",
              "examples/sentence_level/wmt_2019/en_ru/data/en-ru/dev",
              "examples/sentence_level/wmt_2019/en_ru/data/en-ru/test-blind",
              ""],

    "EN-LV-NMT": ["examples/sentence_level/wmt_2018/en_lv/data/en_lv",
              "examples/sentence_level/wmt_2018/en_lv/data/en_lv",
              "examples/sentence_level/wmt_2018/en_lv/data/en_lv",
                  "nmt"],

    "EN-LV-SMT": ["examples/sentence_level/wmt_2018/en_lv/data/en_lv",
                  "examples/sentence_level/wmt_2018/en_lv/data/en_lv",
                  "examples/sentence_level/wmt_2018/en_lv/data/en_lv",
                  "smt"],

}

train_list = []
dev_list = []
test_list = []
index_list = []
test_sentence_pairs_list = []

for key, value in languages.items():

    if value[3] == "nmt":
        train_temp = read_annotated_file(path=value[0], original_file="train.nmt.src", translation_file="train.nmt.mt",
                                    hter_file="train.nmt.hter")
        dev_temp = read_annotated_file(path=value[1], original_file="dev.nmt.src", translation_file="dev.nmt.mt",
                                  hter_file="dev.nmt.hter")
        test_temp = read_test_file(path=value[2], original_file="test.nmt.src", translation_file="test.nmt.mt")

    elif value[3] == "smt":
        train_temp = read_annotated_file(path=value[0], original_file="train.smt.src", translation_file="train.smt.mt",
                                         hter_file="train.smt.hter")
        dev_temp = read_annotated_file(path=value[1], original_file="dev.smt.src", translation_file="dev.smt.mt",
                                       hter_file="dev.smt.hter")
        test_temp = read_test_file(path=value[2], original_file="test.smt.src", translation_file="test.smt.mt")

    else:
        train_temp = read_annotated_file(path=value[0], original_file="train.src", translation_file="train.mt",
                                         hter_file="train.hter")
        dev_temp = read_annotated_file(path=value[1], original_file="dev.src", translation_file="dev.mt",
                                       hter_file="dev.hter")
        test_temp = read_test_file(path=value[2], original_file="test.src", translation_file="test.mt")

    train_temp = train_temp[['original', 'translation', 'hter']]
    dev_temp = dev_temp[['original', 'translation', 'hter']]
    test_temp = test_temp[['index', 'original', 'translation']]


    index_temp = test_temp['index'].to_list()
    train_temp = train_temp.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
    dev_temp = dev_temp.rename(columns={'original': 'text_a', 'translation': 'text_b', 'hter': 'labels'}).dropna()
    test_temp = test_temp.rename(columns={'original': 'text_a', 'translation': 'text_b'}).dropna()

    test_sentence_pairs_temp = list(map(list, zip(test_temp['text_a'].to_list(), test_temp['text_b'].to_list())))

    train_temp = fit(train_temp, 'labels')
    dev_temp = fit(dev_temp, 'labels')

    train_list.append(train_temp)
    dev_list.append(dev_temp)
    test_list.append(test_temp)
    index_list.append(index_temp)
    test_sentence_pairs_list.append(test_sentence_pairs_temp)

train = pd.concat(train_list)

if monotransquest_config["evaluate_during_training"]:
    if monotransquest_config["n_fold"] > 1:
        dev_preds_list = []
        test_preds_list = []

        for dev, test in zip(dev_list, test_list):
            dev_preds = np.zeros((len(dev), monotransquest_config["n_fold"]))
            test_preds = np.zeros((len(test), monotransquest_config["n_fold"]))

            dev_preds_list.append(dev_preds)
            test_preds_list.append(test_preds)

        for i in range(monotransquest_config["n_fold"]):
            if os.path.exists(monotransquest_config['output_dir']) and os.path.isdir(monotransquest_config['output_dir']):
                shutil.rmtree(monotransquest_config['output_dir'])

            model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                        args=monotransquest_config)
            train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED * i)
            model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                              mae=mean_absolute_error)
            model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1,
                                        use_cuda=torch.cuda.is_available(), args=monotransquest_config)

            for dev, test_sentence_pairs, dev_preds, test_preds in zip(dev_list, test_sentence_pairs_list, dev_preds_list, test_preds_list):
                result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                        spearman_corr=spearman_corr,
                                                                        mae=mean_absolute_error)
                predictions, raw_outputs = model.predict(test_sentence_pairs)
                dev_preds[:, i] = model_outputs
                test_preds[:, i] = predictions

        for dev, dev_preds, test, test_preds in zip(dev_list, dev_preds_list, test_list, test_preds_list):
            dev['predictions'] = dev_preds.mean(axis=1)
            test['predictions'] = test_preds.mean(axis=1)

    else:
        model = MonoTransQuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                                    args=monotransquest_config)
        train_df, eval_df = train_test_split(train, test_size=0.1, random_state=SEED)
        model.train_model(train_df, eval_df=eval_df, pearson_corr=pearson_corr, spearman_corr=spearman_corr,
                          mae=mean_absolute_error)
        model = MonoTransQuestModel(MODEL_TYPE, monotransquest_config["best_model_dir"], num_labels=1,
                                    use_cuda=torch.cuda.is_available(), args=monotransquest_config)

        for dev, test, test_sentence_pairs in zip(dev_list, test_list, test_sentence_pairs_list):
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
    for dev, test, test_sentence_pairs in zip(dev_list, test_list, test_sentence_pairs_list):
        result, model_outputs, wrong_predictions = model.eval_model(dev, pearson_corr=pearson_corr,
                                                                    spearman_corr=spearman_corr,
                                                                    mae=mean_absolute_error)
        predictions, raw_outputs = model.predict(test_sentence_pairs)
        dev['predictions'] = model_outputs
        test['predictions'] = predictions

for dev, test, index, language in zip(dev_list, test_list, index_list, [*languages]):
    dev = un_fit(dev, 'labels')
    dev = un_fit(dev, 'predictions')
    test = un_fit(test, 'predictions')
    dev.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE.split(".")[0] + "_" + language + "." + RESULT_FILE.split(".")[1]), header=True, sep='\t', index=False, encoding='utf-8')
    print(language)
    print_stat(dev, 'labels', 'predictions')

    format_submission(df=test, index=index, method="TransQuest",
                          path=os.path.join(TEMP_DIRECTORY, SUBMISSION_FILE.split(".")[0] + "_" + language + "." +
                                            SUBMISSION_FILE.split(".")[1]))

