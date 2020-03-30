import logging
import os
import pandas as pd
import torch
from sklearn.metrics import mean_absolute_error
from sklearn.model_selection import train_test_split

from algo.transformers.evaluation import pearson_corr, spearman_corr
from algo.transformers.run_model import QuestModel
from examples.common.util.draw import draw_scatterplot
from examples.common.util.normalizer import fit, un_fit
from examples.ro_en.transformer_config import TEMP_DIRECTORY, MODEL_TYPE, MODEL_NAME, transformer_config, SEED, \
    RESULT_FILE, RESULT_IMAGE

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

TRAIN_FILE = "data/ro-en/train.roen.df.short.tsv"
TEST_FILE = "data/ro-en/dev.roen.df.short.tsv"

model = QuestModel(MODEL_TYPE, MODEL_NAME, num_labels=1, use_cuda=torch.cuda.is_available(),
                   args=transformer_config)

train = pd.read_csv(TRAIN_FILE, sep='\t')
test = pd.read_csv(TEST_FILE, sep='\t')

train = train[['original', 'translation', 'z_mean']]
test = test[['original', 'translation', 'z_mean']]

train = train.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()
test = test.rename(columns={'original': 'text_a', 'translation': 'text_b', 'z_mean': 'labels'}).dropna()

train = fit(train, 'labels')
test = fit(test, 'labels')

logging.info("Started Training")

if transformer_config["evaluate_during_training"]:
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

test = un_fit(test, 'labels')
test = un_fit(test, 'predictions')
test.to_csv(os.path.join(TEMP_DIRECTORY, RESULT_FILE), header=True, sep='\t', index=False, encoding='utf-8')
draw_scatterplot(test, 'labels', 'predictions', os.path.join(TEMP_DIRECTORY, RESULT_IMAGE), MODEL_NAME + " " + MODEL_TYPE)
