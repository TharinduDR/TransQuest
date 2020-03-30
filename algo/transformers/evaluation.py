from scipy.stats import pearsonr, spearmanr
import numpy as np

def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


def rmse(preds, labels):
    return np.sqrt(((preds - labels) ** 2).mean())
