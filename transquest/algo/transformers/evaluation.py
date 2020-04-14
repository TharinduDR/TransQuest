from scipy.stats import pearsonr, spearmanr
import numpy as np


def pearson_corr(preds, labels):
    return pearsonr(preds, labels)[0]


def spearman_corr(preds, labels):
    return spearmanr(preds, labels)[0]


def rmse(preds, labels):
    return np.sqrt(((np.asarray(preds, dtype=np.float32) - np.asarray(labels, dtype=np.float32)) ** 2).mean())
