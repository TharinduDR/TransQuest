import pandas as pd
from sklearn.metrics import mean_absolute_error

from examples.sentence_level.wmt_2018 import fit
from transquest.algo.sentence_level.monotransquest.evaluation import pearson_corr, spearman_corr, rmse

import matplotlib.pyplot as plt


def draw_scatterplot(data_frame, real_column, prediction_column, path, topic):
    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    data_frame = fit(data_frame, real_column)
    data_frame = fit(data_frame, prediction_column)

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (rmse_value, mae, pearson, spearman)

    plt.figure()
    ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='z_mean', title=topic)
    ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='predicted z_mean',
                    ax=ax)
    ax.text(0.5*data_frame.shape[0], min(min(data_frame[real_column].tolist()), min(data_frame[prediction_column].tolist())), textstr, fontsize=10)

    fig = ax.get_figure()
    fig.savefig(path)


def print_stat(data_frame, real_column, prediction_column):
    data_frame = data_frame.sort_values(real_column)

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    mae = mean_absolute_error(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nMAE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (rmse_value, mae, pearson, spearman)

    print(textstr)