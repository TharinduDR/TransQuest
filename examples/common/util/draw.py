import pandas as pd
from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr, rmse

import matplotlib.pyplot as plt

def draw_scatterplot(data_frame, real_column, prediction_column, path, topic):

    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.4f\nPearson Correlation=%.4f\nSpearman Correlation=%.4f' % (rmse_value, pearson, spearman)

    print(textstr)

    plt.figure()
    ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='z_mean', title=topic)
    ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='predicted z_mean',
                    ax=ax)
    ax.text(0.5*data_frame.shape[0], max(min(data_frame[real_column].tolist()), min(data_frame[prediction_column].tolist())), textstr, fontsize=10)

    fig = ax.get_figure()
    fig.savefig(path)