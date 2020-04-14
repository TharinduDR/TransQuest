import seaborn as sns
import pandas as pd

from transquest.algo.transformers.evaluation import pearson_corr, spearman_corr, rmse

sns.set()


def draw_scatterplot(data_frame, real_column, prediction_column, path, topic):
    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    pearson = pearson_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    spearman = spearman_corr(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())
    rmse_value = rmse(data_frame[real_column].tolist(), data_frame[prediction_column].tolist())

    textstr = 'RMSE=%.3f\n$Pearson Correlation=%.3f$\n$Spearman Correlation=%.3f$' % (rmse_value, pearson, spearman)

    print(textstr)

    ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='z_mean', title=topic)
    ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='predicted z_mean',
                    ax=ax)
    ax.text(1500, 0.05, textstr, fontsize=12)

    fig = ax.get_figure()
    fig.savefig(path)