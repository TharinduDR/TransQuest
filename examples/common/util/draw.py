import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set()


def draw_scatterplot(data_frame, real_column, prediction_column, path, topic):
    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    ax = data_frame.plot(kind='scatter', x='id', y=real_column, color='DarkBlue', label='Similarity', title=topic);
    ax = data_frame.plot(kind='scatter', x='id', y=prediction_column, color='DarkGreen', label='Predicted Similarity',
                    ax=ax);
    # ax.text(1500, 0.05, textstr, fontsize=12)

    fig = ax.get_figure()
    fig.savefig(path)