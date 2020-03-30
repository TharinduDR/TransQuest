import seaborn as sns
import matplotlib.pyplot as plt
import pandas as pd

sns.set()


def draw_scatterplot(data_frame, real_column, prediction_column, path):
    data_frame = data_frame.sort_values(real_column)
    sort_id = list(range(0, len(data_frame.index)))
    data_frame['id'] = pd.Series(sort_id).values

    ax = sns.scatterplot(x="id", y=real_column,
                         data=data_frame)
    ax = sns.scatterplot(x="id", y=prediction_column,
                         data=data_frame, ax=ax)

    fig = ax.get_figure()
    fig.savefig(path)