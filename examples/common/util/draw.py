import seaborn as sns
import matplotlib.pyplot as plt

sns.set()


def draw_scatterplot(data_frame, real_column, prediction_column, path):

    ax = sns.scatterplot(x="total_bill", y=real_column,
                         data=data_frame)
    ax = sns.scatterplot(x="total_bill", y=prediction_column,
                         data=data_frame, ax=ax)

    fig = ax.get_figure()
    fig.savefig(path)