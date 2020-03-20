import pandas as pd
from sklearn import preprocessing


def read_files(src_file, target_file, hter_file):
    src_sentences = [line.rstrip('\n') for line in open(src_file)]
    target_sentences = [line.rstrip('\n') for line in open(target_file)]
    hter = list(map(float, [line.rstrip('\n') for line in open(hter_file)]))

    return pd.DataFrame({'text_a': src_sentences, 'text_b': target_sentences, 'labels': hter})


def min_max_normalisation(df, label):
    x = df[[label]].values.astype(float)
    min_max_scaler = preprocessing.MinMaxScaler()
    x_scaled = min_max_scaler.fit_transform(x)

    df[label] = x_scaled
    return df