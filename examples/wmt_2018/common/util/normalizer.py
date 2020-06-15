from sklearn import preprocessing

min_max_scaler = preprocessing.MinMaxScaler()


def fit(df, label):
    x = df[[label]].values.astype(float)
    x_scaled = min_max_scaler.fit_transform(x)
    df[label] = x_scaled
    return df


def un_fit(df, label):
    x = df[[label]].values.astype(float)
    x_unscaled = min_max_scaler.inverse_transform(x)
    df[label] = x_unscaled
    return df