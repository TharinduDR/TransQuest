def format_submission(df, language_pair, method, index, path, index_type=None):

    if index_type is None:
        index = index

    elif index_type == "Auto":
        index = range(0, df.shape[0])

    predictions = df['predictions']

    with open(path, 'w') as f:
        for number, prediction in zip(index, predictions):
            text = language_pair + "\t" + method + "\t" + str(number) + "\t" + str(prediction)
            f.write("%s\n" % text)
