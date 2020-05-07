def format_submission(df, language_pair, method, index, path):
    predictions = df['predictions']

    with open(path, 'w') as f:
        for number, prediction in zip(index, predictions):
            text = language_pair + "\t" + method + "\t" + number + "\t" + prediction
            f.write("%s\n" % text)