def format_submission(df, method, index, path):
    predictions = df['predictions']

    with open(path, 'w') as f:
        for number, prediction in zip(index, predictions):
            text = method + "\t" + str(number) + "\t" + str(prediction) + "\t" + str(0)
            f.write("%s\n" % text)