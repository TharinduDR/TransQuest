import pandas as pd


def read_files(src_file, target_file, hter_file):
    src_sentences = [line.rstrip('\n') for line in open(src_file)]
    target_sentences = [line.rstrip('\n') for line in open(target_file)]
    hter = list(map(float, [line.rstrip('\n') for line in open(hter_file)]))

    return pd.DataFrame({'text_a': src_sentences, 'text_b': target_sentences, 'labels': hter})
