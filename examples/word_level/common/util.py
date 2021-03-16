import os
import pandas as pd


def reader(path, source_file, target_file, source_tags_file=None, target_tags_file=None):

    with open(os.path.join(path, source_file)) as f:
        source_sentences = f.read().splitlines()

    with open(os.path.join(path, target_file)) as f:
        target_sentences = f.read().splitlines()

    if source_tags_file is not None and target_tags_file is not None:
        with open(os.path.join(path,source_tags_file)) as f:
            source_tags = f.read().splitlines()

        with open(os.path.join(path, target_tags_file)) as f:
            target_tags = f.read().splitlines()

        df = pd.DataFrame(
            {
                "source": source_sentences,
                "target": target_sentences,
                "source_tags": source_tags,
                "target_tags": target_tags
            }
        )

    else:
        df = pd.DataFrame(
            {
                "source": source_sentences,
                "target": target_sentences
            }
        )

    return df


def prepare_testdata(raw_df):
    source_sentences = raw_df["source"].tolist()
    target_sentences = raw_df["target"].tolist()

    test_sentences = []
    for source_sentence, target_sentence in zip(source_sentences, target_sentences):
        test_sentences.append([source_sentence, target_sentence])

    return test_sentences






