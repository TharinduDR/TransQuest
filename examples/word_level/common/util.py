import os
import pandas as pd


def reader(path, args, source_file, target_file, source_tags_file=None, target_tags_file=None):

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
                args["source_column"]: source_sentences,
                args["source_column"]: target_sentences,
                args["source_tags_column"]: source_tags,
                args["target_tags_column"]: target_tags
            }
        )

    else:
        df = pd.DataFrame(
            {
                args["source_column"]: source_sentences,
                args["source_column"]: target_sentences
            }
        )

    return df





