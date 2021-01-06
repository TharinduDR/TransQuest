import os


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

        return source_sentences, target_sentences, source_tags, target_tags

    else:
        return source_sentences, target_sentences




