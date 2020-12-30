import os

def reader(path, source_file, source_tags_file, target_file, target_tags_file):

    with open(os.path.join(path, source_file)) as f:
        source_sentences = f.read().splitlines()

    with open(os.path.join(path,source_tags_file)) as f:
        source_tags = f.read().splitlines()

    with open(os.path.join(path, target_file)) as f:
        target_sentences = f.read().splitlines()

    with open(os.path.join(path, target_tags_file)) as f:
        target_tags = f.read().splitlines()

    return source_sentences, source_tags, target_sentences, target_tags
