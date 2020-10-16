import pandas as pd

def prepare_dat(source_file, source_tags_file, target_file, target_tags_file):

    with open(source_file) as f:
        source_sentences = f.read().splitlines()

    with open(source_tags_file) as f:
        source_tags = f.read().splitlines()

    with open(target_file) as f:
        target_sentences = f.read().splitlines()

    with open(target_tags_file) as f:
        target_tags = f.read().splitlines()

    sentence_id = 0
    data = []
    for source_sentence, source_tag_line, target_sentence, target_tag_lind in zip(source_sentences, source_tags, target_sentences, target_tags):
        for word, tag in zip(source_sentence.split(), source_tag_line.split()):
            data.append([sentence_id, word, tag])

        data.append([sentence_id, "[SEP]", "SEP"])

        target_words = target_sentence.split()
        target_tags = target_tag_lind.split()

        data.append([sentence_id, "[GAP]", target_tags.pop(0)])

        for word in target_words:
            data.append([sentence_id, word, target_tags.pop(0)])
            data.append([sentence_id, "[GAP]", target_tags.pop(0)])

        sentence_id += 1

    return pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])