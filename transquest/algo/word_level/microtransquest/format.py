import pandas as pd


def prepare_data(source_sentences, source_tags, target_sentences, target_tags):

    sentence_id = 0
    data = []
    for source_sentence, source_tag_line, target_sentence, target_tag_lind in zip(source_sentences, source_tags, target_sentences, target_tags):
        for word, tag in zip(source_sentence.split(), source_tag_line.split()):
            data.append([sentence_id, word, tag])

        data.append([sentence_id, "[SEP]", "OK"])

        target_words = target_sentence.split()
        target_tags = target_tag_lind.split()

        data.append([sentence_id, "_", target_tags.pop(0)])

        for word in target_words:
            data.append([sentence_id, word, target_tags.pop(0)])
            data.append([sentence_id, "_", target_tags.pop(0)])

        sentence_id += 1

    return pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])


def prepare_testdata(source_sentences, target_sentences):

    test_sentences = []
    for source_sentence, target_sentence in zip(source_sentences, target_sentences):
        test_sentence = source_sentence + " " + "[SEP]"
        target_words = target_sentence.split()
        for target_word in target_words:
            test_sentence = test_sentence + " " + "_" + " " + target_word
        test_sentence = test_sentence + " " + "_"
        test_sentences.append(test_sentence)

    return test_sentences


def post_process(predicted_sentences):
    sources_tags = []
    targets_tags = []
    for predicted_sentence in predicted_sentences:
        source_tags = []
        target_tags = []
        source_sentence = True
        for word_prediction in predicted_sentence:
            word = list(word_prediction.keys())[0]

            if word == "[SEP]":
                source_sentence = False
                continue
            if source_sentence:
                source_tags.append(list(word_prediction.values())[0])
            else:
                target_tags.append(list(word_prediction.values())[0])
        sources_tags.append(source_tags)
        targets_tags.append(target_tags)

    return sources_tags, targets_tags




