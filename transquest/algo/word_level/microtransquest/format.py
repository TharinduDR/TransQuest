import pandas as pd


def prepare_data(raw_df, args):
    source_sentences = raw_df["source"].tolist()
    source_tags = raw_df["source_tags"].tolist()
    target_sentences = raw_df["target"].tolist()
    target_tags = raw_df["target_tags"].tolist()

    sentence_id = 0
    data = []
    for source_sentence, source_tag_line, target_sentence, target_tag_lind in zip(source_sentences, source_tags,
                                                                                  target_sentences, target_tags):
        for word, tag in zip(source_sentence.split(), source_tag_line.split()):
            data.append([sentence_id, word, tag])

        data.append([sentence_id, "[SEP]", "OK"])

        target_words = target_sentence.split()
        target_tags = target_tag_lind.split()

        data.append([sentence_id, args.tag, target_tags.pop(0)])

        for word in target_words:
            data.append([sentence_id, word, target_tags.pop(0)])
            data.append([sentence_id, args.tag, target_tags.pop(0)])

        sentence_id += 1

    return pd.DataFrame(data, columns=['sentence_id', 'words', 'labels'])


def format_to_test(to_test, args):
    test_sentences = []
    for source_sentence, target_sentence in to_test:
        test_sentence = source_sentence + " " + "[SEP]"
        target_words = target_sentence.split()
        for target_word in target_words:
            test_sentence = test_sentence + " " + args.tag + " " + target_word
        test_sentence = test_sentence + " " + args.tag
        test_sentences.append(test_sentence)

    return test_sentences


def post_process(predicted_sentences, test_sentences, args):
    sources_tags = []
    targets_tags = []
    for predicted_sentence, test_sentence in zip(predicted_sentences, test_sentences):
        source_tags = []
        target_tags = []
        words = test_sentence.split()
        is_source_sentence = True
        source_sentence = test_sentence.split("[SEP]")[0]
        target_sentence = test_sentence.split("[SEP]")[1]

        for idx, word in enumerate(words):

            if word == "[SEP]":
                is_source_sentence = False
                continue
            if is_source_sentence:
                if idx < len(predicted_sentence):
                    source_tags.append(list(predicted_sentence[idx].values())[0])
                else:
                    source_tags.append(args.default_quality)
            else:
                if idx < len(predicted_sentence):
                    target_tags.append(list(predicted_sentence[idx].values())[0])
                else:
                    target_tags.append(args.default_quality)

        assert len(source_tags) == len(source_sentence.split())

        if len(target_sentence.split()) > len(target_tags):
            target_tags = target_tags + [args.default_quality for x in
                                         range(len(target_sentence.split()) - len(target_tags))]

        assert len(target_tags) == len(target_sentence.split())
        sources_tags.append(source_tags)
        targets_tags.append(target_tags)

    return sources_tags, targets_tags

# def post_process(predicted_sentences, test_sentences):
#     sources_tags = []
#     targets_tags = []
#     for predicted_sentence, test_sentence in zip(predicted_sentences,test_sentences):
#         source_tags = []
#         target_tags = []
#         words = test_sentence.split()
#         source_sentence = True
#         for word_prediction in predicted_sentence:
#             word = list(word_prediction.keys())[0]
#
#             if word == "[SEP]":
#                 source_sentence = False
#                 continue
#             if source_sentence:
#                 source_tags.append(list(word_prediction.values())[0])
#             else:
#                 target_tags.append(list(word_prediction.values())[0])
#         sources_tags.append(source_tags)
#         targets_tags.append(target_tags)
#
#     return sources_tags, targets_tags
