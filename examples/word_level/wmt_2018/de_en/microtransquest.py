import os
import shutil

from sklearn.model_selection import train_test_split

from examples.word_level.common.util import reader, prepare_testdata
from examples.word_level.wmt_2018.de_en.microtransquest_config import TRAIN_PATH, TRAIN_SOURCE_FILE, \
    TRAIN_SOURCE_TAGS_FILE, \
    TRAIN_TARGET_FILE, \
    TRAIN_TARGET_TAGS_FLE, MODEL_TYPE, MODEL_NAME, microtransquest_config, TEST_PATH, TEST_SOURCE_FILE, \
    TEST_TARGET_FILE, TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE, SEED, TEST_TARGET_TAGS_FILE, TEST_TARGET_GAPS_FILE, \
    DEV_PATH, DEV_SOURCE_FILE, DEV_TARGET_FILE, DEV_SOURCE_TAGS_FILE, DEV_TARGET_TAGS_FLE, DEV_SOURCE_TAGS_FILE_SUB, \
    DEV_TARGET_TAGS_FILE_SUB, DEV_TARGET_GAPS_FILE_SUB
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

raw_train_df = reader(TRAIN_PATH, TRAIN_SOURCE_FILE, TRAIN_TARGET_FILE, TRAIN_SOURCE_TAGS_FILE,
                      TRAIN_TARGET_TAGS_FLE)
raw_dev_df = reader(DEV_PATH, DEV_SOURCE_FILE, DEV_TARGET_FILE, DEV_SOURCE_TAGS_FILE,
                    DEV_TARGET_TAGS_FLE)
raw_test_df = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)


test_sentences = prepare_testdata(raw_test_df)
dev_sentences = prepare_testdata(raw_dev_df)

fold_sources_tags = []
fold_targets_tags = []

dev_fold_sources_tags = []
dev_fold_targets_tags = []

for i in range(microtransquest_config["n_fold"]):

    if os.path.exists(microtransquest_config['output_dir']) and os.path.isdir(microtransquest_config['output_dir']):
        shutil.rmtree(microtransquest_config['output_dir'])

    if microtransquest_config["evaluate_during_training"]:
        raw_train, raw_eval = train_test_split(raw_train_df, test_size=0.1, random_state=SEED * i)
        model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=["OK", "BAD"], args=microtransquest_config)
        model.train_model(raw_train, eval_data=raw_eval)
        model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=["OK", "BAD"],
                                     args=microtransquest_config)

    else:
        model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=["OK", "BAD"], args=microtransquest_config)
        model.train_model(raw_train_df)

    sources_tags, targets_tags = model.predict(test_sentences, split_on_space=True)
    fold_sources_tags.append(sources_tags)
    fold_targets_tags.append(targets_tags)

    dev_sources_tags, dev_targets_tags = model.predict(dev_sentences, split_on_space=True)
    dev_fold_sources_tags.append(dev_sources_tags)
    dev_fold_targets_tags.append(dev_targets_tags)

source_predictions = []
for sentence_id in range(len(test_sentences)):
    majority_prediction = []
    predictions = []
    for fold_prediction in fold_sources_tags:
        predictions.append(fold_prediction[sentence_id])

    sentence_length = len(predictions[0])

    for word_id in range(sentence_length):
        word_prediction = []
        for prediction in predictions:
            word_prediction.append(prediction[word_id])
        majority_prediction.append(max(set(word_prediction), key=word_prediction.count))
    source_predictions.append(majority_prediction)

target_predictions = []
for sentence_id in range(len(test_sentences)):
    majority_prediction = []
    predictions = []
    for fold_prediction in fold_targets_tags:
        predictions.append(fold_prediction[sentence_id])

    sentence_length = len(predictions[0])

    for word_id in range(sentence_length):
        word_prediction = []
        for prediction in predictions:
            word_prediction.append(prediction[word_id])
        majority_prediction.append(max(set(word_prediction), key=word_prediction.count))
    target_predictions.append(majority_prediction)

test_source_sentences = raw_test_df["source"].tolist()
test_target_sentences = raw_test_df["target"].tolist()

with open(os.path.join(TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE), 'w') as f:
    for sentence_id, (test_source_sentence, source_prediction) in enumerate(
            zip(test_source_sentences, source_predictions)):
        words = test_source_sentence.split()
        for word_id, (word, word_prediction) in enumerate(zip(words, source_prediction)):
            f.write("MicroTransQuest" + "\t" + "source" + "\t" +
                    str(sentence_id) + "\t" + str(word_id) + "\t"
                    + word + "\t" + word_prediction + '\n')

with open(os.path.join(TEMP_DIRECTORY, TEST_TARGET_TAGS_FILE), 'w') as target_f, open(
        os.path.join(TEMP_DIRECTORY, TEST_TARGET_GAPS_FILE), 'w') as gap_f:
    for sentence_id, (test_target_sentence, target_prediction) in enumerate(zip(test_target_sentences, target_predictions)):
        # target_sentence = test_sentence.split("[SEP]")[1]
        words = test_target_sentence.split()
        # word_predictions = target_prediction.split()
        gap_index = 0
        word_index = 0

        for prediction_id, prediction in enumerate(target_prediction):
            if prediction_id % 2 == 0:
                gap_f.write("MicroTransQuest" + "\t" + "gap" + "\t" +
                            str(sentence_id) + "\t" + str(gap_index) + "\t"
                            + "gap" + "\t" + prediction + '\n')
                gap_index += 1
            else:
                target_f.write("MicroTransQuest" + "\t" + "mt" + "\t" +
                               str(sentence_id) + "\t" + str(word_index) + "\t"
                               + words[word_index] + "\t" + prediction + '\n')
                word_index += 1

# Predictions for dev file
dev_source_predictions = []
for sentence_id in range(len(dev_sentences)):
    majority_prediction = []
    predictions = []
    for fold_prediction in dev_fold_sources_tags:
        predictions.append(fold_prediction[sentence_id])

    sentence_length = len(predictions[0])

    for word_id in range(sentence_length):
        word_prediction = []
        for prediction in predictions:
            word_prediction.append(prediction[word_id])
        majority_prediction.append(max(set(word_prediction), key=word_prediction.count))
    dev_source_predictions.append(majority_prediction)

dev_target_predictions = []
for sentence_id in range(len(dev_sentences)):
    majority_prediction = []
    predictions = []
    for fold_prediction in dev_fold_targets_tags:
        predictions.append(fold_prediction[sentence_id])

    sentence_length = len(predictions[0])

    for word_id in range(sentence_length):
        word_prediction = []
        for prediction in predictions:
            word_prediction.append(prediction[word_id])
        majority_prediction.append(max(set(word_prediction), key=word_prediction.count))
    dev_target_predictions.append(majority_prediction)

dev_source_sentences = raw_dev_df["source"].tolist()
dev_target_sentences = raw_dev_df["target"].tolist()
dev_source_gold_tags = raw_dev_df["source_tags"].tolist()
dev_target_gold_tags = raw_dev_df["target_tags"].tolist()

with open(os.path.join(TEMP_DIRECTORY, DEV_SOURCE_TAGS_FILE_SUB), 'w') as f:
    for sentence_id, (dev_source_sentence, dev_source_prediction, source_gold_tag) in enumerate(
            zip(dev_source_sentences, dev_source_predictions, dev_source_gold_tags)):
        words = dev_source_sentence.split()
        gold_predictions = source_gold_tag.split()
        for word_id, (word, word_prediction, gold_prediction) in enumerate(
                zip(words, dev_source_prediction, gold_predictions)):
            f.write("MicroTransQuest" + "\t" + "source" + "\t" +
                    str(sentence_id) + "\t" + str(word_id) + "\t"
                    + word + "\t" + word_prediction + "\t" + gold_prediction + '\n')

with open(os.path.join(TEMP_DIRECTORY, DEV_TARGET_TAGS_FILE_SUB), 'w') as target_f, open(
        os.path.join(TEMP_DIRECTORY, DEV_TARGET_GAPS_FILE_SUB), 'w') as gap_f:
    for sentence_id, (dev_sentence, dev_target_prediction, dev_target_gold_tag) in enumerate(
            zip(dev_target_sentences, dev_target_predictions, dev_target_gold_tags)):
        words = dev_sentence.split()
        gold_predictions = dev_target_gold_tag.split()
        gap_index = 0
        word_index = 0

        for prediction_id, (prediction, gold_prediction) in enumerate(zip(dev_target_prediction, gold_predictions)):
            if prediction_id % 2 == 0:
                gap_f.write("MicroTransQuest" + "\t" + "gap" + "\t" +
                            str(sentence_id) + "\t" + str(gap_index) + "\t"
                            + "gap" + "\t" + prediction + "\t" + gold_prediction + '\n')
                gap_index += 1
            else:
                target_f.write("MicroTransQuest" + "\t" + "mt" + "\t" +
                               str(sentence_id) + "\t" + str(word_index) + "\t"
                               + words[word_index] + "\t" + prediction + "\t" + gold_prediction + '\n')
                word_index += 1

