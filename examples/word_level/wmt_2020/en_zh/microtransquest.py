import shutil

from sklearn.model_selection import train_test_split
import os
from examples.word_level.common.util import reader, prepare_testdata
from examples.word_level.wmt_2020.en_zh.microtransquest_config import TRAIN_PATH, TRAIN_SOURCE_FILE, \
    TRAIN_SOURCE_TAGS_FILE, \
    TRAIN_TARGET_FILE, \
    TRAIN_TARGET_TAGS_FLE, MODEL_TYPE, MODEL_NAME, microtransquest_config, TEST_PATH, TEST_SOURCE_FILE, \
    TEST_TARGET_FILE, TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE, TEST_TARGET_TAGS_FLE, SEED, DEV_TARGET_TAGS_FILE_SUB, \
    DEV_SOURCE_TAGS_FILE_SUB, DEV_PATH, DEV_SOURCE_FILE, DEV_TARGET_FILE, DEV_SOURCE_TAGS_FILE, DEV_TARGET_TAGS_FLE
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

with open(os.path.join(TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE), 'w') as f:
    for _list in source_predictions:
        for _string in _list:
            f.write(str(_string) + ' ')
        f.write(str('\n'))


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

with open(os.path.join(TEMP_DIRECTORY, TEST_TARGET_TAGS_FLE), 'w') as f:
    for _list in target_predictions:
        for _string in _list:
            f.write(str(_string) + ' ')
        f.write(str('\n'))

# --------- Dev Predictions -----------
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

with open(os.path.join(TEMP_DIRECTORY, DEV_SOURCE_TAGS_FILE_SUB), 'w') as f:
    for _list in source_predictions:
        for _string in _list:
            f.write(str(_string) + ' ')
        f.write(str('\n'))


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

with open(os.path.join(TEMP_DIRECTORY, DEV_TARGET_TAGS_FILE_SUB), 'w') as f:
    for _list in target_predictions:
        for _string in _list:
            f.write(str(_string) + ' ')
        f.write(str('\n'))








