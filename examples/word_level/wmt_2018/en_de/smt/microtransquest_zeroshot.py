import os

from examples.word_level.common.util import reader
from examples.word_level.wmt_2018.en_de.smt.microtransquest_config import microtransquest_config, TEST_PATH, \
    TEST_SOURCE_FILE, \
    TEST_TARGET_FILE, TEMP_DIRECTORY, TEST_SOURCE_TAGS_FILE, TEST_TARGET_TAGS_FILE, TEST_TARGET_GAPS_FILE, MODEL_TYPE
from transquest.algo.word_level.microtransquest.format import prepare_testdata, post_process
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel

if not os.path.exists(TEMP_DIRECTORY):
    os.makedirs(TEMP_DIRECTORY)

models = {
    "en_cs": "/content/drive/MyDrive/TransQuestModels/MicroTransQuest/wmt2018/en_cs_smt_it/best_model",
    "en_de_nmt": "/content/drive/MyDrive/TransQuestModels/MicroTransQuest/wmt2018/en_de_nmt_it/best_model",
    "en_de_smt": "/content/drive/MyDrive/TransQuestModels/MicroTransQuest/wmt2018/en_de_smt_it/best_model",
    "en_lv_nmt": "/content/drive/MyDrive/TransQuestModels/MicroTransQuest/wmt2018/en_lv_nmt_pharmaceutical/best_model",
    "en_lv_smt": "/content/drive/MyDrive/TransQuestModels/MicroTransQuest/wmt2018/en_lv_smt_pharmaceutical/best_model"
}

for language, path in models.items():

    if not os.path.exists(os.path.join(TEMP_DIRECTORY, language)):
        os.makedirs(os.path.join(TEMP_DIRECTORY, language))

    if language == "en_lv_smt":
        model = MicroTransQuestModel(MODEL_TYPE, path, labels=["BAD", "OK"], args=microtransquest_config)

    else:
        model = MicroTransQuestModel(MODEL_TYPE, path, labels=["OK", "BAD"], args=microtransquest_config)

    if not os.path.exists(TEMP_DIRECTORY):
        os.makedirs(TEMP_DIRECTORY)

    raw_test_df = reader(TEST_PATH, microtransquest_config, TEST_SOURCE_FILE, TEST_TARGET_FILE)

    test_sentences = prepare_testdata(raw_test_df, args=microtransquest_config)

    fold_sources_tags = []
    fold_targets_tags = []

    predicted_labels, raw_predictions = model.predict(test_sentences, split_on_space=True)
    sources_tags, targets_tags = post_process(predicted_labels, test_sentences, args=microtransquest_config)

    test_source_sentences = raw_test_df[microtransquest_config["source_column"]].tolist()
    test_target_sentences = raw_test_df[microtransquest_config["target_column"]].tolist()

    with open(os.path.join(TEMP_DIRECTORY, language, TEST_SOURCE_TAGS_FILE), 'w') as f:
        for sentence_id, (test_source_sentence, source_prediction) in enumerate(
                zip(test_source_sentences, sources_tags)):
            words = test_source_sentence.split()
            for word_id, (word, word_prediction) in enumerate(zip(words, source_prediction)):
                f.write("MicroTransQuest" + "\t" + "source" + "\t" +
                        str(sentence_id) + "\t" + str(word_id) + "\t"
                        + word + "\t" + word_prediction + '\n')

    with open(os.path.join(TEMP_DIRECTORY, language, TEST_TARGET_TAGS_FILE), 'w') as target_f, open(
            os.path.join(TEMP_DIRECTORY, language, TEST_TARGET_GAPS_FILE), 'w') as gap_f:
        for sentence_id, (test_sentence, target_prediction) in enumerate(zip(test_sentences, targets_tags)):
            target_sentence = test_sentence.split("[SEP]")[1]
            words = target_sentence.split()
            # word_predictions = target_prediction.split()
            gap_index = 0
            word_index = 0
            for word_id, (word, word_prediction) in enumerate(zip(words, target_prediction)):
                if word_id % 2 == 0:
                    gap_f.write("MicroTransQuest" + "\t" + "gap" + "\t" +
                                str(sentence_id) + "\t" + str(gap_index) + "\t"
                                + "gap" + "\t" + word_prediction + '\n')
                    gap_index += 1
                else:
                    target_f.write("MicroTransQuest" + "\t" + "mt" + "\t" +
                                   str(sentence_id) + "\t" + str(word_index) + "\t"
                                   + word + "\t" + word_prediction + '\n')
                    word_index += 1
