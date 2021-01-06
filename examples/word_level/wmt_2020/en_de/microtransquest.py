from sklearn.model_selection import train_test_split

from examples.word_level.common.util import reader
from examples.word_level.wmt_2020.en_de.microtransquest_config import TRAIN_PATH, TRAIN_SOURCE_FILE, \
    TRAIN_SOURCE_TAGS_FILE, \
    TRAIN_TARGET_FILE, \
    TRAIN_TARGET_TAGS_FLE, MODEL_TYPE, MODEL_NAME, microtransquest_config, TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel
from transquest.algo.word_level.microtransquest.format import prepare_data, prepare_testdata, post_process

train_source_sentences, train_target_sentences, train_source_tags, train_target_tags = reader(TRAIN_PATH, TRAIN_SOURCE_FILE, TRAIN_TARGET_FILE, TRAIN_SOURCE_TAGS_FILE, TRAIN_TARGET_TAGS_FLE)
test_source_sentences, test_target_sentences = reader(TEST_PATH, TEST_SOURCE_FILE, TEST_TARGET_FILE)
train_df = prepare_data(train_source_sentences, train_source_tags, train_target_sentences, train_target_tags)

train_df.to_csv("check.csv", sep='\t', encoding='utf-8')

tags = train_df['labels'].unique().tolist()

model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=microtransquest_config)

if microtransquest_config["evaluate_during_training"]:
    train_df, eval_df = train_test_split(train_df, test_size=0.1,  shuffle=False)
    model.train_model(train_df, eval_df=eval_df)

else:
    model.train_model(train_df)

model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=tags, args=microtransquest_config)
test_sentences = prepare_testdata(test_source_sentences, test_target_sentences)

predicted_labels = model.predict(test_sentences, split_on_space=True)

sources_tags, targets_tags = post_process(predicted_labels)








