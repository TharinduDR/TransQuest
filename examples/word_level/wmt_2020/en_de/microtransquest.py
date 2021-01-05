from sklearn.model_selection import train_test_split

from examples.word_level.common.util import reader
from examples.word_level.wmt_2020.en_de.microtransquest_config import TRAIN_PATH, SOURCE_FILE, SOURCE_TAGS_FILE, \
    TARGET_FILE, \
    TARGET_TAGS_FLE, MODEL_TYPE, MODEL_NAME, microtransquest_config
from transquest.algo.word_level.microtransquest.run_model import MicroTransQuestModel
from transquest.algo.word_level.microtransquest.format import prepare_data


train_source_sentences, train_source_tags, train_target_sentences, train_target_tags = reader(TRAIN_PATH, SOURCE_FILE, SOURCE_TAGS_FILE, TARGET_FILE, TARGET_TAGS_FLE)
train_df = prepare_data(train_source_sentences, train_source_tags, train_target_sentences, train_target_tags)

# train_df.to_csv("check.csv", sep='\t', encoding='utf-8')

tags = train_df['labels'].unique().tolist()

model = MicroTransQuestModel(MODEL_TYPE, MODEL_NAME, labels=tags, args=microtransquest_config)

if microtransquest_config["evaluate_during_training"]:
    train_df, eval_df = train_test_split(train_df, test_size=0.1,  shuffle=False)
    model.train_model(train_df, eval_df=eval_df)

else:
    model.train_model(train_df)

model = MicroTransQuestModel(MODEL_TYPE, microtransquest_config["best_model_dir"], labels=tags, args=microtransquest_config)



