from examples.word_level.common.util import reader
from examples.word_level.wmt_2020.en_de.microtransquest_config import TRAIN_PATH, SOURCE_FILE, SOURCE_TAGS_FILE, TARGET_FILE, \
    TARGET_TAGS_FLE
from transquest.algo.word_level.preprocess.format import prepare_data


train_source_sentences, train_source_tags, train_target_sentences, train_target_tags = reader(TRAIN_PATH, SOURCE_FILE, SOURCE_TAGS_FILE, TARGET_FILE, TARGET_TAGS_FLE)
train_df = prepare_data(train_source_sentences, train_source_tags, train_target_sentences, train_target_tags)

