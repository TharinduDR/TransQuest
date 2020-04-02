"""
This examples trains BERT for the STSbenchmark from scratch. It generates sentence embeddings
that can be compared using cosine-similarity to measure the similarity.
"""
from torch.utils.data import DataLoader
import math
from sentence_transformers import SentencesDataset, LoggingHandler,


import logging
from datetime import datetime

from algo.siamese_transformers import losses, models
from algo.siamese_transformers.evaluation.EmbeddingSimilarityEvaluator import EmbeddingSimilarityEvaluator
from algo.siamese_transformers.readers.QEDataReader import QEDataReader
from algo.siamese_transformers.run_model import SiameseQuestModel

#### Just some code to print debug information to stdout


logging.basicConfig(format='%(asctime)s - %(message)s',
                    datefmt='%Y-%m-%d %H:%M:%S',
                    level=logging.INFO,
                    handlers=[LoggingHandler()])
#### /print debug information to stdout

# Read the dataset
train_batch_size = 16
num_epochs = 4
model_save_path = 'output/training_stsbenchmark_bert-'+datetime.now().strftime("%Y-%m-%d_%H-%M-%S")
sts_reader = QEDataReader('datasets/stsbenchmark', normalize_scores=True)

# Use BERT for mapping tokens to embeddings
word_embedding_model = models.bert_model('bert-base-uncased')

# Apply mean pooling to get one fixed sized sentence vector
pooling_model = models.Pooling(word_embedding_model.get_word_embedding_dimension(),
                               pooling_mode_mean_tokens=True,
                               pooling_mode_cls_token=False,
                               pooling_mode_max_tokens=False)

model = SiameseQuestModel(modules=[word_embedding_model, pooling_model])

# Convert the dataset to a DataLoader ready for training
logging.info("Read STSbenchmark train dataset")
train_data = SentencesDataset(sts_reader.get_examples('sts-train.csv'), model)
train_dataloader = DataLoader(train_data, shuffle=True, batch_size=train_batch_size)
train_loss = losses.CosineSimilarityLoss(model=model)


logging.info("Read STSbenchmark dev dataset")
dev_data = SentencesDataset(examples=sts_reader.get_examples('sts-dev.csv'), model=model)
dev_dataloader = DataLoader(dev_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(dev_dataloader)


# Configure the training. We skip evaluation in this example
warmup_steps = math.ceil(len(train_data)*num_epochs/train_batch_size*0.1) #10% of train data for warm-up
logging.info("Warmup-steps: {}".format(warmup_steps))


# Train the model
model.fit(train_objectives=[(train_dataloader, train_loss)],
          evaluator=evaluator,
          epochs=num_epochs,
          evaluation_steps=1000,
          warmup_steps=warmup_steps,
          output_path=model_save_path)


##############################################################################
#
# Load the stored model and evaluate its performance on STS benchmark dataset
#
##############################################################################

model = SiameseQuestModel(model_save_path)
test_data = SentencesDataset(examples=sts_reader.get_examples("sts-test.csv"), model=model)
test_dataloader = DataLoader(test_data, shuffle=False, batch_size=train_batch_size)
evaluator = EmbeddingSimilarityEvaluator(test_dataloader)
model.evaluate(evaluator)
