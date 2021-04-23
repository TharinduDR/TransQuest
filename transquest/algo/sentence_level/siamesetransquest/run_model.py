import logging
import math
import os
import random


import numpy as np
import torch
from sklearn.metrics.pairwise import paired_cosine_distances


from torch.utils.data import DataLoader


from transquest.algo.sentence_level.siamesetransquest.evaluation.embedding_similarity_evaluator import \
    EmbeddingSimilarityEvaluator
from transquest.algo.sentence_level.siamesetransquest.losses.cosine_similarity_loss import CosineSimilarityLoss
from transquest.algo.sentence_level.siamesetransquest.model_args import SiameseTransQuestArgs
from transquest.algo.sentence_level.siamesetransquest.models.siamese_transformer import SiameseTransformer
from transquest.algo.sentence_level.siamesetransquest.readers.input_example import InputExample


logger = logging.getLogger(__name__)


class SiameseTransQuestModel:
    """
    Loads or create a SentenceTransformer model, that can be used to map sentences / text to embeddings.

    :param model_name_or_path: If it is a filepath on disc, it loads the model from that path. If it is not a path, it first tries to download a pre-trained SentenceTransformer model. If that fails, tries to construct a model from Huggingface models repository with that name.
    :param device: Device (like 'cuda' / 'cpu') that should be used for computation. If None, checks if a GPU can be used.
    """

    def __init__(self, model_name: str = None, args=None, device: str = None):

        self.args = self._load_model_args(model_name)

        if isinstance(args, dict):
            self.args.update_from_dict(args)
        elif isinstance(args, SiameseTransQuestArgs):
            self.args = args

        if self.args.thread_count:
            torch.set_num_threads(self.args.thread_count)

        if self.args.manual_seed:
            random.seed(self.args.manual_seed)
            np.random.seed(self.args.manual_seed)
            torch.manual_seed(self.args.manual_seed)
            if self.args.n_gpu > 0:
                torch.cuda.manual_seed_all(self.args.manual_seed)

        self.model = SiameseTransformer(model_name, max_seq_length=self.args.max_seq_length)

    def predict(self, to_predict, verbose=True):
        sentences1 = []
        sentences2 = []

        for text_1, text_2 in to_predict:
            sentences1.append(text_1)
            sentences2.append(text_2)

        embeddings1 = self.model.encode(sentences1, show_progress_bar=verbose, convert_to_numpy=True)
        embeddings2 = self.model.encode(sentences2, show_progress_bar=verbose, convert_to_numpy=True)

        cosine_scores = 1 - (paired_cosine_distances(embeddings1, embeddings2))

        return cosine_scores

    def train_model(self, train_df, eval_df, args=None, output_dir=None, verbose=True):

        train_samples = []
        for index, row in train_df.iterrows():
            score = float(row["labels"])
            inp_example = InputExample(texts=[row['text_a'], row['text_b']], label=score)
            train_samples.append(inp_example)

        eval_samples = []
        for index, row in eval_df.iterrows():
            score = float(row["labels"])
            inp_example = InputExample(texts=[row['text_a'], row['text_b']], label=score)
            eval_samples.append(inp_example)

        train_dataloader = DataLoader(train_samples, shuffle=True,
                                      batch_size=self.args.train_batch_size)
        train_loss = CosineSimilarityLoss(model=self)

        evaluator = EmbeddingSimilarityEvaluator.from_input_examples(eval_samples, name='eval')
        warmup_steps = math.ceil(len(train_dataloader) * self.args.num_train_epochs * 0.1)

        self.model.fit(train_objectives=[(train_dataloader, train_loss)],
                 evaluator=evaluator,
                 epochs=self.args.num_train_epochs,
                 evaluation_steps=self.args.evaluate_during_training_steps,
                 optimizer_params={'lr': self.args.learning_rate,
                                   'eps': self.args.adam_epsilon,
                                   'correct_bias': False},
                 warmup_steps=warmup_steps,
                 weight_decay=self.args.weight_decay,
                 max_grad_norm=self.args.max_grad_norm,
                 output_path=self.args.best_model_dir)

    def save_model_args(self, output_dir):
        os.makedirs(output_dir, exist_ok=True)
        self.args.save(output_dir)

    def _load_model_args(self, input_dir):
        args = SiameseTransQuestArgs()
        args.load(input_dir)
        return args


