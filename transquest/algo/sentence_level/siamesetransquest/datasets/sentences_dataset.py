import logging
from typing import List

import torch
from torch.utils.data import Dataset
from tqdm import tqdm

from transquest.algo.sentence_level.siamesetransquest.readers.input_example import InputExample
from transquest.algo.sentence_level.siamesetransquest.run_model import SiameseTransQuestModel


class SentencesDataset(Dataset):
    """
    DEPRECATED: This class is no longer used. Instead of wrapping your List of InputExamples in a SentencesDataset
    and then passing it to the DataLoader, you can pass the list of InputExamples directly to the dataset loader.
    """
    def __init__(self,
                 examples: List[InputExample],
                 model: SiameseTransQuestModel
                 ):
        self.examples = examples

    def __getitem__(self, item):
        return self.examples[item]

    def __len__(self):
        return len(self.examples)
