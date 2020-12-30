import csv
import gzip
import os
import random

from transquest.algo.sentence_level.siamesetransquest.readers.input_example import InputExample


class QEDataReader:
    """
    Reads in the STS dataset. Each line contains two sentences (s1_col_idx, s2_col_idx) and one label (score_col_idx)

    Default values expects a tab seperated file with the first & second column the sentence pair and third column the score (0...1). Default config normalizes scores from 0...5 to 0...1
    """

    def __init__(self, dataset_folder, s1_col_idx=0, s2_col_idx=1, score_col_idx=2, delimiter="\t",
                 quoting=csv.QUOTE_NONE, normalize_scores=True, min_score=0, header=False, max_score=5):
        self.dataset_folder = dataset_folder
        self.score_col_idx = score_col_idx
        self.s1_col_idx = s1_col_idx
        self.s2_col_idx = s2_col_idx
        self.delimiter = delimiter
        self.quoting = quoting
        self.normalize_scores = normalize_scores
        self.min_score = min_score
        self.max_score = max_score
        self.header = header

    def get_examples(self, filename, max_examples=0, test_file=False):
        """
        filename specified which data split to use (train.csv, dev.csv, test.csv).
        """
        filepath = os.path.join(self.dataset_folder, filename)
        with gzip.open(filepath, 'rt', encoding='utf8') if filename.endswith('.gz') else open(filepath,
                                                                                              encoding="utf-8") as fIn:
            data = csv.reader(fIn, delimiter=self.delimiter, quoting=self.quoting)
            if self.header:
                next(data, None)
            examples = []
            for id, row in enumerate(data):
                if test_file:
                    score = random.uniform(0, 1)
                else:
                    score = float(row[self.score_col_idx])
                if self.normalize_scores:  # Normalize to a 0...1 value
                    score = (score - self.min_score) / (self.max_score - self.min_score)

                s1 = row[self.s1_col_idx]
                s2 = row[self.s2_col_idx]
                examples.append(InputExample(guid=filename + str(id), texts=[s1, s2], label=score))

                if max_examples > 0 and len(examples) >= max_examples:
                    break

        return examples
