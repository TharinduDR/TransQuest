import tensorflow_hub as hub
import numpy as np
import tensorflow_text
from tqdm import tqdm


def batch(iterable, n = 1):
    current_batch = []
    for item in iterable:
        current_batch.append(item)
        if len(current_batch) == n:
            yield current_batch
            current_batch = []
    if current_batch:
        yield current_batch


def get_embeddings(sentences, batch_size=8):
    embeddings = []
    embed = hub.load("https://tfhub.dev/google/universal-sentence-encoder-multilingual-large/3")
    for x in tqdm(batch(sentences, batch_size)):
        embeddings.extend(embed(x))

    return embeddings


def prepare_file(input_file, sentence_column):
    input_file["embedding"] = get_embeddings(input_file[sentence_column])
    return input_file

