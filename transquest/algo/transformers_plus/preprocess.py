import tensorflow_hub as hub
import numpy as np
import tensorflow_text
from tqdm import tqdm
from numpy import dot
from numpy.linalg import norm


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


def extend_file(input_file, sentence_column, quality_column, reference_file):
    input_embeddings = input_file["embedding"].tolist()
    input_sentences = input_file[sentence_column].tolist()

    similar_sentences = []
    similarities = []
    similar_sentence_qualities = []

    reference_embeddings = reference_file["embedding"]
    reference_sentences = reference_file[sentence_column]
    reference_qualities = reference_file[quality_column]

    for input_embedding, input_sentence in tqdm(zip(input_embeddings, input_sentences),  total=len(input_embeddings)):

        maximum_similarity = 0.0
        similar_sentence = None
        similar_sentence_quality = None
        for reference_embedding, reference_sentence, reference_quality in zip(reference_embeddings, reference_sentences, reference_qualities):
            cos_sim = dot(input_embedding, reference_embedding) / (norm(input_embedding) * norm(reference_embedding))
            if cos_sim > maximum_similarity and input_sentence != reference_sentence:
                maximum_similarity = cos_sim
                similar_sentence = reference_sentence
                similar_sentence_quality = reference_quality

        similar_sentences.append(similar_sentence)
        similar_sentence_qualities.append(similar_sentence_quality)
        similarities.append(maximum_similarity)

    input_file["similar_sentence"] = similar_sentences
    input_file["similarity"] = similarities
    input_file["similar_sentence_quality"] = similar_sentence_qualities

    return input_file

