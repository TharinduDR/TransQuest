from numpy import dot
from numpy.dual import norm


def sentence_pairs_predict(model, sentence_pairs):
    originals = [sentence_pair[0] for sentence_pair in sentence_pairs]
    translations = [sentence_pair[1] for sentence_pair in sentence_pairs]
    predictions = []
    original_embeddings = model.encode(originals)
    translation_embeddings = model.encode(translations)

    for original_embedding, translation_embedding in zip(original_embeddings, translation_embeddings):
        predictions.append(dot(original_embedding, translation_embedding) / (norm(original_embedding) * norm(translation_embedding)))

    return predictions