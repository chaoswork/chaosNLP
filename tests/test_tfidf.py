import numpy as np
from chaosNLP.representation import TfIdfModel
from sklearn.feature_extraction.text import TfidfVectorizer


def test_default_tfidf():

    seqs = [[2, 3, 4, 5, 6, 7, 4, 8, 6],
            [4, 9, 6],
            [2, 3, 4, 10, 6, 7, 4, 11, 6]]
    vocab = {f"hello{x}": x for x in range(12)}
    corpus = [' '.join([f"hello{x}" for x in y]) for y in seqs]
    vectorizer = TfidfVectorizer(vocabulary=vocab)
    expected = vectorizer.fit_transform(corpus).toarray()
    mat = TfIdfModel().transform(seqs)
    assert np.allclose(mat, expected)
