import numpy as np
from chaosNLP.representation import BagOfWordModel


def test_default_text_reader():

    seqs = [[2, 3, 4, 5, 6, 7, 4, 8, 6],
            [4, 9, 6],
            [2, 3, 4, 10, 6, 7, 4, 11, 6]]
    mat = BagOfWordModel().transform(seqs)
    expected = np.array(
        [[0, 0, 1, 1, 2, 1, 2, 1, 1, 0, 0, 0],
         [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0],
         [0, 0, 1, 1, 2, 0, 2, 1, 0, 0, 1, 1]])
    assert np.array_equal(mat.toarray(), expected)

