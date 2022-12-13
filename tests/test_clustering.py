import sys
import numpy as np
import random
from sklearn.metrics.pairwise import cosine_similarity

sys.path.append('..')
from chaosNLP.utils.clustering import SinglePassCluster


def test_single_pass_cluster_cosine():
    cluster = SinglePassCluster(0.6)
    x = np.random.rand(5)
    cluster.single_pass([x])
    y = np.random.rand(5)
    _, score1 = cluster.get_similar_cluster_id(y)
    score2 = cosine_similarity(x.reshape(1, -1), y.reshape(1, -1))[0][0]
    assert score1 - score2 < 1e-4
    # assert score - cosine_similarity(x, y) < 1e-4


def test_single_pass_cluster():
    cluster = SinglePassCluster(0.6)
    x = [
        [1, 5],
        [10, 1],
        [11, 1],
        [9, 1],
        [2, 10.01],
    ]
    cluster.single_pass(x)
    # print(cluster._cluster_dict)
    assert len(cluster._cluster_dict) == 2
    assert cluster._cluster_centroid.shape == (2, 2)
    assert cluster.get_similar_cluster_id([1, 6])[0] == 0
    assert cluster.get_similar_cluster_id([10.5, 1])[0] == 1
