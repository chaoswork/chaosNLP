import numpy as np


class SinglePassCluster:
    """
    Single pass cluster algorithm
    """
    def __init__(self, similarity_threshold):
        """
        Params:
          similarity_threshold: float, [0, 1]
            能够聚类的相似度阈值。
        """
        self._cluster_dict = {}
        self._cluster_centroid = None
        self.similarity_threshold = similarity_threshold

    def get_similar_cluster_id(self, vec):
        """
        找到满足最相似且满足相似度阈值的聚类簇id，如果没找到，则返回None
        Params:
          vec: array-like
            query vector
        Returns:
          (cluster_id or None, similarity)
        """
        if self._cluster_centroid is None:
            return None, None
        query = np.array(vec).reshape(1, -1) / np.linalg.norm(vec)
        cos_similarity = np.dot(self._cluster_centroid, query.T)
        max_index = np.argmax(cos_similarity)
        print('debug-max_index', query, self._cluster_centroid, max_index, cos_similarity)
        if cos_similarity[max_index][0] >= self.similarity_threshold:
            return max_index, cos_similarity[max_index][0]
        return None, cos_similarity[max_index][0]

    def single_pass(self, vectors):
        """
        Params:
          vectors: list of vector
        """
        vecs = np.array(vectors)
        vecs = vecs / np.linalg.norm(vecs, axis=1, keepdims=True)
        for i in range(vecs.shape[0]):
            max_index, _ = self.get_similar_cluster_id(vecs[i])
            if max_index is None:
                print('debug-new_index')
                new_index = len(self._cluster_dict)
                self._cluster_dict[new_index] = vecs[i].reshape(1, -1)
                # TODO VSTACK
                if self._cluster_centroid is None:
                    self._cluster_centroid = vecs[i].reshape(1, -1)
                else:
                    self._cluster_centroid = np.vstack(
                        [self._cluster_centroid, vecs[i]])
            else:
                self._cluster_dict[max_index] = np.vstack(
                    [self._cluster_dict[max_index], vecs[i]])
                print('debug-mean', self._cluster_dict[max_index], self._cluster_centroid)
                self._cluster_centroid[max_index] = np.mean(self._cluster_dict[max_index], axis=0)
                print('debug-mean2', self._cluster_centroid[max_index])
                self._cluster_centroid[max_index] /= np.linalg.norm(self._cluster_centroid[max_index])
