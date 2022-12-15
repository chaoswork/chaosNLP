import scipy
import numpy as np
from chaosNLP.representation import BagOfWordModel


class TfIdfModel:
    def __init__(self, norm=True):
        """
        Parameters
        ----------
        norm: bool
            是否对最终的tfidf向量做初始化
        """
        self.norm = norm

    def transform(self, token_ids_seqs, vocab_size=None):
        """
        Parameters
        ----------
        token_ids_seqs: list of list
            原始文本id化后的序列
        vocab_size: int or None
            词典大小，注意要比id的最大值还要大。如果为空，则通过id自动生成
        """
        tf = BagOfWordModel().transform(token_ids_seqs)
        # document frequency
        df = np.bincount(tf.indices, minlength=tf.shape[1])
        # inverse document frequency
        idf = np.log((tf.shape[0] + 1) / (1 + df)) + 1
        idf = scipy.sparse.diags(idf, shape=(tf.shape[1], tf.shape[1]))
        # tf-idf
        tfidf = tf * idf
        if self.norm:
            norm_array = scipy.sparse.linalg.norm(tfidf, axis=1)
            tfidf /= norm_array.reshape(-1, 1)
        print(tfidf)
        return tfidf

