"""
Bag of Word
"""


class BagOfWordModel:
    """
    词袋模型，类似sklearn的CountVectorizer
    """

    def transform(self, token_ids_seqs, vocab_size=None):
        """
        Parameters
        ----------
        token_ids_seqs: list of list
            原始文本id化后的序列
        vocab_size: int or None
            词典大小，注意要比id的最大值还要大。如果为空，则通过id自动生成

        Returns
        -------
        csr_matrix
        词袋用压缩稀疏矩阵来表示。
        """
        from collections import Counter
        from scipy.sparse import csr_matrix
        data = []
        cols = []
        rows = []
        max_id = 0
        for (idx, token_ids) in enumerate(token_ids_seqs):
            id_counts = Counter(token_ids).items()
            data += [x[1] for x in id_counts]
            cols += [x[0] for x in id_counts]
            rows += [idx] * len(id_counts)
            max_id = max(max_id, max(token_ids))
        if vocab_size is None:
            vocab_size = max_id + 1
        print(vocab_size, data)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(token_ids_seqs), vocab_size))
