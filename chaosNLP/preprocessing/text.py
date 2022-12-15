"""
文本预处理模块
"""


class Tokenizer(object):
    """
    - 切词模块，中文用jieba，英文用nltk
    - 同时带停用词过滤功能
    """
    def __init__(self, langs='zh', stopwords=None):
        """
        Parameters
        ----------
        langs: string
            要支持的语料,zh/en
        stopwords: string
            - 如果是"zh", 则采用默认的中文停用词
            - 如果是"en", 则采用默认的英文停用词

        """
        if langs == 'zh':
            import jieba
            self.wordseg_fn = jieba.lcut
        elif langs == 'en':
            import nltk
            self.wordseg_fn = nltk.tokenize.word_tokenize
        else:
            raise NotImplementedError("目前只支持jieba分词")

        # stopwords
        self.stopwords_list = []
        if isinstance(stopwords, list):
            self.stopwords_list = stopwords
        elif isinstance(stopwords, str):
            if stopwords in ['zh']:
                self.stopwords_list = self.load_stopwords(stopwords)
            else:
                raise NotImplementedError("目前stopwords仅支持zh")

    def tokenize(self, text):
        """
        Parameters
        ----------
        text: string
            要切词的文本

        Returns
        -------
        list
        切词的结果
        """
        return self.wordseg_fn(text)

    def tokenize_list(self, texts):
        """
        Parameters
        ----------
        texts: string or list of string
            构建的文本，可以有多种类型
            - list of string, 先进行分词
            - string, 通过"\n\r\t "等whitespace先进行分句。

        Returns
        -------
        list of list
        切词后的结果
        """
        texts_list = texts
        if isinstance(texts, str):
            texts_list = texts.splitlines()
        assert isinstance(texts_list, list), "输入texts必须为list或者string"

        # 切词并且去掉停用词
        texts_seg_list = [
            [word for word in self.wordseg_fn(text)
             if word not in self.stopwords_list] for text in texts_list]
        return texts_seg_list

    def load_stopwords(self, stopwords_type):
        from pystopwords import stopwords
        return stopwords('zh', 'all')


class textReader(object):
    """
    将文本语料转换为 id sequence。
    包含如下功能：
    1. 分词。  TODO: Tokenizer目前自带了一个Dictionary，后续独立出来
    2. 停用词过滤
    3. 高低频过滤
    """
    def __init__(self,
                 langs='zh',
                 vocabulary=None,
                 lower=True,
                 stopwords=None,
                 min_freq=0,
                 max_freq=None,
                 max_word_nums=None):
        """
        Parameters
        ----------
        langs: string
            要支持的语料,zh/en, 主要用与设置切词方法
        vocabulary: dict or None
            要使用的词典，如果为None则新建一个。
        lower: bool
            对英文采用小写，默认为True
        stopwords: string or None
            - 如果是"zh", 则采用默认的中文停用词
            - 如果是"en", 则采用默认的英文停用词
        min_freq: int or float
            最小词频，如果是[0,1]之间的浮点数，则按比例计算
        max_freq: int or float
            最大词频，如果是[0,1]之间的浮点数，则按比例计算
        max_size: int
            最大词数量，满足min_freq和max_freq后，保留词频多的词。
        """
        self.lower = lower
        self.langs = langs
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.max_word_nums = max_word_nums

        self.tokenizer = Tokenizer(langs=langs, stopwords=stopwords)

        # dictoionary
        self.pad_token, self.pad_id = '<pad>', 0  # pad字符
        self.oov_token, self.oov_id = '<unk>', 1  # 未知字符
        self.special_tokens = [self.pad_token, self.oov_token]
        if vocabulary:
            assert vocabulary.get(self.pad_token, self.pad_id) == self.pad_id,\
                f"词典中{self.pad_token}的index必须为{self.pad_id}"
            assert vocabulary.get(self.oov_token, self.oov_id) == self.oov_id,\
                f"词典中{self.oov_token}的index必须为{self.oov_id}"
            self.word_index = vocabulary
        else:
            self.word_index = {self.pad_token: self.pad_id,
                               self.oov_token: self.oov_id}
        self.word_count = {self.pad_token: 0, self.oov_token: 0}

    def fit_texts(self, texts, return_seqs=False,
                  padding=False, max_seq_len=None):
        """
        根据texts构建词典。

        Parameters
        ----------
        texts: string
            构建的文本，可以有多种类型
            - list of string, 先进行分词
            - string, 通过"\n\r"等whitespace先进行分句。
            - TODO: 支持迭代器类型
        return_seqs: bool
            是否返回texts的id sequence
        padding: bool
            return_seqs为True时生效，在末尾添加padding字符，需要配合max_seq_len一起使用
        max_seq_len: int/None
            return_seqs为True时生效，每个序列的最大长度

        """
        texts_seg_list = self.tokenizer.tokenize_list(texts)
        # 处理文本
        for segs in texts_seg_list:
            if self.lower:
                segs = [elem.lower() for elem in segs]
            for word in segs:
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
                if word not in self.word_count:
                    self.word_count[word] = 0
                self.word_count[word] += 1

        deleted_ids = set()
        if self.min_freq is not None or self.max_freq is not None:
            # 存储要删除的word
            words_should_delete = set()
            total_nums = sum(self.word_count.values())
            for word in self.word_count:
                # 特殊字符不处理
                if word in self.special_tokens:
                    continue
                if isinstance(self.min_freq, int) and \
                   self.word_count[word] < self.min_freq:
                    words_should_delete.add(word)
                if isinstance(self.min_freq, float):
                    assert 0.0 <= self.min_freq <= 1.0, "min_freq为浮点数必须在[0,1]内"
                    if self.word_count[word] / total_nums < self.min_freq:
                        words_should_delete.add(word)
                if isinstance(self.max_freq, int) and \
                   self.word_count[word] > self.max_freq:
                    words_should_delete.add(word)
                if isinstance(self.max_freq, float):
                    assert 0.0 <= self.max_freq <= 1.0, "max_freq为浮点数必须在[0,1]内"
                    if self.word_count[word] / total_nums > self.max_freq:
                        words_should_delete.add(word)

            for word in words_should_delete:
                deleted_ids.add(self.word_index[word])
                del self.word_index[word]
                del self.word_count[word]

        # 不能超过最大词汇量
        if self.max_word_nums is not None and \
           len(self.word_index) > self.max_word_nums:
            sorted_count = sorted(self.word_count.items(), key=lambda x: x[1])
            diff = len(self.word_index) - self.max_word_nums
            i = 0
            while diff:
                word = sorted_count[i][0]
                if word not in self.special_tokens:
                    deleted_ids.add(self.word_index[word])
                    del self.word_index[word]
                    del self.word_count[word]
                    diff -= 1
                i += 1
        # reindex
        if len(deleted_ids):
            deleted_ids = sorted(list(deleted_ids))
            import bisect
            for word in self.word_index:
                # 前面已经删除了n个数字，所以原来的下标要减少n
                n = bisect.bisect_left(deleted_ids, self.word_index[word])
                self.word_index[word] -= n

        # to sequence
        if return_seqs:
            return self.tokenlist_to_seqs(texts_seg_list, padding, max_seq_len)

    def text_to_seqs(self, texts, padding=False, max_seq_len=None):
        """
        将texts转换为id序列

        Parameters
        ----------
        texts: string or list of string
            构建的文本，可以有多种类型
            - list of string
            - string, 通过"\n\r"等whitespace先进行分句。
        padding: bool
            在末尾添加padding字符，需要配合max_seq_len一起使用
        max_seq_len: int/None
            每个序列的最大长度

        Returns
        -------
        a generator
        为了内存考虑，返回一个生成器。
        """
        return self.tokenlist_to_seqs(
            self.tokenizer.tokenize_list(texts), padding, max_seq_len)

    def tokenlist_to_seqs(self, texts_seg_list, padding, max_seq_len):
        """
        Parameters
        ----------
        texts_seg_list: list of list
            tokenize后的list
        padding: bool
            在末尾添加padding字符，需要配合max_seq_len一起使用
        max_seq_len: int/None
            每个序列的最大长度

        Yields
        ------
        list
        list of token_ids
        """
        for tokens in texts_seg_list:
            token_ids = [self.get_word_id(x) for x in tokens]
            if padding:
                assert isinstance(max_seq_len, int), "max_seq_len未设置"
                token_ids = token_ids[:max_seq_len]
                if len(token_ids) < max_seq_len:
                    token_ids += [self.pad_id] * (max_seq_len - len(token_ids))
            yield token_ids

    def text_to_count_vector(self, texts):
        """
        将文本转换为Vector Space Model的词频矩阵
        """
        from collections import Counter
        from scipy.sparse import csr_matrix
        texts_seg_list = self.tokenizer.tokenize_list(texts)
        data = []
        cols = []
        rows = []
        for (idx, tokens) in enumerate(texts_seg_list):
            print('debug', tokens)
            id_counts = Counter(
                [self.get_word_id(x) for x in tokens]).items()
            data += [x[1] for x in id_counts]
            cols += [x[0] for x in id_counts]
            rows += [idx] * len(id_counts)
        return csr_matrix((data, (rows, cols)),
                          shape=(len(texts_seg_list), len(self.word_index)))

    def get_word_id(self, word):
        """
        返回词的id
        """
        if self.lower:
            return self.word_index.get(word.lower(), self.oov_id)
        return self.word_index.get(word, self.oov_id)
