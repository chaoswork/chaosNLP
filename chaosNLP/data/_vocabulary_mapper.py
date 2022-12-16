
class VocabularyMapper:
    """
    NLP处理中，非常重要的一步就是将token转变为id，这个类就实现这个功能。
    比起最简单的数据结构字典，该类还提供了如下的功能：
    1. 按照词频和词的数量对词典进行裁剪
    2. 将词典锁定，不再更新

    注意该模块只负责token映射相关的工作，并不负责切词
    """
    def __init__(self,
                 lower=True,
                 min_freq=0,
                 max_freq=None,
                 max_word_nums=None):
        """
        Parameters
        ----------
        lower: bool
            是否进行lower处理，默认为True。
        min_freq: int or float
            最小词频，如果是[0,1]之间的浮点数，则按比例计算
        max_freq: int or float
            最大词频，如果是[0,1]之间的浮点数，则按比例计算
        max_size: int
            最大词数量，满足min_freq和max_freq后，保留词频多的词。
        """
        self.lower = lower
        self.is_freezed = False
        self.min_freq = min_freq
        self.max_freq = max_freq
        self.max_word_nums = max_word_nums

        # 目前包含两个特殊字符
        self.pad_token, self.pad_id = '<pad>', 0  # pad字符
        self.oov_token, self.oov_id = '<unk>', 1  # 未知字符
        self.special_tokens = [self.pad_token, self.oov_token]
#        if vocabulary:
#            assert vocabulary.get(self.pad_token, self.pad_id) == self.pad_id,\
#                f"词典中{self.pad_token}的index必须为{self.pad_id}"
#            assert vocabulary.get(self.oov_token, self.oov_id) == self.oov_id,\
#                f"词典中{self.oov_token}的index必须为{self.oov_id}"
#            self.word_index = vocabulary
#        else:
        self.word_index = {self.pad_token: self.pad_id,
                           self.oov_token: self.oov_id}
        self.word_count = {self.pad_token: 0, self.oov_token: 0}

    def fit(self, tokenized_texts, prune_after_fit=True):
        """
        Parameters
        ----------
        tokenized_texts : list of list
            切词后的文本
        prune_after_fit: bool
            fit后是否裁剪词典，默认为True。如果特殊常见需要多次fit，可以自己控制裁剪的频率
        """
        for segs in tokenized_texts:
            for word in segs:
                if self.lower:
                    word = word.lower()
                if word not in self.word_index:
                    self.word_index[word] = len(self.word_index)
                if word not in self.word_count:
                    self.word_count[word] = 0
                self.word_count[word] += 1

        # 裁剪
        if not prune_after_fit:
            return
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

    def transform(self, tokenized_texts, padding=False, max_seq_len=None):
        """
        Parameters
        ----------
        tokenized_texts: list of list
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
        for tokens in tokenized_texts:
            # self[x] for self.__getitem__(x)
            token_ids = [self[x] for x in tokens]
            if padding:
                assert isinstance(max_seq_len, int), "max_seq_len未设置"
                token_ids = token_ids[:max_seq_len]
                if len(token_ids) < max_seq_len:
                    token_ids += [self.pad_id] * (max_seq_len - len(token_ids))
            yield token_ids

    def __len__(self):
        """
        词典大小
        """
        return len(self.word_index)

    def __getitem__(self, word):
        """
        返回词的id
        """
        if self.lower:
            return self.word_index.get(word.lower(), self.oov_id)
        return self.word_index.get(word, self.oov_id)

    def __contains__(self, word):
        if self.lower:
            return word.lower() in self.word_index
        return word in self.word_index
