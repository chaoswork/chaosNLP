from chaosNLP.data import VocabularyMapper


def test_lower():
    vocab_mapper = VocabularyMapper()
    tokenized_texts = [
        ['This', 'is'],
        ['THIS', 'IS'],
    ]
    vocab_mapper.fit(tokenized_texts)

    assert len(vocab_mapper) == len(vocab_mapper.special_tokens) + 2


def test_no_lower():
    vocab_mapper = VocabularyMapper(lower=False)
    tokenized_texts = [
        ['This', 'is'],
        ['THIS', 'IS'],
    ]
    vocab_mapper.fit(tokenized_texts)

    assert len(vocab_mapper) == len(vocab_mapper.special_tokens) + 4


def test_prune():
    vocab_mapper = VocabularyMapper(max_word_nums=4)
    tokenized_texts = [
        ['one', 'two', 'three', 'four', 'five'],
        ['two', 'three', 'four', 'five'],
        ['three', 'four', 'five'],
        ['four', 'five'],
        ['five'],
    ]
    # only 'four', 'five' survived
    vocab_mapper.fit(tokenized_texts)
    assert 'one' not in vocab_mapper
    assert 'two' not in vocab_mapper
    assert 'three' not in vocab_mapper
    assert vocab_mapper['four'] == 2
    assert vocab_mapper['five'] == 3


def test_not_prune():
    vocab_mapper = VocabularyMapper(max_word_nums=4)
    tokenized_texts = [
        ['one', 'two', 'three', 'four', 'five'],
        ['two', 'three', 'four', 'five'],
        ['three', 'four', 'five'],
        ['four', 'five'],
        ['five'],
    ]
    vocab_mapper.fit(tokenized_texts, prune_after_fit=False)
    assert vocab_mapper['one'] == len(vocab_mapper.special_tokens) + 0
    assert vocab_mapper['two'] == len(vocab_mapper.special_tokens) + 1
    assert vocab_mapper['three'] == len(vocab_mapper.special_tokens) + 2
    assert vocab_mapper['four'] == len(vocab_mapper.special_tokens) + 3
    assert vocab_mapper['five'] == len(vocab_mapper.special_tokens) + 4


def test_fit_transform():
    tokenized_texts = [
        'This is the first line , the start line'.split(),
        'the second line'.split(),
        'this is the third line , the last line'.split()
    ]
    vocab_mapper = VocabularyMapper()
    vocab_mapper.fit(tokenized_texts)
    seqs = list(vocab_mapper.transform(tokenized_texts))
    # This/2 is/3 the/4 first/5 line/6 ,/7 the/4 start/8 line/6
    assert seqs[0] == [2, 3, 4, 5, 6, 7, 4, 8, 6]
    # the/4 second/9 line/6
    assert seqs[1] == [4, 9, 6]
    # this/2 is/3 the/4 third/10 line/6 ,/7 the/4 last/11 line/6
    assert seqs[2] == [2, 3, 4, 10, 6, 7, 4, 11, 6]


def test_fit_two_times():
    tokenized_texts_a = [
        'This is the first line , the start line'.split(),
        'the second line'.split(),
    ]
    tokenized_texts_b = [
        'this is the third line , the last line'.split()
    ]
    vocab_mapper = VocabularyMapper()
    vocab_mapper.fit(tokenized_texts_a)
    vocab_mapper.fit(tokenized_texts_b)
    seqs = list(vocab_mapper.transform(tokenized_texts_a + tokenized_texts_b))
    # This/2 is/3 the/4 first/5 line/6 ,/7 the/4 start/8 line/6
    assert seqs[0] == [2, 3, 4, 5, 6, 7, 4, 8, 6]
    # the/4 second/9 line/6
    assert seqs[1] == [4, 9, 6]
    # this/2 is/3 the/4 third/10 line/6 ,/7 the/4 last/11 line/6
    assert seqs[2] == [2, 3, 4, 10, 6, 7, 4, 11, 6]
