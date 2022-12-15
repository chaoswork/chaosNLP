from chaosNLP.preprocessing.text import textReader


def test_default_text_reader():
    text_reader = textReader()
    test_texts = "经济保持恢复发展。\n国内生产总值达到114万亿元，增长8.1%。\n全国财政收入突破20万亿元，增长10.7%。"
    text_reader.fit_texts(test_texts)
    assert text_reader.word_index['<pad>'] == 0
    assert text_reader.word_index['<unk>'] == 1
    assert text_reader.word_index['经济'] == 2
    seqs = list(text_reader.text_to_seqs(test_texts))
    assert len(seqs) == 3
    assert seqs[0] == [2, 3, 4, 5, 6]
    assert seqs[0][-1] == text_reader.word_index['。']


def test_text_reader_with_stopwords():
    text_reader = textReader(stopwords='zh')
    test_texts = "经济保持恢复发展。\n国内生产总值达到114万亿元，增长8.1%。\n全国财政收入突破20万亿元，增长10.7%。"
    text_reader.fit_texts(test_texts)
    assert text_reader.word_index['<pad>'] == 0
    assert text_reader.word_index['<unk>'] == 1
    assert text_reader.word_index['经济'] == 2
    seqs = list(text_reader.text_to_seqs(test_texts))
    assert len(seqs) == 3
    assert seqs[0] == [2, 3, 4]
    assert seqs[0][0] == text_reader.word_index['经济']
    assert seqs[0][1] == text_reader.word_index['恢复']
    assert seqs[0][2] == text_reader.word_index['发展']


def test_text_reader_padding():
    text_reader = textReader(stopwords='zh')
    test_texts = "经济保持恢复发展。\n国内生产总值达到114万亿元，增长8.1%。\n全国财政收入突破20万亿元，增长10.7%。"
    text_reader.fit_texts(test_texts)
    assert text_reader.word_index['<pad>'] == 0
    assert text_reader.word_index['<unk>'] == 1
    assert text_reader.word_index['经济'] == 2
    seqs = list(text_reader.text_to_seqs(test_texts, padding=True,
                                         max_seq_len=32))
    assert len(seqs) == 3
    assert len(seqs[0]) == 32
    assert len(seqs[1]) == 32
    assert len(seqs[2]) == 32


def test_text_reader_max_words_num():
    n = 5
    text_reader = textReader(max_word_nums=n)
    test_texts = [
        "经济保持恢复发展。",
        "经济快速发展",
        "经济保持"
    ]
    text_reader.fit_texts(test_texts)
    assert text_reader.word_index['<pad>'] == 0
    assert text_reader.word_index['<unk>'] == 1
    assert len(text_reader.word_count) == n
    assert len(text_reader.word_index) == n
    for word in ['<pad>', '<unk>', '经济', '保持', '发展']:
        assert word in text_reader.word_index
        assert word in text_reader.word_count
        assert text_reader.word_index[word] < n


def test_text_reader_max_min_freq():

    text_reader = textReader(min_freq=2, max_freq=3)
    test_texts = [
        "经济保持恢复发展。",
        "经济快速发展",
        "经济保持"
        "经济保持"
    ]
    text_reader.fit_texts(test_texts)
    assert text_reader.word_index['<pad>'] == 0
    assert text_reader.word_index['<unk>'] == 1
    word_reminds = ['保持', '发展', '<pad>', '<unk>']
    assert len(text_reader.word_count) == len(word_reminds)
    assert len(text_reader.word_index) == len(word_reminds)
    for word in word_reminds:
        assert word in text_reader.word_index
        assert word in text_reader.word_count
        assert text_reader.word_index[word] < len(word_reminds)


def test_text_reader_fit_transform():
    corpus = [
        'This is the first line, the start line',
        'the second line',
        'this is the third line, the last line'
    ]
    text_reader = textReader(langs='en')
    seqs = list(text_reader.fit_texts(corpus, return_seqs=True))
    # This/2 is/3 the/4 first/5 line/6 ,/7 the/4 start/8 line/6
    assert seqs[0] == [2, 3, 4, 5, 6, 7, 4, 8, 6]
    # the/4 second/9 line/6
    assert seqs[1] == [4, 9, 6]
    # this/2 is/3 the/4 third/10 line/6 ,/7 the/4 last/11 line/6
    assert seqs[2] == [2, 3, 4, 10, 6, 7, 4, 11, 6]


def test_text_reader_count_vector():
    import numpy as np
    text_reader = textReader(langs='en')

    corpus = [
        'This is the first line, the start line',
        'the second line',
        'this is the third line, the last line'
    ]
    text_reader.fit_texts(corpus)
    count_vector = text_reader.text_to_count_vector(corpus)
    assert count_vector.shape == (len(corpus), len(text_reader.word_index))
    print(text_reader.word_index)
    print(count_vector.toarray())
    assert np.array_equal(count_vector.toarray()[0],
                          [0, 0, 1, 1, 2, 1, 2, 1, 1, 0, 0, 0])
    assert np.array_equal(count_vector.toarray()[1],
                          [0, 0, 0, 0, 1, 0, 1, 0, 0, 1, 0, 0])
    assert np.array_equal(count_vector.toarray()[2],
                          [0, 0, 1, 1, 2, 0, 2, 1, 0, 0, 1, 1])
