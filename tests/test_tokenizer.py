import sys
sys.path.append('..')

from chaosNLP.preprocessing.text import Tokenizer


def test_default_tokenizer():
    tokenizer = Tokenizer()
    test_texts = "经济保持恢复发展。\n国内生产总值达到114万亿元，增长8.1%。\n全国财政收入突破20万亿元，增长10.7%。"
    tokenizer.fit_texts(test_texts)
    assert tokenizer.word_index['<pad>'] == 0
    assert tokenizer.word_index['<unk>'] == 1
    assert tokenizer.word_index['经济'] == 2
    seqs = list(tokenizer.text_to_seqs(test_texts))
    assert len(seqs) == 3
    assert seqs[0] == [2, 3, 4, 5, 6]
    assert seqs[0][-1] == tokenizer.word_index['。']


def test_tokenizer_with_stopwords():
    tokenizer = Tokenizer(stopwords='zh')
    test_texts = "经济保持恢复发展。\n国内生产总值达到114万亿元，增长8.1%。\n全国财政收入突破20万亿元，增长10.7%。"
    tokenizer.fit_texts(test_texts)
    assert tokenizer.word_index['<pad>'] == 0
    assert tokenizer.word_index['<unk>'] == 1
    assert tokenizer.word_index['经济'] == 2
    seqs = list(tokenizer.text_to_seqs(test_texts))
    assert len(seqs) == 3
    assert seqs[0] == [2, 3, 4]
    assert seqs[0][0] == tokenizer.word_index['经济']
    assert seqs[0][1] == tokenizer.word_index['恢复']
    assert seqs[0][2] == tokenizer.word_index['发展']


def test_tokenizer_padding():
    tokenizer = Tokenizer(stopwords='zh')
    test_texts = "经济保持恢复发展。\n国内生产总值达到114万亿元，增长8.1%。\n全国财政收入突破20万亿元，增长10.7%。"
    tokenizer.fit_texts(test_texts)
    assert tokenizer.word_index['<pad>'] == 0
    assert tokenizer.word_index['<unk>'] == 1
    assert tokenizer.word_index['经济'] == 2
    seqs = list(tokenizer.text_to_seqs(test_texts, padding=True, max_seq_len=32))
    assert len(seqs) == 3
    assert len(seqs[0]) == 32
    assert len(seqs[1]) == 32
    assert len(seqs[2]) == 32


def test_tokenizer_max_words_num():
    n = 5
    tokenizer = Tokenizer(max_word_nums=n)
    test_texts = [
        "经济保持恢复发展。",
        "经济快速发展",
        "经济保持"
    ]
    tokenizer.fit_texts(test_texts)
    assert tokenizer.word_index['<pad>'] == 0
    assert tokenizer.word_index['<unk>'] == 1
    assert len(tokenizer.word_count) == n
    assert len(tokenizer.word_index) == n
    for word in ['<pad>', '<unk>', '经济', '保持', '发展']:
        assert word in tokenizer.word_index
        assert word in tokenizer.word_count
        assert tokenizer.word_index[word] < n
