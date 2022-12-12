"""
Python code implement the Snowball algorithm.
Paper: Snowball: Extracting Relations from Large Plain-Text Collections
PaperUrl: http://www.cs.columbia.edu/~gravano/Papers/2000/dl00.pdf

https://github.com/davidsbatista/Snowball
"""

from chaosNLP.preprocessing.text import Tokenizer


class PatternTuple:
    """
    挖掘到到两个实体的pattern tuple, 对应论文中的(before, e1, middle, e2, after)
    """
    def __init__(self, before, entity_a, type_a, middle, entity_b, type_b, after):
        """
        Params:
          before: string, pattern的前缀
          entity_a: string, 第一个实体
          type_a: string, 第一个实体类型
          middle: string, pattern的中间部分
          entity_b: string, 第二个实体
          type_b: string, 第二个实体类型
          after: string, pattern的后缀 
        """
        self.before = before
        self.entity_a = entity_a
        self.type_a = type_a
        self.middle = middle
        self.entity_b = entity_b
        self.type_b = type_b
        self.after = after


class Snowball:
    """
    1. generate_tuples
    2. bootstrapping
    """

    def __init__(self, langs,
                 min_tokens_away=1, max_tokens_away=6,
                 context_window_size=3,
                 number_iterations=4,
                 update_weight=0.5,
                 unknown_weight=0.1,
                 negative_weight=2,
                 similarity_threshold=0.6,
                 confidence_threshold=0.8,
                 alpha=0.2,
                 beta=0.6,
                 gamma=0.2,
                 ):
        """
        Params:
          langs: zh/en
            语料是中文或者英文
          min_tokens_away: int
            两个实体之间最小的词数
          max_tokens_away: int
            两个实体之间最大的词数
          context_window_size: int
            前缀和后嘴的最大词数
          update_weight: float, [0, 1]
            更新confidence时，最新分数的权重，历史的权重为1 - update_weight
          number_iterations: int
            number of iterations of the system
          unknown_weight: float, [0, 1]
            Weight given to unknown relationships extracted seeds. i.e., since they are not in the seed set, nothing can be said about them
          negative_weight: float,
            Weight given to negative seeds, i.e., negative examples of the relationships to be extracted
          similarity_threshold: float, [0, 1]
            threshold similarity for clustering/extracting instances
          confidence_threshold: float, [0, 1]
            confidence threshold of an instance to used as seed
          alpha: float, [0, 1]
            计算相似度时before的权重
          beta: float, [0, 1]
            计算相似度时between的权重
          gamma: float, [0, 1]
            计算相似度时after的权重

        """
        self.min_tokens_away = min_tokens_away
        self.max_tokens_away = max_tokens_away
        self.context_window_size = context_window_size
        self.number_iterations = number_iterations
        self.update_weight = update_weight
        self.unknown_weight = unknown_weight
        self.negative_weight = negative_weight
        self.similarity_threshold = similarity_threshold
        self.confidence_threshold = confidence_threshold
        self.alpha = alpha
        self.beta = beta
        self.gamma = gamma
        self.tokenizer = Tokenizer(langs=langs)

    def load_labeled_corpus(self, filename):
        """
        标注文件的格式样例如下：
        > The statements made today by euro deputy <PER>Martin Schulz</PER> at the
        > assembly in <LOC>Strasbourg</LOC> constitute a serious and unacceptable
        > insult to the dignity of the president of the ...
        """
        import re
        regex_simple = re.compile('<[A-Z]+>[^<]+</[A-Z]+>', re.U)
        regex_tag = '<([^]]+)>(.*)</([^]]+)>'

        patterns = []

        with open(filename) as f:
            for line in f:
                matches = []
                for m in re.finditer(regex_simple, line):
                    matches.append(m)
                if len(matches) < 2:
                    continue
                start = 0
                print(matches)
                for i in range(0, len(matches) - 1):
                    start = 0 if i == 0 else matches[i - 1].end()
                    end = len(line) if i + 2 >= len(matches) else matches[i + 2].start()

                    # 只取window_size个token
                    print(self.tokenizer.tokenize(line[start: matches[i].start()])[-self.context_window_size:])
                    before = ' '.join(self.tokenizer.tokenize(line[start: matches[i].start()])[-self.context_window_size:])
                    after = ' '.join(self.tokenizer.tokenize(line[matches[i + 1].end(): end])[:self.context_window_size])

                    between = line[matches[i].end(): matches[i + 1].start()]
                    n_bet_tokens = len(self.tokenizer.tokenize(between))
                    if n_bet_tokens < self.min_tokens_away or n_bet_tokens > self.max_tokens_away:
                        continue
                    
                    type_a, entity_a, _ = re.findall(regex_tag, matches[i].group())[0]
                    type_b, entity_b, _ = re.findall(regex_tag, matches[i + 1].group())[0]

                    patterns.append(PatternTuple(before, entity_a, type_a,
                                                 between, entity_b, type_b, after))

        return patterns
