from nltk.probability import FreqDist
from nltk.probability import *
from nltk.corpus import brown
from collections import Counter


# print(len(brown.words()))
# print(FreqDist(brown.words()))

# counter_obj = Counter(brown.words())
# print(counter_obj)

# words = FreqDist()
corpus_word_dict = {}
for sentence in brown.sents():
    for word in sentence:
        count = corpus_word_dict.get(word.lower(), -1)
        if count == -1:
            corpus_word_dict[word.lower()] = 1
        else:
            corpus_word_dict[word.lower()] = count + 1

print(corpus_word_dict["and"])
