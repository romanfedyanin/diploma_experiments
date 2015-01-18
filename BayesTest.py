__author__ = 'romanfedyanin'
import os
import nltk
from NBC import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
import math

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)


class BaseParser(object):
    """Parser returns dict {sentence:polarity}
    """

    def loaddata(self):
        """Parses the text."""
        pass


class NegPosParser(BaseParser):

    labeled_docs = {}

    def loaddata(self):
        result_dict = {}
        dir_name = os.path.dirname(os.path.abspath(__file__)) + "/neg"
        # sum = 0
        for file in os.listdir(dir_name):
            if file.endswith(".txt"):
                print(file)
                with open(dir_name + "/" + file, "r") as myfile:
                    data = myfile.read().replace('\n', '')
                    # sum += 1
                    # if sum < 100:
                    result_dict[data] = -1
        dir_name = os.path.dirname(os.path.abspath(__file__)) + "/pos"
        # sum = 0
        for file in os.listdir(dir_name):
            if file.endswith(".txt"):
                print(file)
                with open(dir_name + "/" + file, "r") as myfile:
                    data = myfile.read().replace('\n', '')
                    # sum += 1
                    # if sum < 100:
                    result_dict[data] = 1
        self.labeled_docs = result_dict.copy()
        return result_dict

    def train(self):
        train_dict = self.loaddata()
        all_feats = []
        for text, value in train_dict.items():
            tokens = nltk.tokenize.word_tokenize(text)
            # feats = [(token, True) for token in tokens]
            feats = dict(((token, True) for token in tokens))
            if train_dict[text] == 1:
                all_feats.append((feats, 'pos'))
            else:
                all_feats.append((feats, 'neg'))

        self._classifier = NaiveBayesClassifier.train(all_feats)


def write_array_of_triples_in_file(array, filename):
    f = open(filename,'wb')
    for i in range(len(array)):
        tuple = array[i]
        f.write(str(tuple[0]) + ";" + str(tuple[1]) + ";" + str(tuple[2]) +"\n")
    f.close()


def features_from_text(text):
    terms = tokenizer.tokenize(text)
    return dict(((token, True) for token in terms))

parser = NegPosParser()
parser.train()

number_of_docs = 0
right_classification_docs = 0
docs_probs = []
for doc in parser.labeled_docs:
    features = features_from_text(doc)
    prob_dist = parser._classifier.prob_classify(features)
    neg_prob = prob_dist.prob('neg')
    pos_prob = prob_dist.prob('pos')
    docs_probs.append(("%.4f" % pos_prob, "%.4f" % neg_prob, "neg" if parser.labeled_docs[doc] == -1 else "pos"))
    label = parser._classifier.classify(features)
    number_of_docs += 1
    if ((label == 'neg') & (parser.labeled_docs[doc] == -1)) | ((label == 'pos') & (parser.labeled_docs[doc] == 1)):
        right_classification_docs += 1

write_array_of_triples_in_file(docs_probs, "docs_probs.csv")

print 2

feats = {'great':True, 'good':True}
prob_dist = parser._classifier.prob_classify(feats)

classification=prob_dist.max()
p_pos=prob_dist.prob('pos')
p_neg=prob_dist.prob("neg")

print 1



from textblob.base import BaseSentimentAnalyzer
from textblob.en.sentiments import (DISCRETE, CONTINUOUS,
                                PatternAnalyzer, NaiveBayesAnalyzer)

analyzer = NaiveBayesAnalyzer()