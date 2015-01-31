__author__ = 'romanfedyanin'
import os
import nltk
from NBC import NaiveBayesClassifier
from nltk.tokenize import RegexpTokenizer
from numpy import genfromtxt
import numpy
import pandas as pd

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)


class BaseParser(object):
    """Parser returns dict {sentence:polarity}
    """

    def loaddata(self):
        """Parses the text."""
        pass


class NegPosParser(BaseParser):

    labeled_docs = {}

    def loaddata(self, intervals):
        result_dict = {}
        df = pd.read_csv('training.csv', sep=',',header=None)
        csv_strings = df.values
        print len(csv_strings)

        current_pos_string_number = 0
        current_neg_string_number = 0
        for i in range(len(csv_strings)):
            csv_string = csv_strings[i]
            if len(csv_string[5]) < 10:
                continue
            if csv_string[0] == 0:
                current_neg_string_number += 1
                negpos = -1
                if not number_inside_intervals(current_neg_string_number, intervals):
                    continue
            else:
                current_pos_string_number += 1
                negpos = 1
                if not number_inside_intervals(current_pos_string_number, intervals):
                    continue
            result_dict[csv_string[5]] = negpos

        self.labeled_docs = result_dict.copy()
        return result_dict

    def train(self, intervals):
        train_dict = self.loaddata(intervals)
        all_feats = []
        for text, value in train_dict.items():
            try:
                tokens = nltk.tokenize.word_tokenize(text)

            except UnicodeDecodeError:
                print "Oops!  That was no valid text..."
            # tokens = nltk.tokenize.word_tokenize(text)
            feats = dict(((token, True) for token in tokens))
            if train_dict[text] == 1:
                all_feats.append((feats, 'pos'))
            else:
                all_feats.append((feats, 'neg'))

        self._classifier = NaiveBayesClassifier.train(all_feats)
        print self._classifier.most_informative_features()


def write_array_of_triples_in_file(array, filename):
    f = open(filename,'wb')
    for i in range(len(array)):
        tuple = array[i]
        f.write(str(tuple[0]) + ";" + str(tuple[1]) + ";" + str(tuple[2]) +"\n")
    f.close()


def features_from_text(text):
    terms = tokenizer.tokenize(text)
    return dict(((token, True) for token in terms))


def number_inside_intervals(number, intervals):
    for interval in intervals:
        if interval[0] <= number <= interval[1]:
            return True
    return False

FIRST_INTERVAL = (1, 25000)
SECOND_INTERVAL = (25001, 50000)
THIRD_INTERVAL = (50001, 75000)
INTERVALS_ARRAY = [
    [[FIRST_INTERVAL, SECOND_INTERVAL], [THIRD_INTERVAL]],
    [[FIRST_INTERVAL, THIRD_INTERVAL], [SECOND_INTERVAL]],
    [[SECOND_INTERVAL, THIRD_INTERVAL], [FIRST_INTERVAL]]
]

for i in range(len(INTERVALS_ARRAY)):
    intervals_for_learning = INTERVALS_ARRAY[i][0]
    test_intervals = INTERVALS_ARRAY[i][1]

    parser = NegPosParser()
    parser.train(intervals_for_learning)


    test_parser = NegPosParser()
    test_parser.loaddata(test_intervals)

    number_of_docs = 0
    right_classification_docs = 0
    wrong_classified_to_pos = 0
    wrong_classified_to_neg = 0
    docs_probs = []
    for doc in test_parser.labeled_docs:
        features = features_from_text(doc)
        prob_dist = parser._classifier.prob_classify(features)
        neg_prob = prob_dist.prob('neg')
        pos_prob = prob_dist.prob('pos')
        docs_probs.append(("%.7f" % pos_prob, "%.7f" % neg_prob, "neg" if test_parser.labeled_docs[doc] == -1 else "pos"))
        label = parser._classifier.classify(features)
        number_of_docs += 1
        if ((label == 'neg') & (test_parser.labeled_docs[doc] == -1)) | ((label == 'pos') & (test_parser.labeled_docs[doc] == 1)):
            right_classification_docs += 1
        else:
            print "\nWrong classified:"
            print pos_prob
            print neg_prob
            print len(features)

            label = parser._classifier.classify(features)
            if label == 'pos':
                wrong_classified_to_pos += 1
            else:
                wrong_classified_to_neg += 1

    print "Configuration " + str(i)
    print "Number of documents: " + str(number_of_docs)
    print "Right classified documents: " + str(right_classification_docs)
    print "Wrong classified documents as positive: " + str(wrong_classified_to_pos)
    print "Wrong classified documents as negative: " + str(wrong_classified_to_neg)
    write_array_of_triples_in_file(docs_probs, "configuration_" + str(i) + "_docs_probs.csv")

