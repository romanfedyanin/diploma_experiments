from __future__ import division
from nltk import FreqDist, tokenize
from NBC import NaiveBayesClassifier
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
import numpy as np
import json
from datetime import datetime

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
stemmer = PorterStemmer()

FIRST_INTERVAL = (1, 1000)
SECOND_INTERVAL = (1001, 2000)
THIRD_INTERVAL = (2001, 3000)

INTERVALS_ARRAY = [
    [[SECOND_INTERVAL, THIRD_INTERVAL], [FIRST_INTERVAL]],
    [[FIRST_INTERVAL, THIRD_INTERVAL], [SECOND_INTERVAL]],
    [[FIRST_INTERVAL, SECOND_INTERVAL], [THIRD_INTERVAL]]
]

TEXTS_FREQ_DISTS = {}

EXCLUDE_WORDS = False


def number_inside_intervals(number, intervals):
    for interval in intervals:
        if interval[0] <= number <= interval[1]:
            return True
    return False


def print_time():
    print datetime.now().strftime('%H:%M:%S')


def add_text_freq_dist(doc_text):
    if doc_text not in TEXTS_FREQ_DISTS:
        words = tokenizer.tokenize(doc_text)
        # lemmatize_words(words)
        freq_dist = FreqDist(words)
        TEXTS_FREQ_DISTS[doc_text] = freq_dist


class NegPosParser():

    labeled_docs = {}
    words_docs_count_dict = {}
    not_excluding_words = []
    excluding_words = []
    words_docs_count_dict = {}

    def add_text_to_corpus(self, doc_text, label):
        if EXCLUDE_WORDS:
            add_text_freq_dist(doc_text)
            for word in TEXTS_FREQ_DISTS[doc_text]:
                self.words_docs_count_dict[word] = self.words_docs_count_dict.get(word, 0.0) + 1.0
        self.labeled_docs[doc_text] = label


    def load_data(self, intervals):
        filenames = [
            "neg_tweets.json",
            "pos_tweets.json"
        ]
        # filenames = [
        #     "neg_reviews.json",
        #     "pos_reviews.json"
        # ]
        for i in range(len(filenames)):
            filename = filenames[i]
            json_data = open(filename)
            file_data = json.load(json_data)
            j = 1
            for text, label in file_data.iteritems():
                if number_inside_intervals(j, intervals):
                    self.add_text_to_corpus(text, label)
                j += 1

        if EXCLUDE_WORDS:
            for (word, value) in self.words_docs_count_dict.iteritems():
                if value > 1:
                    self.not_excluding_words.append(word)
                else:
                    self.excluding_words.append(word)

            print "Words count: " + str(len(self.words_docs_count_dict))
            print "Not excluding words count: " + str(len(self.not_excluding_words))

    def train(self):
        train_dict = self.labeled_docs
        all_feats = []
        for text, value in train_dict.items():
            try:
                tokens = tokenize.word_tokenize(text)

            except UnicodeDecodeError:
                print "Oops!  That was no valid text..."
            if EXCLUDE_WORDS:
                for word in self.excluding_words:
                    if tokens.__contains__(word):
                        tokens.remove(word)
            # tokens = nltk.tokenize.word_tokenize(text)
            feats = dict(((token, True) for token in tokens))
            if train_dict[text] == 1:
                all_feats.append((feats, 'pos'))
            else:
                all_feats.append((feats, 'neg'))

        self._classifier = NaiveBayesClassifier.train(all_feats)
        print "Most informative values:"
        print self._classifier.most_informative_features(20)


def lemmatize_words(words):
    for i in range(len(words)):
        words[i] = stemmer.stem(words[i])

def features_from_text(text, excluding_words):
    terms = tokenizer.tokenize(text)
    if EXCLUDE_WORDS:
        for word in excluding_words:
            if terms.__contains__(word):
                terms.remove(word)
    return dict(((token, True) for token in terms))

for i in range(len(INTERVALS_ARRAY)):
    intervals_for_learning = INTERVALS_ARRAY[i][0]
    test_intervals = INTERVALS_ARRAY[i][1]
    print "\n\nConfiguration " + str(i)
    print "Learning: loading data"
    print_time()

    parser = None
    test_parser = None
    svc = None
    Z = None

    parser = NegPosParser()
    parser.labeled_docs = {}
    parser.load_data(intervals_for_learning)
    print "Learning: training"
    print_time()
    parser.train()

    print "Test: loading data"
    print_time()
    test_parser = NegPosParser()
    test_parser.labeled_docs = {}
    test_parser.load_data(test_intervals)

    print "Test: classifying data"
    number_of_docs = 0
    right_classification_docs = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    wrong_classified_to_pos = 0
    wrong_classified_to_neg = 0
    docs_probs = []
    for doc in test_parser.labeled_docs:
        features = features_from_text(doc, parser.excluding_words)
        prob_dist = parser._classifier.prob_classify(features)
        neg_prob = prob_dist.prob('neg')
        pos_prob = prob_dist.prob('pos')
        docs_probs.append(("%.7f" % pos_prob, "%.7f" % neg_prob, "neg" if test_parser.labeled_docs[doc] == -1 else "pos"))
        label = parser._classifier.classify(features)
        number_of_docs += 1
        if (label == 'neg') & (test_parser.labeled_docs[doc] == -1):
            true_negative += 1
        elif (label == 'pos') & (test_parser.labeled_docs[doc] == 1):
            true_positive += 1
        else:
            label = parser._classifier.classify(features)
            if label == 'pos':
                false_positive += 1
            else:
                false_negative += 1

    print "Configuration " + str(i)
    print "Classified docs: " + str(number_of_docs)
    precision = true_positive/(true_positive + false_positive)
    print "Precision: " + str(precision)
    recall = true_positive/(true_positive + false_negative)
    print "Recall: " + str(recall)
    print "F-measure: " + str(2/(1/precision + 1/recall))
    print "Accuracy: " + str((true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive))
    # print "Right classified documents: " + str(right_classification_docs)
    # print "Wrong classified documents as positive: " + str(wrong_classified_to_pos)
    # print "Wrong classified documents as negative: " + str(wrong_classified_to_neg)

    print "Precision: " + str(right_classification_docs/number_of_docs)
    print_time()
    print "-----\n\n"
