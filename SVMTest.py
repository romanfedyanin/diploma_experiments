from __future__ import division
import os
from nltk import FreqDist
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn import svm, datasets
import json
from datetime import datetime

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
stemmer = PorterStemmer()

FIRST_INTERVAL = (1, 100)
SECOND_INTERVAL = (101, 200)
THIRD_INTERVAL = (201, 300)

INTERVALS_ARRAY = [
    [[SECOND_INTERVAL, THIRD_INTERVAL], [FIRST_INTERVAL]],
    [[FIRST_INTERVAL, THIRD_INTERVAL], [SECOND_INTERVAL]],
    [[FIRST_INTERVAL, SECOND_INTERVAL], [THIRD_INTERVAL]]
]

TEXTS_FREQ_DISTS = {}


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
    features_words = []

    def add_text_to_corpus(self, doc_text, label):
        self.labeled_docs[doc_text] = label

    def choose_features(self):
        self.words_docs_count_dict = {}
        for doc_text, label in self.labeled_docs.iteritems():
            add_text_freq_dist(doc_text)
            for k in TEXTS_FREQ_DISTS[doc_text]:
                self.words_docs_count_dict[k] = self.words_docs_count_dict.get(k, 0.0) + 1.0
        self.features_words = self.words_docs_count_dict.keys()

    def load_data(self, intervals):
        filenames = [
            "neg_reviews.json",
            "pos_reviews.json"
        ]
        for i in range(len(filenames)):
            filename = filenames[i]
            json_data = open(filename)
            file_data = json.load(json_data)
            j = 1
            for text, label in file_data.iteritems():
                if number_inside_intervals(j, intervals):
                    self.add_text_to_corpus(text, label)
                j += 1

    def get_data_for_svm(self):
        X = []
        Y = []
        for doc_text, label in self.labeled_docs.iteritems():
            features = [0] * len(self.features_words)
            add_text_freq_dist(doc_text)
            for key in TEXTS_FREQ_DISTS[doc_text]:
                index = self.features_words.index(key)
                features[index] = 1
            X.append(features)
            Y.append(label)
        return (X, Y)


def lemmatize_words(words):
    for i in range(len(words)):
        words[i] = stemmer.stem(words[i])


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
    print "Learning: choosing features"
    print_time()
    parser.choose_features()
    print "Learning: get data for svm"
    print_time()
    (data, target) = parser.get_data_for_svm()

    print "Learning: learning svm"
    print_time()
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(data, target)

    print "Test: loading data"
    print_time()
    test_parser = NegPosParser()
    test_parser.labeled_docs = {}
    test_parser.load_data(test_intervals)

    print "Test: preparing data"
    print_time()
    X = []
    for doc_text, label in test_parser.labeled_docs.iteritems():
        features = [0] * len(parser.features_words)
        add_text_freq_dist(doc_text)
        for key in TEXTS_FREQ_DISTS[doc_text]:
            if key in parser.features_words:
                index = parser.features_words.index(key)
                features[index] = 1
        X.append(features)
    print "Test: classifying data"
    print_time()
    Z = svc.predict(X)
    right_classified_docs = 0
    classified_docs = 0
    i = 0
    for doc_text, label in test_parser.labeled_docs.iteritems():
        classified_docs += 1
        if Z[i] == label:
            right_classified_docs += 1
        i += 1

    print "Classified docs: " + str(classified_docs)
    print "Right classified docs: " + str(right_classified_docs)
    print "Precision: " + str(right_classified_docs/classified_docs)
    print_time()
    print "-----\n\n"
