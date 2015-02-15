from __future__ import division
import os
from nltk import FreqDist
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
import numpy as np
from sklearn import svm, datasets

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
stemmer = PorterStemmer()

FIRST_INTERVAL = (1, 200)
SECOND_INTERVAL = (201, 400)
THIRD_INTERVAL = (401, 600)

INTERVALS_ARRAY = [
    [[SECOND_INTERVAL, THIRD_INTERVAL], [FIRST_INTERVAL]],
    [[FIRST_INTERVAL, THIRD_INTERVAL], [SECOND_INTERVAL]],
    [[FIRST_INTERVAL, SECOND_INTERVAL], [THIRD_INTERVAL]]
]


def number_inside_intervals(number, intervals):
    for interval in intervals:
        if interval[0] <= number <= interval[1]:
            return True
    return False


class NegPosParser():

    labeled_docs = {}
    words_docs_count_dict = {}
    features_words = []

    def add_text_to_corpus(self, doc_text, label):
        self.labeled_docs[doc_text] = label

    def choose_features(self):
        self.words_docs_count_dict = {}
        for doc_text, label in self.labeled_docs.iteritems():
            words = tokenizer.tokenize(doc_text)
            lemmatize_words(words)
            freq_dist = FreqDist(words)
            for k in freq_dist:
                self.words_docs_count_dict[k] = self.words_docs_count_dict.get(k, 0.0) + 1.0
        self.features_words = self.words_docs_count_dict.keys()

    def load_data(self, intervals):
        settings_array = [
            {
                "path": "/neg",
                "negpos": -1
            },
            {
                "path": "/pos",
                "negpos": 1
            }
        ]
        for i in range(len(settings_array)):
            setting_dict = settings_array[i]
            dir_name = os.path.dirname(os.path.abspath(__file__)) + setting_dict["path"]
            file_names = os.listdir(dir_name)
            for j in range(len(file_names)):
                if number_inside_intervals(j, intervals):
                    file_name = file_names[j]
                    if file_name.endswith(".txt"):
                        with open(dir_name + "/" + file_name, "r") as myfile:
                            data = myfile.read().replace('\n', '')
                            self.add_text_to_corpus(data, setting_dict["negpos"])

    def get_data_for_svm(self):
        X = []
        Y = []
        for doc_text, label in self.labeled_docs.iteritems():
            features = []
            words = tokenizer.tokenize(doc_text)
            lemmatize_words(words)
            freq_dist = FreqDist(words)
            for value in self.features_words:
                if freq_dist[value] > 0:
                    features.append(1)
                else:
                    features.append(0)
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
    print "Start loading learning data"

    parser = None
    test_parser = None
    svc = None
    Z = None

    parser = NegPosParser()
    parser.labeled_docs = {}
    parser.load_data(intervals_for_learning)
    parser.choose_features()
    (data, target) = parser.get_data_for_svm()

    print "Start learning svm"
    C = 1.0  # SVM regularization parameter
    svc = svm.SVC(kernel='linear', C=C).fit(data, target)

    print "Start loading test data"
    test_parser = NegPosParser()
    test_parser.labeled_docs = {}
    test_parser.load_data(test_intervals)

    print "Classification of test data"
    X = []
    for doc_text, label in test_parser.labeled_docs.iteritems():
        features = []
        words = tokenizer.tokenize(doc_text)
        lemmatize_words(words)
        freq_dist = FreqDist(words)
        for value in parser.features_words:
            if freq_dist[value] > 0:
                features.append(1)
            else:
                features.append(0)
        X.append(features)
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
    print "-----\n\n"




