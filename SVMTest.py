from __future__ import division
import json

from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
from sklearn import svm
import pymorphy2

from CommonSettings import *

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
morphAnalyzer = pymorphy2.MorphAnalyzer()

TEXTS_FREQ_DISTS = {}

corpus = Corpus(Corpus.russian_reviews)
INTERVALS_ARRAY = corpus.intervals_array()
CORPUS_FILE_NAMES = corpus.file_names()

LEMMATIZE = True


def add_text_freq_dist(doc_text):
    if doc_text not in TEXTS_FREQ_DISTS:
        if LEMMATIZE:
            words = [morphAnalyzer.parse(word)[0].normal_form for word in tokenizer.tokenize(doc_text)]
        else:
            words = tokenizer.tokenize(doc_text)
        freq_dist = FreqDist(words)
        TEXTS_FREQ_DISTS[doc_text] = freq_dist


class NegPosParser():

    def __init__(self):
        pass

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

        for i in range(len(CORPUS_FILE_NAMES)):
            filename = CORPUS_FILE_NAMES[i]
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

    classified_docs = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    result_dict_for_json = {}
    j = 0
    for doc_text, label in test_parser.labeled_docs.iteritems():
        classified_docs += 1

        if Z[j] == label:
            result_dict_for_json[doc_text] = 1
            if label == 1:
                true_positive += 1
            else:
                true_negative += 1
        else:
            result_dict_for_json[doc_text] = 0
            if label == 1:
                false_positive += 1
            else:
                false_negative += 1
        j += 1

    with open("result_dict_svm_russian.json", 'w') as outfile:
        json.dump(result_dict_for_json, outfile)

    print "Classified docs: " + str(classified_docs)
    precision_pos = true_positive/(true_positive + false_positive)
    precision_neg = true_negative/(true_negative + false_negative)
    macro_precision = (precision_pos + precision_neg) * 0.5
    print "Macro Precision: " + str(macro_precision)
    recall_pos = true_positive/(true_positive + false_negative)
    recall_neg = true_negative/(true_negative + false_positive)
    macro_recall = (recall_pos + recall_neg) * 0.5
    print "Macro Recall: " + str(macro_recall)
    macro_fmeasure = 2*macro_recall*macro_precision/(macro_precision + macro_recall)
    print "Macro F-measure: " + str(macro_fmeasure)
    accuracy = (true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive)
    print "Accuracy: " + str(accuracy)
    print "---------------------------------------\n\n"

    # csv_string = "%.3f;%.3f;%.3f;%.3f" % (macro_precision, macro_recall, macro_fmeasure, accuracy)
    # csv_filename = "SVM_configuration_" + str(i) + "_"
    # csv_filename += "russian_reviews.csv"
    # with open(csv_filename, "wb") as text_file:
    #     text_file.write(csv_string)

