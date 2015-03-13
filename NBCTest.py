 #!/usr/bin/python
 # -*- coding: utf-8 -*-
from __future__ import division
import json

from nltk import FreqDist
from nltk.tokenize import RegexpTokenizer
import pymorphy2

from NBC import NaiveBayesClassifier
from CommonSettings import *

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
morphAnalyzer = pymorphy2.MorphAnalyzer()

corpus = Corpus(Corpus.russian_reviews)
INTERVALS_ARRAY = corpus.intervals_array() #интервалы текстов для обучения и тестирования
CORPUS_FILE_NAMES = corpus.file_names()

TEXTS_FREQ_DISTS = {} #частоты для слов в текстах (чтобы не считать для каждой конфигурации)

EXCLUDE_WORDS = True #опция "не рассматривать редкие слова" (менее 3 раз в корпусе)
LEMMATIZE = True #опция приведения слова к нормальной форме


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
    not_excluding_words = []
    excluding_words = []

    def add_text_to_corpus(self, doc_text, label):
        if EXCLUDE_WORDS:
            add_text_freq_dist(doc_text)
            for word in TEXTS_FREQ_DISTS[doc_text]:
                self.words_docs_count_dict[word] = self.words_docs_count_dict.get(word, 0.0) + 1.0
        self.labeled_docs[doc_text] = label

    def clear(self):
        self.labeled_docs = {}
        self.words_docs_count_dict = {}
        self.excluding_words = []
        self.not_excluding_words = []

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

        if EXCLUDE_WORDS:
            for (word, value) in self.words_docs_count_dict.iteritems():
                if value > 2:
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
                if LEMMATIZE:
                    tokens = [morphAnalyzer.parse(word)[0].normal_form.lower() for word in tokenizer.tokenize(text)]
                else:
                    tokens = [token.lower() for token in tokenizer.tokenize(text)]
            except UnicodeDecodeError:
                print "Oops!  That was no valid text..."
            if EXCLUDE_WORDS:
                for word in self.excluding_words:
                    if tokens.__contains__(word):
                        tokens.remove(word)
            feats = dict(((token, True) for token in tokens))
            if train_dict[text] == 1:
                all_feats.append((feats, 'pos'))
            else:
                all_feats.append((feats, 'neg'))

        self._classifier = NaiveBayesClassifier.train(all_feats)


def features_from_text(text, excluding_words):
    if LEMMATIZE:
        terms = [morphAnalyzer.parse(word)[0].normal_form for word in tokenizer.tokenize(text)]
    else:
        terms = tokenizer.tokenize(text)
    if EXCLUDE_WORDS:
        for word in excluding_words:
            if terms.__contains__(word):
                terms.remove(word)
    return dict(((token, True) for token in terms))


#-------Experiments for different configurations-----------

for i in range(len(INTERVALS_ARRAY)):
    intervals_for_learning = INTERVALS_ARRAY[i][0]
    test_intervals = INTERVALS_ARRAY[i][1]
    print "\n\nConfiguration " + str(i)
    print "Learning: loading data"
    print_time()

    parser = None
    test_parser = None

    parser = NegPosParser()
    parser.clear()
    parser.load_data(intervals_for_learning)

    print "Learning: training"
    print_time()
    parser.train()

    print "Test: loading data"
    print_time()
    test_parser = NegPosParser()
    test_parser.clear()
    test_parser.load_data(test_intervals)

    print "Test: classifying data"
    number_of_docs = 0
    true_positive = 0
    true_negative = 0
    false_positive = 0
    false_negative = 0
    result_dict_for_json = {}
    for doc in test_parser.labeled_docs:
        features = features_from_text(doc, parser.excluding_words)
        prob_dist = parser._classifier.prob_classify(features)
        neg_prob = prob_dist.prob('neg')
        pos_prob = prob_dist.prob('pos')
        label = parser._classifier.classify(features)
        number_of_docs += 1
        if (label == 'neg') & (test_parser.labeled_docs[doc] == -1):
            true_negative += 1
            result_dict_for_json[doc] = 1
        elif (label == 'pos') & (test_parser.labeled_docs[doc] == 1):
            result_dict_for_json[doc] = 1
            true_positive += 1
        else:
            result_dict_for_json[doc] = 0
            label = parser._classifier.classify(features)
            if label == 'pos':
                false_positive += 1
            else:
                false_negative += 1

# results:
    with open("result_dict_russian_nbc.json", 'w') as outfile:
        json.dump(result_dict_for_json, outfile)

    print "Configuration " + str(i)
    print "Classified docs: " + str(number_of_docs)
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
    # csv_filename = "NBC_configuration_" + str(i) + "_"
    # csv_filename += "russian_reviews.csv"
    # with open(csv_filename, "wb") as text_file:
    #     text_file.write(csv_string)

