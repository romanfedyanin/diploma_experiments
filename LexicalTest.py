 #!/usr/bin/python
 # -*- coding: utf-8 -*-
from __future__ import division
import json

from nltk import FreqDist
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
import pymorphy2

from CommonSettings import *

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
stemmer = PorterStemmer()
morphAnalyzer = pymorphy2.MorphAnalyzer()

corpus = Corpus(Corpus.russian_reviews)
INTERVALS_ARRAY = corpus.intervals_array()
CORPUS_FILE_NAMES = corpus.file_names()

docs = {}
pos_neg_docs = {}
words_docs_count_dict = {}
words_docs_pos_count_dict = {}
words_docs_neg_count_dict = {}

LEMMATIZE = True
N_DIFF = 40
N_TIMES_DIFF = 1.3
MIN_TIMES = 2
GOOD_WORDS_COUNTS_ARRAY = [1000] # Различное количество "хорошо характерихующих" слов (good words)

def load_data(intervals):
    global docs
    docs = {}
    global pos_neg_docs
    pos_neg_docs = {}
    global words_docs_count_dict
    words_docs_count_dict = {}
    global words_docs_pos_count_dict
    words_docs_pos_count_dict = {}
    global words_docs_neg_count_dict
    words_docs_neg_count_dict = {}

    for i in range(len(CORPUS_FILE_NAMES)):
        filename = CORPUS_FILE_NAMES[i]
        json_data = open(filename)
        file_data = json.load(json_data)
        j = 1
        for text, label in file_data.iteritems():
            if number_inside_intervals(j, intervals):
                add_text_to_corpus(text, label)
            j += 1


def change_docs_dict():
    global docs
    global pos_neg_docs
    pos_neg_docs = {}
    new_docs = {}
    for key in docs:
        pos_neg_docs[key] = docs[key]
        if LEMMATIZE:
            words = [morphAnalyzer.parse(word)[0].normal_form for word in tokenizer.tokenize(key)]
        else:
            words = tokenizer.tokenize(key)
        freq_dict = FreqDist(words)
        normalize_freq_dist(freq_dict, words)
        new_docs[key] = freq_dict
    docs = new_docs


def add_text_to_corpus(text, negpos):
    global docs
    text = text.lower()
    docs[text] = negpos
    if LEMMATIZE:
        words = [morphAnalyzer.parse(word)[0].normal_form for word in tokenizer.tokenize(text)]
    else:
        words = tokenizer.tokenize(text)
    freq_dist = FreqDist(words)
    for k in freq_dist:
        words_docs_count_dict[k] = words_docs_count_dict.get(k, 0.0) + 1.0
        if negpos == 1:
            words_docs_pos_count_dict[k] = words_docs_pos_count_dict.get(k, 0.0) + 1.0
        else:
            words_docs_neg_count_dict[k] = words_docs_neg_count_dict.get(k, 0.0) + 1.0


def normalize_freq_dist(freq_dist, words):
    length = len(words)
    for k in freq_dist:
        freq_dist[k] /= length


#-------Experiments for different configurations-----------

for i in range(len(INTERVALS_ARRAY)):
    intervals_for_learning = INTERVALS_ARRAY[i][0]
    test_intervals = INTERVALS_ARRAY[i][1]
    print "Start loading learning data"
    load_data(intervals_for_learning)
    print "Loading finished \n\n"

    good_pos_words_array = []
    good_neg_words_array = []

    for key in words_docs_neg_count_dict:
        if key in words_docs_pos_count_dict:
            if (words_docs_neg_count_dict[key] / words_docs_pos_count_dict[key] > N_TIMES_DIFF) & (words_docs_pos_count_dict[key] > MIN_TIMES):
                good_neg_words_array.append((key, words_docs_pos_count_dict[key], words_docs_neg_count_dict[key], words_docs_neg_count_dict[key] / words_docs_pos_count_dict[key]))
            elif (words_docs_pos_count_dict[key] / words_docs_neg_count_dict[key] > N_TIMES_DIFF) & (words_docs_neg_count_dict[key] > MIN_TIMES):
                good_pos_words_array.append((key, words_docs_pos_count_dict[key], words_docs_neg_count_dict[key], words_docs_pos_count_dict[key] / words_docs_neg_count_dict[key]))

    good_pos_words_array.sort(key=lambda tup: tup[3], reverse=True)
    good_neg_words_array.sort(key=lambda tup: tup[3], reverse=True)

    print "Positive good words count: " + str(len(good_pos_words_array))
    print "Negative good words count: " + str(len(good_neg_words_array))
    print "\n\n"

    print "Start loading test data"
    load_data(test_intervals)
    print "Loading finished \n\n"

    print "Changing doc dict"
    change_docs_dict()
    print "Finished \n\n"

    for j in range(len(GOOD_WORDS_COUNTS_ARRAY)):
        good_words_count = GOOD_WORDS_COUNTS_ARRAY[j]

        top_good_pos_words_array = list(x[0] for x in good_pos_words_array[:good_words_count])
        top_good_neg_words_array = list(x[0] for x in good_neg_words_array[:good_words_count])

        true_positive = 0
        false_negative = 0
        true_negative = 0
        false_positive = 0
        equal_count_positive_text = 0
        equal_count_negative_text = 0
        for key in docs:
            number_of_good_pos_words_in_doc = 0
            number_of_good_neg_words_in_doc = 0
            doc_dict = docs[key]
            for word in top_good_pos_words_array:
                if word in doc_dict:
                    number_of_good_pos_words_in_doc += 1
            for word in top_good_neg_words_array:
                if word in doc_dict:
                    number_of_good_neg_words_in_doc += 1
            if pos_neg_docs[key] == 1:
                if number_of_good_pos_words_in_doc == number_of_good_neg_words_in_doc:
                    equal_count_positive_text += 1
                elif number_of_good_pos_words_in_doc > number_of_good_neg_words_in_doc:
                    true_positive += 1
                else:
                    false_negative += 1
            else:
                if number_of_good_pos_words_in_doc == number_of_good_neg_words_in_doc:
                    equal_count_negative_text += 1
                if number_of_good_pos_words_in_doc < number_of_good_neg_words_in_doc:
                    true_negative += 1
                else:
                    false_positive += 1
        true_positive += int(equal_count_positive_text * 0.5)
        false_negative += int(equal_count_positive_text * 0.5)
        true_negative += int(equal_count_negative_text * 0.5)
        false_positive += int(equal_count_negative_text * 0.5)

        print "\n\n--------------Results--------------"
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
        # csv_filename = "KW_configuration_" + str(i) + "_"
        # csv_filename += "russian_reviews.csv"
        # with open(csv_filename, "wb") as text_file:
        #     text_file.write(csv_string)



