from __future__ import division
import os
import nltk
from nltk import FreqDist
from nltk.stem.porter import *
from nltk.tokenize import RegexpTokenizer
from collections import Counter

tokenizer = RegexpTokenizer(r'((?<=[^\w\s])\w(?=[^\w\s])|(\W))+', gaps=True)
stemmer = PorterStemmer()

number_of_documents = 0
number_of_pos_documents = 0
number_of_neg_documents = 0
docs = {}
pos_neg_docs = {}
words_docs_count_dict = {}
words_docs_pos_count_dict = {}
words_docs_neg_count_dict = {}

LEMMATIZE = True
N_DIFF = 40
GOOD_WORDS_COUNTS_ARRAY = [50]


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
    global number_of_documents
    number_of_documents = 0
    global number_of_pos_documents
    number_of_pos_documents = 0
    global number_of_neg_documents
    number_of_neg_documents = 0
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
                # print "add number #" + str(j)
                file_name = file_names[j]
                if file_name.endswith(".txt"):
                    with open(dir_name + "/" + file_name, "r") as myfile:
                        data = myfile.read().replace('\n', '')
                        add_text_to_corpus(data, setting_dict["negpos"])
        # dir_name = os.path.dirname(os.path.abspath(__file__)) + "/pos"
        # for file in os.listdir(dir_name):
        #     if file.endswith(".txt"):
        #         with open(dir_name + "/" + file, "r") as myfile:
        #             data = myfile.read().replace('\n', '')
        #             add_text_to_corpus(data, 1)


def number_inside_intervals(number, intervals):
    for interval in intervals:
        if interval[0] <= number <= interval[1]:
            return True
    return False


def change_docs_dict():
    global docs
    global pos_neg_docs
    pos_neg_docs = {}
    new_docs = {}
    for key in docs:
        pos_neg_docs[key] = docs[key]
        # if docs[key] == 1:
        #     continue
        words = tokenizer.tokenize(key)
        lemmatize_words(words)
        freq_dict = FreqDist(words)
        normalize_freq_dist(freq_dict, words)
        dict_of_words_in_text = {}
        for word in freq_dict:
            tfidf_all = number_of_documents * freq_dict[word] / words_docs_count_dict[word]
            dict_of_words_in_text[word] = tfidf_all
        new_docs[key] = dict_of_words_in_text
    docs = new_docs


def add_text_to_corpus(text, negpos):
    global number_of_documents, number_of_pos_documents, number_of_neg_documents
    number_of_documents += 1
    if negpos == 1:
        number_of_pos_documents += 1
    else:
        number_of_neg_documents += 1
    text = text.lower()
    docs[text] = negpos
    words = tokenizer.tokenize(text)
    lemmatize_words(words)
    freq_dist = FreqDist(words)
    for k in freq_dist:
        words_docs_count_dict[k] = words_docs_count_dict.get(k, 0.0) + 1.0
        if negpos == 1:
            words_docs_pos_count_dict[k] = words_docs_pos_count_dict.get(k, 0.0) + 1.0
        else:
            words_docs_neg_count_dict[k] = words_docs_neg_count_dict.get(k, 0.0) + 1.0


def number_of_words_in_text(text):
    return len(tokenizer.tokenize(text))


def normalize_freq_dist(freq_dist, words):
    length = len(words)
    for k in freq_dist:
        freq_dist[k] /= length


def make_text_with_tfidf(text):
    words = tokenizer.tokenize(text)
    lemmatize_words(words)
    freq_dict = FreqDist(words)
    normalize_freq_dist(freq_dict, words)

    for word in freq_dict:
        print freq_dict[word]
        print words_docs_count_dict[word]
        tfidf_all = number_of_documents * freq_dict[word] / words_docs_count_dict[word]
        text = text.replace(word + " ", word + "(%f) " % (tfidf_all, ))
    return text


def lemmatize_words(words):
    if not LEMMATIZE:
        return
    for i in range(len(words)):
        words[i] = stemmer.stem(words[i])


def write_array_of_tuples_in_file(array, filename):
    f = open(filename,'wb')
    for i in range(len(array)):
        tuple = array[i]
        f.write(str(tuple[0]) + ";" + str(tuple[1]) + "\n")
    f.close()


def write_array_of_triples_in_file(array, filename):
    f = open(filename,'wb')
    for i in range(len(array)):
        tuple = array[i]
        f.write(tuple[0] + ";" + str(tuple[1]) + ";" + str(tuple[2]) +"\n")
    f.close()


def write_array_in_file(array, filename):
    f = open(filename,'wb')
    for i in range(len(array)):
        f.write(str(array[i]) + "\n")
    f.close()


FIRST_INTERVAL = (1, 333)
SECOND_INTERVAL = (334, 666)
THIRD_INTERVAL = (667, 1000)
INTERVALS_ARRAY = [
    [[FIRST_INTERVAL, SECOND_INTERVAL], [THIRD_INTERVAL]],
    [[FIRST_INTERVAL, THIRD_INTERVAL], [SECOND_INTERVAL]],
    [[SECOND_INTERVAL, THIRD_INTERVAL], [FIRST_INTERVAL]]
]


for i in range(len(INTERVALS_ARRAY)):
    intervals_for_learning = INTERVALS_ARRAY[i][0]
    test_intervals = INTERVALS_ARRAY[i][1]
    print "Start loading learning data"
    load_data(intervals_for_learning)
    print "Loading finished \n\n"

    # words_docs_count_dict = Counter(words_docs_count_dict)
    # words_docs_pos_count_dict = Counter(words_docs_pos_count_dict)
    # words_docs_neg_count_dict = Counter(words_docs_neg_count_dict)

    good_pos_words_array = []
    good_neg_words_array = []

    for key in words_docs_neg_count_dict:
        if key not in words_docs_pos_count_dict:
            if words_docs_neg_count_dict[key] > N_DIFF:
                good_neg_words_array.append(key, 0, words_docs_neg_count_dict[key])
        elif words_docs_neg_count_dict[key] - words_docs_pos_count_dict[key] > N_DIFF:
            good_neg_words_array.append((key, words_docs_pos_count_dict[key], words_docs_neg_count_dict[key]))
        elif words_docs_pos_count_dict[key] - words_docs_neg_count_dict[key] > N_DIFF:
            good_pos_words_array.append((key, words_docs_pos_count_dict[key], words_docs_neg_count_dict[key]))
    for key in words_docs_pos_count_dict:
        if key not in words_docs_neg_count_dict:
            if words_docs_pos_count_dict[key] > N_DIFF:
                good_pos_words_array.append((key, words_docs_pos_count_dict[key], 0))

    good_pos_words_array.sort(key=lambda tup: tup[1], reverse=True)
    good_neg_words_array.sort(key=lambda tup: tup[2], reverse=True)

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
        # print "-----Testing of good words count " + str(good_words_count)

        top_good_pos_words_array = good_pos_words_array[:good_words_count]
        top_good_neg_words_array = good_neg_words_array[:good_words_count]
        # print "Positive words:"
        # print top_good_pos_words_array
        # print "Negative words:"
        # print top_good_neg_words_array

        top_good_pos_words_array = list(x[0] for x in top_good_pos_words_array)
        top_good_neg_words_array = list(x[0] for x in top_good_neg_words_array)

        pos_counts = []
        neg_counts = []
        for key in docs:
            number_of_good_pos_words_in_doc = 0
            number_of_good_neg_words_in_doc = 0
            doc_dict = docs[key]
            for word in top_good_pos_words_array:
                # word = top_good_pos_words_array[i]
                if word in doc_dict:
                    number_of_good_pos_words_in_doc += 1
            for word in top_good_neg_words_array:
                # word = top_good_neg_words_array[i]
                if word in doc_dict:
                    number_of_good_neg_words_in_doc += 1
            if pos_neg_docs[key] == 1:
                pos_counts.append((number_of_good_pos_words_in_doc, number_of_good_neg_words_in_doc))
            else:
                neg_counts.append((number_of_good_pos_words_in_doc, number_of_good_neg_words_in_doc))


        number_of_docs = 0
        true_positive = 0
        false_negative = 0
        for (pos_words_in_doc, neg_words_in_doc) in pos_counts:
            if pos_words_in_doc > neg_words_in_doc:
                true_positive += 1
            else:
                false_negative += 1
            number_of_docs += 1
        # print "\n-----Positive texts results"
        # print "(" + str(number_of_right_classified_docs) + "/" + str(number_of_docs) + ") = "  + str(number_of_right_classified_docs/number_of_docs)

        number_of_docs = 0
        true_negative = 0
        false_positive = 0
        for (pos_words_in_doc, neg_words_in_doc) in neg_counts:
            if pos_words_in_doc < neg_words_in_doc:
                true_negative += 1
            else:
                false_positive += 1
            number_of_docs += 1
        print "\n\n--------------Results--------------"
        precision = true_positive/(true_positive + false_positive)
        print "Precision: " + str(precision)
        recall = true_positive/(true_positive + false_negative)
        print "Recall: " + str(recall)
        print "F-measure: " + str(2/(1/precision + 1/recall))
        print "Accuracy: " + str((true_negative + true_positive) / (true_negative + true_positive + false_negative + false_positive))
        # print "(" + str(number_of_right_classified_docs) + "/" + str(number_of_docs) + ") = "  + str(number_of_right_classified_docs/number_of_docs)
        print "---------------------------------------\n\n"
        # print "\n\n\n\n---Pos counts:"
        # print pos_counts
        # print "\n\n\n\n---Neg counts:"
        # print neg_counts
        # csv_file_name = "configuration_" + str(i) + "_" + "good_words_" + str(good_words_count)
        # write_array_of_tuples_in_file(pos_counts, csv_file_name + "_in_positive_texts.csv")
        # write_array_of_tuples_in_file(neg_counts, csv_file_name + "_in_negative_texts.csv")
# print words_docs_pos_count_dict.most_common(10)
# print words_docs_neg_count_dict.most_common(10)

# words_docs_count_array = []
# words_pos_docs_count_array = []
# words_neg_docs_count_array = []
#
# for key in words_docs_count_dict:
#     words_docs_count_array.append((key, words_docs_count_dict[key]))
# for key in words_docs_neg_count_dict:
#     words_neg_docs_count_array.append((key, words_docs_neg_count_dict[key]))
# for key in words_docs_pos_count_dict:
#     words_pos_docs_count_array.append((key, words_docs_pos_count_dict[key]))
#
# words_docs_count_array.sort(key=lambda tup: tup[1], reverse=True)
# words_pos_docs_count_array.sort(key=lambda tup: tup[1], reverse=True)
# words_neg_docs_count_array.sort(key=lambda tup: tup[1], reverse=True)


good_words_array = []
for key in words_docs_neg_count_dict:
    if key not in words_docs_pos_count_dict:
        # good_words_array.append((key, 0, words_docs_neg_count_dict[key]))
        continue
    if abs(words_docs_neg_count_dict[key] - words_docs_pos_count_dict[key]) > N_DIFF:
        good_words_array.append((key, words_docs_pos_count_dict[key], words_docs_neg_count_dict[key]))
# for key in words_docs_pos_count_dict:
#     if key not in words_docs_neg_count_dict:
#         good_words_array.append((key, words_docs_pos_count_dict[key], 0))
#         continue
good_words_array.sort(key=lambda tup: tup[1], reverse=True)
good_words_array = good_words_array[:200]


change_docs_dict()

pos_good_words_dict = {}
neg_good_words_dict = {}
pos_good_words_count = 0
neg_good_words_count = 0
for i in range(len(good_words_array)):
    tuple = good_words_array[i]
    if tuple[1] > tuple[2]:
        if pos_good_words_count < 50:
            pos_good_words_dict[tuple[0]] = (tuple[1], tuple[2])
            pos_good_words_count += 1
    elif neg_good_words_count < 50:
        neg_good_words_dict[tuple[0]] = (tuple[1], tuple[2])
        neg_good_words_count += 1
    if pos_good_words_count >= 50 and neg_good_words_count >= 50:
        break


pos_counts = []
neg_counts = []
for key in docs:
    number_of_good_pos_words_in_doc = 0
    number_of_good_neg_words_in_doc = 0
    doc_dict = docs[key]
    for word in pos_good_words_dict:
        if word in doc_dict:
            number_of_good_pos_words_in_doc += 1
    for word in neg_good_words_dict:
        if word in doc_dict:
            number_of_good_neg_words_in_doc += 1
    if pos_neg_docs[key] == 1:
        pos_counts.append((number_of_good_pos_words_in_doc, number_of_good_neg_words_in_doc))
    else:
        neg_counts.append((number_of_good_pos_words_in_doc, number_of_good_neg_words_in_doc))

write_array_of_tuples_in_file(pos_counts, 'count_of_good_words_in_positive_texts.csv')
write_array_of_tuples_in_file(neg_counts, 'count_of_good_words_in_negative_texts.csv')

