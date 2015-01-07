from __future__ import division
import os
import nltk
from nltk import FreqDist
from nltk.stem.porter import *

stemmer = PorterStemmer()
number_of_documents = 0
number_of_pos_documents = 0
number_of_neg_documents = 0
docs = {}
words_docs_count_dict = {}
words_docs_pos_count_dict = {}
words_docs_neg_count_dict = {}

LEMMATIZE = True


def load_data():
    dir_name = os.path.dirname(os.path.abspath(__file__)) + "/neg"
    for file in os.listdir(dir_name):
        if file.endswith(".txt"):
            with open(dir_name + "/" + file, "r") as myfile:
                data = myfile.read().replace('\n', '')
                add_text_to_corpus(data, -1)
    dir_name = os.path.dirname(os.path.abspath(__file__)) + "/pos"
    for file in os.listdir(dir_name):
        if file.endswith(".txt"):
            with open(dir_name + "/" + file, "r") as myfile:
                data = myfile.read().replace('\n', '')
                add_text_to_corpus(data, 1)


def change_docs_dict():
    global docs
    new_docs = {}
    for key in docs:
        # if docs[key] == 1:
        #     continue
        words = nltk.tokenize.word_tokenize(key)
        lemmatize_words(words)
        freq_dict = FreqDist(words)
        normalize_freq_dist(freq_dict, words)
        dict_of_words_in_text = []
        for word in freq_dict:
            tfidf_all = number_of_documents * freq_dict[word] / words_docs_count_dict[word]
            dict_of_words_in_text.append((word, tfidf_all))
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
    words = nltk.tokenize.word_tokenize(text)
    lemmatize_words(words)
    freq_dist = FreqDist(words)
    for k in freq_dist:
        words_docs_count_dict[k] = words_docs_count_dict.get(k, 0.0) + 1.0
        if negpos == 1:
            words_docs_pos_count_dict[k] = words_docs_pos_count_dict.get(k, 0.0) + 1.0
        else:
            words_docs_neg_count_dict[k] = words_docs_neg_count_dict.get(k, 0.0) + 1.0


def number_of_words_in_text(text):
    return len(nltk.tokenize.word_tokenize(text))


def normalize_freq_dist(freq_dist, words):
    length = len(words)
    for k in freq_dist:
        freq_dist[k] /= length


def make_text_with_tfidf(text):
    words = nltk.tokenize.word_tokenize(text)
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

load_data()
# text = "one thing to bear in mind is that there are many ways to text "
#
#
# print "Text before: "
# print text
# text = make_text_with_tfidf(text)
# print "Text after: "

neg_texts_only_words_freqs = []
pos_texts_only_words_freqs = []
for key in words_docs_count_dict:
    if key not in words_docs_neg_count_dict:
        pos_texts_only_words_freqs.append((key, words_docs_count_dict[key]))
    if key not in words_docs_pos_count_dict:
        neg_texts_only_words_freqs.append((key, words_docs_count_dict[key]))

neg_texts_only_words_freqs.sort(key=lambda tup: tup[1], reverse=True)
neg_texts_only_words_freqs = neg_texts_only_words_freqs[:100]
print neg_texts_only_words_freqs

pos_texts_only_words_freqs.sort(key=lambda tup: tup[1], reverse=True)
pos_texts_only_words_freqs = pos_texts_only_words_freqs[:100]
print pos_texts_only_words_freqs

change_docs_dict()

f = open('filename.csv','wb')
for doc in docs:
    doc_array = docs[doc]
    doc_array.sort(key=lambda tup: tup[1], reverse=True)
    doc_array = doc_array[:10]
    string_to_file = ""
    for i in range(len(doc_array)-1):
        string_to_file += doc_array[i][0] + ";"
    string_to_file += doc_array[len(doc_array)-1][0] + "\n"
    f.write(string_to_file)
f.close()

