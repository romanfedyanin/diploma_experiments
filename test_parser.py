from __future__ import division
import os
import nltk
from nltk import FreqDist

number_of_documents = 0
docs = {}
words_docs_count_dict = {}


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


def add_text_to_corpus(text, negpos):
    global number_of_documents
    number_of_documents += 1
    text = text.lower()
    docs[text] = negpos
    words = nltk.tokenize.word_tokenize(text)
    freq_dist = FreqDist(words)
    for k in freq_dist:
        words_docs_count_dict[k] = words_docs_count_dict.get(k, 0.0) + 1.0


def number_of_words_in_text(text):
    return len(nltk.tokenize.word_tokenize(text))


def normalize_freq_dist(freq_dist, words):
    length = len(words)
    for k in freq_dist:
        freq_dist[k] /= length


def make_text_with_tfidf(text):
    words = nltk.tokenize.word_tokenize(text)
    freq_dict = FreqDist(words)
    normalize_freq_dist(freq_dict, words)

    for word in freq_dict:
        print freq_dict[word]
        print words_docs_count_dict[word]
        text = text.replace(word + " ", word + "(%f) " % (number_of_documents * freq_dict[word] / words_docs_count_dict[word]))
    return text


load_data()
text = "one thing to bear in mind is that there are many ways to text "


print "Text before: "
print text
text = make_text_with_tfidf(text)
print "Text after: "
print text

