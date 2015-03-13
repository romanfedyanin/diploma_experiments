 #!/usr/bin/python
 # -*- coding: utf-8 -*-
__author__ = 'romanfedyanin'
from datetime import datetime


# не успел красиво сделать enum((
# в 3ем питоне, вроде как, все проще
class Corpus:
    english_reviews = 1
    english_tweets = 2
    russian_reviews = 3

    corpus_type = 1

    def __init__(self, corpus):
        self.corpus_type = corpus

    #разделение по интервалам, чтобы разделять данные для обучения и данные для тестирования
    #лучше, конечно, определять автоматически для произвольного количества интервалов, но быстрее было сделать так
    def intervals_array(self):
        if self.corpus_type == Corpus.english_reviews:
            first_interval = (1, 333)
            second_interval = (334, 666)
            third_interval = (667, 999)
        elif self.corpus_type == Corpus.english_tweets:
            first_interval = (1, 3000)
            second_interval = (3001, 6000)
            third_interval = (6001, 9000)
        elif self.corpus_type == Corpus.russian_reviews:
            first_interval = (1, 1500)
            second_interval = (1501, 3000)
            third_interval = (3001, 4500)
        intervals_array = [
            [[second_interval, third_interval], [first_interval]],
            [[first_interval, third_interval], [second_interval]],
            [[first_interval, second_interval], [third_interval]]
        ]
        return intervals_array

    def file_names(self):
        if self.corpus_type == Corpus.english_reviews:
            file_names = [
                "neg_reviews.json",
                "pos_reviews.json"
            ]
        elif self.corpus_type == Corpus.english_tweets:
            file_names = [
                "neg_tweets.json",
                "pos_tweets.json"
            ]
        elif self.corpus_type == Corpus.russian_reviews:
            file_names = [
                "neg_russian_reviews.json",
                "pos_russian_reviews.json"
            ]
        return file_names


def number_inside_intervals(number, intervals):
    for interval in intervals:
        if interval[0] <= number <= interval[1]:
            return True
    return False


def print_time():
    print datetime.now().strftime('%H:%M:%S')