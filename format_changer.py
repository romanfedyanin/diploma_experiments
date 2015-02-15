__author__ = 'romanfedyanin'
import os
import json
import pandas as pd

def change_format_reviews():
    settings_array = [
        {
            "path": "/neg",
            "negpos": -1,
            "filename": "neg.json"
        },
        {
            "path": "/pos",
            "negpos": 1,
            "filename": "pos.json"
        }
    ]
    for i in range(len(settings_array)):
        setting_dict = settings_array[i]
        dir_name = os.path.dirname(os.path.abspath(__file__)) + setting_dict["path"]
        print "start loading " + str(setting_dict["path"])
        file_names = os.listdir(dir_name)
        json_data = {}
        for j in range(len(file_names)):
            file_name = file_names[j]
            if file_name.endswith(".txt"):
                with open(dir_name + "/" + file_name, "r") as myfile:
                    data = myfile.read().replace('\n', '')
                    json_data[data] = setting_dict["negpos"]
        with open(setting_dict["filename"], 'w') as outfile:
            json.dump(json_data, outfile)


def read_data_from_fson(filename):
    json_data = open(filename)
    data = json.load(json_data)
    print len(data)
    print 1


def change_format_tweets():
    result_dict_pos = {}
    result_dict_neg = {}

    df = pd.read_csv('training.csv', sep=',', header=None)
    csv_strings = df.values

    max_count = 10020

    current_pos_string_number = 0
    current_neg_string_number = 0
    for i in range(len(csv_strings)):
        csv_string = csv_strings[i]
        if len(csv_string[5]) < 10:
            continue
        if csv_string[0] == 0:
            if current_neg_string_number < max_count:
                result_dict_neg[csv_string[5]] = -1
                current_neg_string_number += 1
        else:
            if current_pos_string_number < max_count:
                result_dict_pos[csv_string[5]] = 1
                current_pos_string_number += 1
        if current_neg_string_number >= max_count and current_pos_string_number >= max_count:
            break
    with open("neg_tweets.json", 'w') as outfile:
            json.dump(result_dict_neg, outfile)
    with open("pos_tweets.json", 'w') as outfile:
            json.dump(result_dict_pos, outfile)

read_data_from_fson("pos_tweets.json")
read_data_from_fson("neg_tweets.json")
# change_format_tweets()