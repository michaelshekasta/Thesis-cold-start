#!/usr/bin/env python
# -*- coding: utf-8 -*-
import math
import re

import nltk
import pandas as pd
from nltk.corpus import stopwords

import system_utils

punctuation_tokens = ['.', '..', '...', ',', ';', ':', '(', ')', '"', '\'', '[', ']', '{', '}', '?', '!', '-', u'–',
                      '+', '*', '--', '\'\'', '``', '_']
punctuation = u'?.!/;:()&+,1234567890'
sentence_detector = nltk.data.load('tokenizers/punkt/german.pickle', format='pickle')
stop_words = stopwords.words('german')


def replace_umlauts(text):
    res = text
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    return res


def get_list_files(input_path):
    if input_path is None:
        return
    from os import listdir
    from os.path import isfile, join
    only_files = [f for f in listdir(input_path) if isfile(join(input_path, f))]
    return only_files


def read_file(file_path):
    return pd.read_csv(file_path, delimiter=';', encoding='utf8')


def remove_empty_description(df_file):
    bad_indexes = []
    for index, row in df_file.iterrows():
        try:
            des = row.description
            if math.isnan(des):
                bad_indexes.append(index)
        except:
            pass
    return df_file.drop(df_file.index[bad_indexes])


class Catalog(object):
    def __init__(self, dir_path, use_german_token=True, mode=1, to_replace_umlauts=False,
                 clean_words_punctuation=True):
        self.use_german_token = use_german_token
        self.mode = mode
        self.catalog_df = None
        self.item_id = "product_id"
        # self.description = "description"
        self.description_html = "Volltext-Beschreibung"
        self.description = "description"
        self.title = "title"
        self.category = "categorie"
        self.to_replace_umlauts = to_replace_umlauts
        self.clean_words_punctuation = clean_words_punctuation
        for file in get_list_files(dir_path):
            df_file = read_file("%s/%s" % (dir_path, file))
            df_file = remove_empty_description(df_file)
            if self.catalog_df is None:
                self.catalog_df = df_file
            else:
                self.catalog_df = pd.concat([self.catalog_df, df_file])
                self.catalog_df.drop_duplicates(subset=self.item_id, inplace=True)
        # remove nan
        self.catalog_df = self.catalog_df[self.catalog_df.product_id == self.catalog_df.product_id]
        self.catalog_df.sort_values(by=self.item_id, inplace=True)
        self.catalog_df.to_csv("catalog_new.csv", sep=';', encoding='utf8')
        self.word2idx = None
        self.items = None
        self.idx2word = None
        self.item2idx = None
        self.n_words = 0
        self.item2cat = None
        self.calc_dicts()
        print("number of words = %s" % str(len(self.idx2word)))
        print("number of items = %s" % str(len(self.item2idx)))

    def calc_dicts(self):
        self.calc_word2idx()
        self.calc_cat2ind()
        self.calc_item2idx()

    def get_col_name(self):
        return list(self.catalog_df)

    def get_df(self):
        return self.catalog_df.copy()

    def dump_catalog(self, dump_path):
        self.catalog_df.to_csv(dump_path, index=False, encoding='utf8')

    def restore_catalog(self, dump_path):
        self.catalog_df = pd.read_csv(dump_path, encoding='utf8')

    def calc_word2idx(self):
        self.items = self.catalog_df[self.item_id].values
        html_descriptions = self.catalog_df[self.description_html].values
        descriptions = self.catalog_df[self.description].values
        titles = self.catalog_df[self.title].values
        words = set()
        if self.mode == 1 or self.mode == 4 or self.mode == 5 or self.mode == 7:
            for html_description in html_descriptions:
                words = words.union(set(
                    self.description_to_word(description=html_description, use_german_token=self.use_german_token,
                                             remove_marks=True)))
        if self.mode == 2 or self.mode == 5 or self.mode == 6 or self.mode == 7:
            for description in descriptions:
                words = words.union(set(self.description_to_word(description, self.use_german_token)))
        if self.mode == 3 or self.mode == 4 or self.mode == 6 or self.mode == 7:
            for title in titles:
                words = words.union(set(self.description_to_word(title, self.use_german_token)))
        self.n_words = len(words)
        self.word2idx = dict((v, i) for i, v in enumerate(words))
        self.idx2word = list(words)
        self.words = words

    def calc_cat2ind(self):
        categories = self.catalog_df[self.category].unique()
        self.cat2idx = dict((v, i) for i, v in enumerate(categories))
        self.idx2cat = list(categories)
        self.n_cat = len(categories)

    def description_to_word(self, description, use_german_token=False, remove_marks=False):
        if type(description) == type(0.0) and math.isnan(description):
            return []
        if remove_marks:
            description_new = system_utils.stripHTML(description)
        else:
            description_new = description
        if self.to_replace_umlauts:
            sentence = replace_umlauts(description_new)
            sentence = sentence.lower()
        else:
            sentence = description_new
        if use_german_token:
            words = nltk.word_tokenize(text=sentence, language='german')
        else:
            words = nltk.word_tokenize(sentence)
        if self.clean_words_punctuation:
            words = [x for x in words if x not in punctuation_tokens]
            words = [re.sub('[' + punctuation + ']', '', x) for x in words]
            words = [x for x in words if x not in stop_words]
            # onlyletters = re.compile('[^a-zA-Z ]')
            words = [re.sub('[^a-zA-Z ]', '', x) for x in words]
        # words = re.compile('[^a-zA-Z ]').sub('',words)
        # words = nltk.word_tokenize(text=sentence, language='german')
        words = [x for x in words if len(x) > 0]
        return words

    # mode 1 - only html
    # mode 2 - only description
    # mode 3 - only title
    # mode 4 - 1+2
    # mode 5 - 1+3
    # mode 6 - 2+3
    # mode 7 - 1+2+3
    def description_to_index(self, html_description=None, description=None, title=None, mode=1):
        mode_1 = []
        mode_2 = []
        mode_3 = []
        if self.mode == 1 or self.mode == 4 or self.mode == 5 or self.mode == 7:
            mode_1 = [self.word2idx[i] for i in
                      self.description_to_word(description=html_description, use_german_token=self.use_german_token,
                                               remove_marks=True)]
        if self.mode == 2 or self.mode == 5 or self.mode == 6 or self.mode == 7:
            mode_2 = [self.word2idx[i] for i in self.description_to_word(description, self.use_german_token)]
        if self.mode == 3 or self.mode == 4 or self.mode == 6 or self.mode == 7:
            mode_3 = [self.word2idx[i] for i in self.description_to_word(title, self.use_german_token)]

        if mode == 2:
            return mode_2
        if mode == 3:
            return mode_3
        if mode == 4:
            return mode_2 + mode_1
        if mode == 5:
            return mode_3 + mode_1
        if mode == 6:
            return mode_3 + mode_2
        if mode == 7:
            return mode_3 + mode_2 + mode_1
        return mode_1

    def item_to_index(self, item_id):
        return self.item2idx[item_id]

    def calc_item2idx(self):
        self.item2idx = dict()
        self.item2cat = dict()
        for index, row in self.catalog_df.iterrows():
            self.item2idx[row[self.item_id]] = self.description_to_index(row[self.description_html],
                                                                         row[self.description], row[self.title],
                                                                         self.mode)
            self.item2cat[int(row[self.item_id])] = row[self.category]

    def get_items(self):
        return self.item2idx.keys()
