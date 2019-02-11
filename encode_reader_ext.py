import numpy as np
import pandas as pd
from keras.preprocessing.sequence import pad_sequences

import system_utils

"""
This class read the session data and create Training and Testing Sets.
"""


class EncodeReaderExt(object):
    def __init__(self, train_df, test_df1, test_df2, catalog, item2vec,
                 encode_path='data_after_encode',
                 max_session_size=10, encode_mode=2, load_encode_model=True):
        # struct for help
        self.items_in_train = -1
        self.catalog = catalog
        self.item2vec = item2vec
        self.train_df = train_df
        self.test_df1 = test_df1
        self.test_df2 = test_df2
        self.word2ind = None
        self.ind2word = None

        self.x_train_path = "%s/x_train.npy" % encode_path
        self.y_train_path = "%s/y_train.npy" % encode_path
        self.x_test_path1 = "%s/x_test1.npy" % encode_path
        self.x_test_path2 = "%s/x_test2.npy" % encode_path
        self.y_test_path1 = "%s/y_test1.npy" % encode_path
        self.y_test_path2 = "%s/y_test2.npy" % encode_path

        # resize the data
        self.max_session_size = max_session_size

        # more parameters
        self.encode_mode = encode_mode

        # try to load
        if load_encode_model and system_utils.is_file_exist(self.x_train_path) and system_utils.is_file_exist(
                self.y_train_path) and system_utils.is_file_exist(self.x_test_path1) and system_utils.is_file_exist(
            self.y_test_path1) and system_utils.is_file_exist(self.x_test_path2) and system_utils.is_file_exist(
            self.y_test_path2):
            self.x_train = np.load(self.x_train_path)
            self.y_train = np.load(self.y_train_path)
            self.x_test1 = np.load(self.x_test_path1)
            self.y_test1 = np.load(self.y_test_path1)
            self.x_test2 = np.load(self.x_test_path2)
            self.y_test2 = np.load(self.y_test_path2)

        else:
            self.create_train_set()
            if self.check_empty_set(self.test_df1):
                self.x_test1 = None
                self.y_test1 = None
            else:
                self.create_test_set1()
            if self.check_empty_set(self.test_df2):
                self.x_test2 = None
                self.y_test2 = None
            else:
                self.create_test_set2()
        # test

    def check_empty_set(self, set):
        return set is None or len(set.shape) == 0 or set.shape[0] == 0

    def create_train_set(self):
        self.seq_session_train = None
        if not system_utils.is_file_exist(self.x_train_path) or not system_utils.is_file_exist(self.y_train_path):
            seq_session_train = self.train_df
            if system_utils.is_file_exist(self.x_train_path):
                self.x_train = np.load(self.x_train_path)
            else:
                if self.encode_mode == 1:  # embedding item id
                    # items = self.catalog.get_items()
                    items = set()
                    for session in seq_session_train.actions.values:
                        for item in session:
                            items.add(int(item))
                    self.word2ind = {word: (index + 1) for index, word in enumerate(items)}
                    self.ind2word = {(index + 1): word for index, word in enumerate(items)}
                    self.items_in_train = items
                    self.x_train = self.encode_item_x(seq_session_train.actions.values, self.word2ind, self.ind2word
                                                      )

                elif self.encode_mode == 2:  # use embedding dict what made before
                    self.x_train = self.encode_des_x(seq_session_train.actions.values, self.item2vec
                                                     )
                else:  # no embedding (for test only)
                    self.x_train = seq_session_train.actions.values
                np.save(self.x_train_path, self.x_train)

                if not system_utils.is_file_exist(self.y_train_path):
                    self.y_train = self.create_y(seq_session_train)
                    np.save(self.y_train_path, self.y_train)
                else:
                    self.y_train = np.load(self.y_train_path)
        else:
            self.x_train = np.load(self.x_train_path)
            self.y_train = np.load(self.y_train_path)

    def check_exist(self, path):
        return system_utils.is_file_exist(path)

    def encode_des_x(self, seq_session_train, item2vec, batch_size=8 * 1024):
        batch_size = min(batch_size, seq_session_train.shape[0])
        X = None
        count = 0
        temp = []
        for session in seq_session_train:
            x = []
            for item in session:
                x += [item2vec[int(item)]]
            temp += [x]
            if count >= batch_size:
                count = 0
                current_np_array = pad_sequences(np.asarray(temp), maxlen=self.max_session_size, dtype='float32')
                if X is None:
                    X = current_np_array
                else:
                    X = np.append(X, current_np_array, 0)
                temp = []
            count += 1
        if count > 0:
            current_np_array = pad_sequences(np.asarray(temp), maxlen=self.max_session_size, dtype='float32')
            if X is None:
                X = current_np_array
            else:
                X = np.append(X, current_np_array, 0)
        # X = pad_sequences(X, maxlen=max_len_session, dtype='float32')
        return X

    def encode_item_x(self, seq_session_train, word2ind, ind2word=None, batch_size=8 * 1024):
        batch_size = min(batch_size, seq_session_train.shape[0])
        X = None
        count = 0
        temp = []
        for session in seq_session_train:
            x = []
            for item in session:
                x += [word2ind.get(int(item), 0)]
            temp += [x]
            if count >= batch_size:
                count = 0
                current_np_array = pad_sequences(np.asarray(temp), maxlen=self.max_session_size, dtype='float32')
                if X is None:
                    X = current_np_array
                else:
                    X = np.append(X, current_np_array, 0)
                temp = []
            count += 1
        if count > 0:
            current_np_array = pad_sequences(np.asarray(temp), maxlen=self.max_session_size, dtype='float32')
            if X is None:
                X = current_np_array
            else:
                X = np.append(X, current_np_array, 0)
        # X = pad_sequences(X, maxlen=max_len_session, dtype='float32')
        return X

    # def create_test_set1(self):
    #     if not system_utils.is_file_exist(self.x_test_path1) or not system_utils.is_file_exist(
    #             self.y_test_path1) or not system_utils.is_file_exist(
    #         self.x_test_path2) or not system_utils.is_file_exist(self.y_test_path2):
    #         seq_session_test = self.test_df1
    #         if self.encode_mode == 1:  #
    #             self.x_test = self.encode_item_x(seq_session_test.actions.values, self.word2ind, self.ind2word,
    #                                              batch_size=4 * 1024)
    #         elif self.encode_mode == 2:
    #             self.x_test = self.encode_des_x(seq_session_test.actions.values, self.item2vec)
    #
    #         else:
    #             self.x_test = seq_session_test.actions.values
    #
    #         np.save(self.x_test_path1, self.x_test1)
    #         self.y_test1 = self.create_y(seq_session_test)
    #         np.save(self.y_test_path1, self.y_test1)
    #     else:
    #         self.x_test = np.load(self.x_test_path1)
    #         self.y_test = np.load(self.y_test_path1)

    def create_test_set1(self):
        if not system_utils.is_file_exist(self.x_test_path1) or not system_utils.is_file_exist(
                self.y_test_path1) or not system_utils.is_file_exist(
            self.x_test_path2) or not system_utils.is_file_exist(self.y_test_path2):
            seq_session_test = self.test_df1
            if self.encode_mode == 1:  #
                self.x_test1 = self.encode_item_x(seq_session_test.actions.values, self.word2ind, self.ind2word,
                                                  batch_size=4 * 1024)
            elif self.encode_mode == 2:
                self.x_test1 = self.encode_des_x(seq_session_test.actions.values, self.item2vec)

            else:
                self.x_test1 = seq_session_test.actions.values

            np.save(self.x_test_path1, self.x_test1)
            self.y_test1 = self.create_y(seq_session_test)
            np.save(self.y_test_path1, self.y_test1)
        else:
            self.x_test1 = np.load(self.x_test_path1)
            self.y_test1 = np.load(self.y_test_path1)

    def create_test_set2(self):
        if not system_utils.is_file_exist(self.x_test_path2) or not system_utils.is_file_exist(
                self.y_test_path2):
            seq_session_test = self.test_df2
            if self.encode_mode == 1:  #
                self.x_test2 = self.encode_item_x(seq_session_test.actions.values, self.word2ind, self.ind2word,
                                                  batch_size=4 * 1024)
            elif self.encode_mode == 2:
                self.x_test2 = self.encode_des_x(seq_session_test.actions.values, self.item2vec)

            else:
                self.x_test2 = seq_session_test.actions.values

            np.save(self.x_test_path2, self.x_test2)
            self.y_test2 = self.create_y(seq_session_test)
            np.save(self.y_test_path2, self.y_test2)
        else:
            self.x_test2 = np.load(self.x_test_path2)
            self.y_test2 = np.load(self.y_test_path2)

    def create_y(self, seq_session_train):
        end_with_purchase_train = np.zeros(seq_session_train.shape[0])
        count = 0
        for end_with in seq_session_train.buy.values:
            if end_with > 0:
                end_with_purchase_train[count] = 1
            count += 1
        return end_with_purchase_train

    def get_x_train(self):
        return self.x_train

    def get_y_train(self):
        return self.y_train

    def get_x_test1(self):
        return self.x_test1

    def get_x_test2(self):
        return self.x_test2

    def get_y_test1(self):
        return self.y_test1

    def get_y_test2(self):
        return self.y_test2
