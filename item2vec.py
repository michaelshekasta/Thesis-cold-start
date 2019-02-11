#!/usr/bin/env python
# -*- coding: utf-8 -*-

import numpy as np
from keras.callbacks import EarlyStopping
from keras.layers import Embedding, LSTM, Dense, Activation, SimpleRNN, GRU
from keras.models import Sequential
from keras.preprocessing.sequence import pad_sequences
from keras.utils.np_utils import to_categorical

np.random.seed(0)


class Item2vec(object):
    def __init__(self, embedding_size, catalog, max_len, hidden_size=150, epoches=10):
        self.epochs = epoches
        self.catalog = catalog
        self.embedding_size = embedding_size
        self.item2emb = None
        self.max_len = max_len
        self.lstm_hidden_size = hidden_size
        n_words = self.catalog.n_words
        sentences_idx = []
        categories_idx = []
        for item_id in self.catalog.item2idx.keys():
            vector = self.catalog.item2idx[item_id]
            sentences_idx.append(vector)
            idx_cat = self.catalog.cat2idx[self.catalog.item2cat[item_id]]
            # hot_vector_y = to_categorical(idx_cat, num_classes=self.catalog.n_cat).tolist()[0]
            hot_vector_y = to_categorical([idx_cat], num_classes=self.catalog.n_cat).tolist()[0]
            #            hot_vector_y = [0] * self.catalog.n_catz
            #            hot_vector_y[idx_cat] = 1
            categories_idx.append(hot_vector_y)
        sentences_array = np.asarray(pad_sequences(sentences_idx, maxlen=max_len, truncating='post', padding='post'),
                                     dtype='int32')
        categories = np.asarray(categories_idx, dtype='int32')

        my_callbacks = [
            EarlyStopping(monitor='loss')
        ]
        model = Sequential()
        model.add(Embedding(n_words, embedding_size, input_length=max_len, mask_zero=True))
        # models.add(SimpleRNN(self.catalog.n_cat))

        model.add(GRU(self.lstm_hidden_size))
        # models.add(Dense(embedding_size*4))
        # models.add(Dense(embedding_size*2))
        model.add(Dense(embedding_size))
        model.add(Dense(self.catalog.n_cat))
        model.add(Activation("softmax"))

        # models.compile(optimizer='adam', loss='categorical_crossentropy')
        # sparse_categorical_crossentropy
        # models.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
        model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy', 'categorical_accuracy'])
        model.summary()
        # fit the models to predict what color each person is
        model.fit(sentences_array, categories, epochs=self.epochs, verbose=2, callbacks=my_callbacks)
        # self.item2vec = models.layers[3].W.get_value()
        model.pop()
        model.pop()
        self.item2emb = dict()
        # for item_id, vector in self.catalog.item2idx.iteritems():
        # for item_id, vector in self.catalog.item2idx.items():
        for item_id in self.catalog.item2idx.keys():
            vector = self.catalog.item2idx[item_id]
            self.item2emb[item_id] = model.predict(np.asarray(pad_sequences([vector], maxlen=max_len), dtype='int32'))[
                0]

    def item_to_emb(self, item_id):
        return self.item2emb[item_id]
