#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import keras
from keras.callbacks import EarlyStopping, CSVLogger, ModelCheckpoint, Callback
from keras.layers import Embedding
from sklearn import metrics

import yoochose_catalog
import system_utils
from item2vec import Item2vec
from session_reader import SessionReader
from sklearn.utils import class_weight

# from sklearn.metrics import roc_curve
# import matplotlib.pyplot as plt

random.seed(0)

import numpy as np
from keras.layers.core import Dense, Masking
from keras.layers.recurrent import LSTM
from keras.models import Sequential, load_model
from keras.preprocessing.sequence import pad_sequences


class printTest(Callback):
    def on_epoch_end(self, epoch, logs=None):
        y_pred = self.model.predict(x=x_test)
        with open("model_predict_%s.csv" % epoch, "w") as fw:
            fw.write("x_test,y_test,y_pred\n")
            for i in range(y_pred.shape[0]):
                y_pred[i] = y_pred[i][0]
                fw.write("%s,%s,%s\n" % (str(x_test[i]).replace("\n", " ").replace(",", ";"),
                                         str(y_test[i]), str(float(y_pred[i]))))
        auc = metrics.roc_auc_score(y_test, y_pred)
        print(" auc test = %s" % str(auc))


def create_y(seq_session_train):
    end_with_purchase_train = np.zeros(seq_session_train.shape[0])
    count = 0
    for end_with in seq_session_train.buy.values:
        if end_with > 0:
            end_with_purchase_train[count] = 1
        count += 1
    return end_with_purchase_train


def encode_des_x(seq_session_train, item2vec, batch_size=8 * 1024):
    batch_size = min(batch_size, seq_session_train.shape[0])
    X = None
    count = 0
    temp = []
    for session in seq_session_train:
        x = []
        for item in session:
            x += [item2vec.item_to_emb(int(item))]
        temp += [x]
        if count >= batch_size:
            count = 0
            current_np_array = pad_sequences(np.asarray(temp), maxlen=max_len_session, dtype='float32')
            if X is None:
                X = current_np_array
            else:
                X = np.append(X, current_np_array, 0)
            temp = []
        count += 1
    if count > 0:
        current_np_array = pad_sequences(np.asarray(temp), maxlen=max_len_session, dtype='float32')
        if X is None:
            X = current_np_array
        else:
            X = np.append(X, current_np_array, 0)
    # X = pad_sequences(X, maxlen=max_len_session, dtype='float32')
    return X


def encode_item_x(seq_session_train, word2ind, ind2word, batch_size=8 * 1024):
    batch_size = min(batch_size, seq_session_train.shape[0])
    X = None
    count = 0
    temp = []
    for session in seq_session_train:
        x = []
        for item in session:
            x += [word2ind[int(item)]]
        temp += [x]
        if count >= batch_size:
            count = 0
            current_np_array = pad_sequences(np.asarray(temp), maxlen=max_len_session, dtype='float32')
            if X is None:
                X = current_np_array
            else:
                X = np.append(X, current_np_array, 0)
            temp = []
        count += 1
    if count > 0:
        current_np_array = pad_sequences(np.asarray(temp), maxlen=max_len_session, dtype='float32')
        if X is None:
            X = current_np_array
        else:
            X = np.append(X, current_np_array, 0)
    # X = pad_sequences(X, maxlen=max_len_session, dtype='float32')
    return X


def create_test():
    print('create testing')
    if not system_utils.is_file_exist("x_test.npy") or not system_utils.is_file_exist("y_test.npy"):
        print('create set set..')
        seq_session_test = s.get_test()
        if use_item_emb:
            x_test = encode_des_x(seq_session_test.actions.values, item2vec)
        else:
            x_test = encode_item_x(seq_session_test.actions.values, word2ind, ind2word, batch_size=4 * 1024)
        np.save("x_test.npy", x_test)
        y_test = create_y(seq_session_test)
        np.save("y_test.npy", y_test)
        del seq_session_test
    else:
        print('load x_test')
        x_test = np.load("x_test.npy")
        print('load y_test')
        y_test = np.load("y_test.npy")
    return x_test, y_test


def run_exp():
    if use_item_emb:
        if system_utils.is_file_exist("models.h5"):
            print
            "loading models.."
            model = load_model("models.h5")
            print
            "loaded.."
        else:
            print("train models")
            # max_features = len(items) + 1
            model = Sequential()
            # models.add(Embedding(max_features, embedding_size, input_length=max_len_session, mask_zero=True))
            model.add(Masking(mask_value=0.0, input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(LSTM(hidden_size_gru))
            model.add(Dense(dense_layer_size))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=0.00001, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                          decay=0.0)
                          , metrics=['accuracy'])
            print("models summary:")
            print(model.summary())
            print('training size = %s' % str(x_train.shape[0]))
            print('training..')
            if use_sample_weight:
                model.fit(x=x_train, y=y_train, epochs=epochs_model, sample_weight=sample_weight,
                          callbacks=my_callbacks)
            else:
                if use_class_weight:
                    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
                    print(class_weights)
                    print("class_weight is %s" % str(class_weights))
                    model.fit(x=x_train, y=y_train, epochs=epochs_model, class_weight=dict(enumerate(class_weights)),
                              callbacks=my_callbacks)
                else:
                    model.fit(x=x_train, y=y_train, epochs=epochs_model, callbacks=my_callbacks)
            try:
                print('saving models..')
                model.save("models.h5")
                print('models saved')
            except:
                print
                'cant save the models'
        print('testing..')
        y_pred = model.predict(x=x_test)
        with open("model_predict.csv", "w") as fw:
            fw.write("x_test,y_test,y_pred\n")
            for i in range(y_pred.shape[0]):
                y_pred[i] = y_pred[i][0]
                fw.write("%s,%s,%s\n" % (str(x_test[i]).replace("\n", " ").replace(",", ";"),
                                         str(y_test[i]), str(float(y_pred[i]))))
                # fw.write("%s\n" % (str(y_test[i]), str(float(y_pred[i]))))
        auc = metrics.roc_auc_score(y_test, y_pred)
        print
        auc
        if not debug:
            system_utils.send_email(body='the auc is %s' % str(auc))
    else:
        if system_utils.is_file_exist("models.h5"):
            print
            "loading models.."
            model = load_model("models.h5")
            print
            "loaded.."
        else:
            max_features = len(items) + 1
            model = Sequential()
            model.add(Embedding(max_features, embedding_size, input_length=max_len_session, mask_zero=True))
            # models.add(Masking(mask_value=0.0, input_shape=(x_train.shape[1], x_train.shape[2])))
            model.add(LSTM(hidden_size_gru))
            model.add(Dense(dense_layer_size))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
            print("models summary:")
            print(model.summary())
            print('training size = %s' % str(x_train.shape[0]))
            print('testing size = %s' % str(x_test.shape[0]))
            print('training..')
            if use_sample_weight:
                model.fit(x=x_train, y=y_train, epochs=epochs_model, sample_weight=sample_weight)
            else:
                if use_class_weight:
                    class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
                    print(class_weights)
                    model.fit(x=x_train, y=y_train, epochs=epochs_model, class_weight=dict(enumerate(class_weights)),
                              callbacks=my_callbacks)
                else:
                    model.fit(x=x_train, y=y_train, epochs=epochs_model, callbacks=my_callbacks)
            try:
                print('saving models...')
                model.save("models.h5")
                print('models saved')
            except:
                print
                'cant save the models'
        print('testing..')
        y_pred = model.predict(x=x_test)
        with open("model_predict.csv", "w") as fw:
            fw.write("x_test,y_test,y_pred\n")
            for i in range(y_pred.shape[0]):
                y_pred[i] = y_pred[i][0]
                fw.write("%s,%s,%s\n" % (str(x_test[i]).replace("\n", " ").replace(",", ";"),
                                         str(y_test[i]), str(float(y_pred[i]))))
        auc = metrics.roc_auc_score(y_test, y_pred)
        print
        auc
        if not debug:
            system_utils.send_email(body='the auc is %s' % str(auc))

            # fpr_grd2, tpr_grd2, _ = roc_curve(y_test, y_pred)
            # plt.figure(1)
            # plt.plot([0, 1], [0, 1], 'k--')
            # # plt.plot(fpr_grd1, tpr_grd1, label='hmm')
            # plt.plot(fpr_grd2, tpr_grd2, label='deep')
            # plt.xlabel('False positive rate')
            # plt.ylabel('True positive rate')
            # plt.title('ROC curve')
            # plt.legend(loc='best')
            # plt.show()


if __name__ == "__main__":
    # check_data()
    # debug = True
    debug = False
    use_item_emb = True
    use_class_weight = True
    # use_item_emb = False
    # min_len_session = None
    min_len_session = 2
    max_len_session = 10
    hidden_size_gru = 150
    embedding_size = 50
    dense_layer_size = 200
    use_sample_weight = False
    item2vec_epoch = 30
    epochs_model = 20

    if debug:
        hidden_size_gru = 30
        item2vec_epoch = 50
        epochs_model = 10
        item2vec_epoch = 1

    my_callbacks = [
        printTest(),
        CSVLogger('epochs_info.csv'),
        ModelCheckpoint("best_model.h5", monitor='loss', save_best_only=True, verbose=1,
                        )]
    # EarlyStopping(monitor='loss', patience=2, verbose=1)
    print('starting...')
    print('create catalog...')
    c = yoochose_catalog.Catalog(
        dir_path="catalog")
    dir = 'data'
    items = c.get_items()
    print("prepare session file")
    dates_for_test = ['2016-08-31']
    print('create item2vec...')
    if use_item_emb:
        item2vec = Item2vec(catalog=c, embedding_size=embedding_size, hidden_size=10, max_len=10,
                            epoches=item2vec_epoch)
    else:
        word2ind = {word: (index + 1) for index, word in enumerate(items)}
        ind2word = {(index + 1): word for index, word in enumerate(items)}
    print("create sets..")
    if system_utils.is_file_exist("x_train.npy") and system_utils.is_file_exist(
            "y_train.npy") and system_utils.is_file_exist("x_test.npy") and system_utils.is_file_exist("y_test.npy"):
        x_train = np.load("x_train.npy")
        y_train = np.load("y_train.npy")
        x_test = np.load("x_test.npy")
        y_test = np.load("y_test.npy")
    else:
        s = SessionReader(input_path_session_actions='%s/eventsquance.txt' % dir,
                          input_path_session_info='%s/list session.csv' % dir, items_list=items,
                          test_dates=dates_for_test
                          , maxlen=max_len_session
                          , minlen=min_len_session)
        seq_session_train = None
        if system_utils.is_file_exist("models.h5"):
            model = load_model("models.h5")
        else:
            if not system_utils.is_file_exist("x_train.npy") or not system_utils.is_file_exist(
                    "y_train.npy"):
                seq_session_train = s.get_train()
                print('create training')
                if system_utils.is_file_exist("x_train.npy"):
                    print('load x_train')
                    x_train = np.load("x_train.npy")
                    print("saved x_train....")
                else:
                    if use_item_emb:
                        x_train = encode_des_x(seq_session_train.actions.values, item2vec)
                    else:
                        x_train = encode_item_x(seq_session_train.actions.values, word2ind, ind2word)
                    print("save x_train....")
                    np.save("x_train.npy", x_train)

                if not system_utils.is_file_exist("y_train.npy"):
                    y_train = create_y(seq_session_train)
                    np.save("y_train.npy", y_train)
                else:
                    print('load y_train')
                    y_train = np.load("y_train.npy")
            else:
                print('load training')
                x_train = np.load("x_train.npy")
                y_train = np.load("y_train.npy")
            del seq_session_train

        # test
        x_test, y_test = create_test()
        del c
    if use_sample_weight:
        print("create sample weights..")
        buy_0 = 0
        buy_1 = 0
        for i in y_train:
            if i == 0:
                buy_0 += 1
            else:
                buy_1 += 1
        sample_weight = np.ones(y_train.shape[0])
        weight = int((buy_0 + 0.0) / (buy_1 + 0.0))
        print("weight = %s" % str(weight))
        for i in range(y_train.shape[0]):
            if not y_train[i] == 0:
                sample_weight[i] = weight
        print("done sample weights..")
    print('done preproccesing')
    run_exp()
