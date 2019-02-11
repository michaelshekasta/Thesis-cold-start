"""
This class will represent the experiment of regular calc
"""
import keras
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback, EarlyStopping
from keras.layers import Masking, LSTM, Dense, Embedding, Conv1D, GlobalMaxPooling1D
from keras.models import load_model, Sequential
from sklearn import metrics
from sklearn.utils import class_weight

import system_utils
import numpy as np

michael_test = 0

class printTest(Callback):
    def __init__(self, x_test1, y_test1, x_test2, y_test2, epoch=0):
        Callback.__init__(self)
        self.x_test1 = x_test1
        self.y_test1 = y_test1
        self.x_test2 = x_test2
        self.y_test2 = y_test2
        self.epoch = epoch

    def on_epoch_end(self, epoch, logs=None):
        self.calc_auc_epoch(epoch, 'regular', self.x_test1, self.y_test1)
        self.calc_auc_epoch(epoch, 'cold_start', self.x_test2, self.y_test2)
        self.calc_auc_epoch(epoch, 'integrated', np.append(self.x_test1, self.x_test2, 0),
                            np.append(self.y_test1, self.y_test2, 0))
        self.epoch += 1
        self.model.params['stopped_epoch'] = self.epoch

    def calc_auc_epoch(self, epoch, name, x_test, y_test):
        if x_test is None or len(x_test.shape) == 0:
            return
        y_pred = self.model.predict(x=x_test)
        with open("detailed_prediction/model_predict_%s_%s.csv" % (name, epoch), "w") as fw:
            fw.write("x_test,y_test,y_pred\n")
            for i in range(y_pred.shape[0]):
                y_pred[i] = y_pred[i][0]
                fw.write("%s,%s,%s\n" % (str(x_test[i]).replace("\n", " ").replace(",", ";"),
                                         str(y_test[i]), str(float(y_pred[i]))))
        if self.has_auc(y_test):
            auc = metrics.roc_auc_score(y_test, y_pred)
        else:
            auc = 'no auc'
        print("%s auc test = %s" % (name, str(auc)))

    def has_auc(self, y_test):
        count_buy_in_y_test = 0
        for i in y_test:
            count_buy_in_y_test += i
        has_auc_bool = count_buy_in_y_test > 0 and count_buy_in_y_test < y_test.shape[0]
        return has_auc_bool


class ColdStartExpVal(object):
    def run_exp(self, x_train, y_train, x_test, y_test, x_test_cold, y_test_cold, shuffle=True,
                x_val=None, y_val=None,
                use_cnn=False,
                validation_split=None):
        self.print_states("Run Training set ", y_train)
        self.print_states("Run Test set", y_test)
        if validation_split is None and x_val is None and y_val is None:
            callbacks = [
                printTest(x_test1=x_test, y_test1=y_test, x_test2=x_test_cold, y_test2=y_test_cold),
                CSVLogger('epochs_info.csv'),
                ModelCheckpoint("%s/best_model.h5" % self.model_path, monitor='loss', save_best_only=True,
                                verbose=1,
                                ),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
            ]
        else:
            callbacks = [
                printTest(x_test1=x_test, y_test1=y_test, x_test2=x_test_cold, y_test2=y_test_cold),
                CSVLogger('epochs_info.csv'),
                ModelCheckpoint("%s/best_model.h5" % self.model_path, monitor='val_loss',
                                save_best_only=True,
                                verbose=1, mode='min'
                                ),
                EarlyStopping(monitor='val_loss', min_delta=0, patience=0, verbose=0, mode='auto')
            ]
        if system_utils.is_file_exist("%s/models.h5" % self.model_path):
            print("loading models..")
            model = load_model("%s/models.h5" % self.model_path)
            print("loaded..")
        else:
            print("train models")
            # max_features = len(items) + 1
            model = Sequential()
            if self.encode_mode == 1:
                if use_cnn:
                    model.add(Embedding(self.max_features, self.embedding_size, input_length=self.max_len_session,

                                        ))
                else:
                    model.add(Embedding(self.max_features, self.embedding_size, input_length=self.max_len_session,
                                        mask_zero=True
                                        ))
            if self.encode_mode == 2:
                model.add(Masking(mask_value=0.0, input_shape=(x_train.shape[1], x_train.shape[2])))
            if use_cnn:
                model.add(Conv1D(self.hidden_lstm_size,
                                 3,
                                 padding='valid',
                                 activation='relu',
                                 strides=1))
                # we use max pooling:
                model.add(GlobalMaxPooling1D())

            else:
                model.add(LSTM(self.hidden_lstm_size))
            model.add(Dense(self.dense_layer_size
                            ))
            model.add(Dense(1, activation='sigmoid'))
            model.compile(loss='binary_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                          decay=0.0)
                          , metrics=['accuracy'])
            print("models summary:")
            print(model.summary())
            print('training size = %s' % str(x_train.shape[0]))

            print('training..')
            if self.use_class_weight:
                class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
                print(class_weights)
                print("class_weight is %s" % str(class_weights))
                if validation_split is None and x_val is None and x_val is None:
                    model.fit(x=x_train, y=y_train, epochs=self.epochs_model,
                              class_weight=dict(enumerate(class_weights)),
                              batch_size=self.batch_size, shuffle=shuffle,
                              callbacks=callbacks)
                elif validation_split is None:
                    model.fit(x=x_train, y=y_train, epochs=self.epochs_model,
                              class_weight=dict(enumerate(class_weights)),
                              batch_size=self.batch_size, shuffle=shuffle,
                              callbacks=callbacks, validation_data=(x_val, y_val))
                else:
                    model.fit(x=x_train, y=y_train, epochs=self.epochs_model,
                              class_weight=dict(enumerate(class_weights)),
                              batch_size=self.batch_size, shuffle=shuffle,
                              callbacks=callbacks, validation_split=validation_split)
            else:
                print('no class weight')
                if validation_split is None:
                    model.fit(x=x_train, y=y_train, epochs=self.epochs_model, batch_size=self.batch_size,
                              callbacks=callbacks, shuffle=shuffle)
                else:
                    model.fit(x=x_train, y=y_train, epochs=self.epochs_model, batch_size=self.batch_size,
                              callbacks=callbacks, shuffle=shuffle, validation_split=validation_split)
            try:
                print('saving models..')
                print(model.params['stopped_epoch'])
                model.save("%s/models.h5" % self.model_path)
                print('models saved')
            except:
                print('cant save the models')
        print('testing..')
        return (self.evalute_model(model, x_test, y_test), self.evalute_model(model, x_test_cold, y_test_cold))

    def __init__(self, encode_mode=2, use_class_weight=True, max_len_session=10, max_features=None,
                 hidden_lstm_size=150, lr=0.001,
                 embedding_size=50, dense_layer_size=200, epochs_model=1000, batch_size=1024 * 16, model_path="models",
                 print_log=False,
                 predict_path="detailed_prediction/model_predict.csv"):
        self.encode_mode = encode_mode
        self.use_class_weight = use_class_weight
        self.max_len_session = max_len_session
        self.hidden_lstm_size = hidden_lstm_size
        self.embedding_size = embedding_size
        self.dense_layer_size = dense_layer_size
        self.epochs_model = epochs_model
        self.print_log = print_log
        self.model_path = model_path
        self.predict_path = predict_path
        self.max_features = max_features
        self.batch_size = batch_size
        self.lr = lr

    def evalute_model(self, model, x_test, y_test):
        if x_test is None or len(x_test.shape) == 0:
            return -1
        y_pred = model.predict(x=x_test)
        with open(self.predict_path, "w") as fw:
            fw.write("x_test,y_test,y_pred\n")
            for i in range(y_pred.shape[0]):
                y_pred[i] = y_pred[i][0]
                fw.write("%s,%s,%s\n" % (str(x_test[i]).replace("\n", " ").replace(",", ";"),
                                         str(y_test[i]), str(float(y_pred[i]))))
        if self.has_auc(y_test):
            auc = metrics.roc_auc_score(y_test, y_pred)
        else:
            auc = 'no auc'
        print(auc)
        return auc

    def has_auc(self, y_test):
        count_buy_in_y_test = 0
        for i in y_test:
            count_buy_in_y_test += i
        has_auc_bool = count_buy_in_y_test > 0 and count_buy_in_y_test < y_test.shape[0]
        return has_auc_bool

    def print_states(self, name, y):
        len = y.shape[0]
        print('%s = %s' % (name, len))
        end_with_pur = y.sum()
        print('%s end with purchase = %s' % (str(name), str(end_with_pur)))
        print('%s end with without purchase = %s' % (str(name), str(
            len - end_with_pur)))
