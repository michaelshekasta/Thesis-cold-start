"""
This class will represent the experiment of regular calc
"""
import keras
from keras import Input, Model
from keras.callbacks import CSVLogger, ModelCheckpoint, Callback
from keras.layers import Masking, LSTM, Dense, Embedding, Conv1D, GlobalMaxPooling1D, concatenate
from keras.models import load_model, Sequential
from sklearn import metrics
from sklearn.utils import class_weight

import system_utils
import numpy as np


class printTest(Callback):
    def __init__(self, x_test11, x_test12, y_test1, x_test21, x_test22, y_test2, exp_name='exp'):
        Callback.__init__(self)
        self.x_test11 = x_test11
        self.x_test12 = x_test12
        self.y_test1 = y_test1
        self.x_test21 = x_test21
        self.x_test22 = x_test22
        self.y_test2 = y_test2
        self.exp_name = exp_name
        f = open("auc_total_%s.csv" % exp_name, "w")
        f.write("epoch,type,auc\n")
        f.close()

    def on_epoch_end(self, epoch, logs=None):
        self.calc_auc_epoch(epoch, 'regular', self.x_test11, self.x_test12, self.y_test1)
        self.calc_auc_epoch(epoch, 'cold_start', self.x_test21, self.x_test22, self.y_test2)
        self.calc_auc_epoch(epoch, 'integrated', np.append(self.x_test11, self.x_test21, 0),
                            np.append(self.x_test12, self.x_test22, 0),
                            np.append(self.y_test1, self.y_test2, 0))

    def calc_auc_epoch(self, epoch, name, x_test1, x_test2, y_test):
        if x_test1 is None or len(x_test1.shape) == 0:
            return
        y_pred = self.model.predict(x=[x_test1, x_test2])
        params = self.exp_name.split("_")
        model = params[0]
        exp_type = ''.join(params[1:-1])
        percent = int(params[-1])
        with open("%s/detailed_prediction/model_predict_%s_%s.csv" % (self.exp_name, name, epoch), "w") as fw:
            with open("predictions.csv", "a") as fw2:
                fw.write("x_test,y_test,y_pred\n")
                for i in range(y_pred.shape[0]):
                    y_pred[i] = y_pred[i][0]
                    fw.write("%s,%s,%s\n" % ("NaN",
                                             str(y_test[i]), str(float(y_pred[i]))))
                    if float(y_pred[i]) > 0.5:
                        pred = 1
                    else:
                        pred = 0
                    if pred == y_test[i]:
                        right = 1
                    else:
                        right = 0
                    fw2.write("%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                        self.exp_name, model, exp_type, str(percent), str(epoch), str(i),
                        str('NA').replace("\n", " ").replace(",", ";"),
                        str(y_test[i]), str(float(y_pred[i])), str(pred), str(right)))
        if self.has_auc(y_test):
            auc = metrics.roc_auc_score(y_test, y_pred)
        else:
            auc = 'no auc'
        print("%s auc test = %s" % (name, str(auc)))
        f = open("auc_total_%s.csv" % self.exp_name, "a")
        f.write("%s,%s,%s\n" % (epoch, name, auc))
        f.close()

    def has_auc(self, y_test):
        count_buy_in_y_test = 0
        for i in y_test:
            count_buy_in_y_test += i
        has_auc_bool = count_buy_in_y_test > 0 and count_buy_in_y_test < y_test.shape[0]
        return has_auc_bool


class ColdStartIntegrated(object):
    def run_exp(self, x_train1, x_train2, y_train, x_test1, x_test2, y_test, x_test_cold1, x_test_cold2, y_test_cold,
                shuffle=True, use_cnn=False,
                validation_split=None):
        self.print_states("Run Training set ", y_train)
        self.print_states("Run non new item test set", y_test)
        self.print_states("Run new item test set", y_test_cold)
        self.print_states("Run full test set", np.append(y_test, y_test_cold, 0))
        if self.exp_name == None:
            info_path = 'epochs_info.csv'
        else:
            info_path = "%s/%s" % (self.exp_name, 'epochs_info.csv')
        if not validation_split is None:
            callbacks = [
                printTest(x_test11=x_test1, x_test12=x_train2, y_test1=y_test, x_test21=x_test_cold1,
                          x_test22=x_test_cold2, y_test2=y_test_cold,
                          exp_name=self.exp_name),
                CSVLogger(info_path),
                ModelCheckpoint("%s/best_model.h5" % self.model_path, monitor='val_loss', save_best_only=True,
                                verbose=1, mode='min'
                                )]
        else:
            callbacks = [
                printTest(x_test11=x_test1, x_test12=x_test2, y_test1=y_test, x_test21=x_test_cold1,
                          x_test22=x_test_cold2, y_test2=y_test_cold,
                          exp_name=self.exp_name),
                CSVLogger(info_path),
                ModelCheckpoint("%s/best_model.h5" % self.model_path, monitor='loss', save_best_only=True, verbose=1,
                                )]
        if system_utils.is_file_exist("%s/models.h5" % self.model_path):
            print("loading models..")
            model = load_model("%s/models.h5" % self.model_path)
            print("loaded..")
        else:
            print("train models")
            # max_features = len(items) + 1
            input1 = Input(shape=(x_train1.shape[1],), dtype='int32')
            emb_layer = Embedding(self.max_features, self.embedding_size, input_length=self.max_len_session,
                                  mask_zero=True)(input1)
            rnn_layer1 = LSTM(self.hidden_lstm_size)(emb_layer)
            input2 = Input(shape=(x_train2.shape[1], x_train2.shape[2]), dtype='float32')
            masking_layer = Masking(mask_value=0.0, input_shape=(x_train2.shape[1], x_train2.shape[2]))(input2)
            rnn_layer2 = LSTM(self.hidden_lstm_size)(masking_layer)
            concat = concatenate([rnn_layer1, rnn_layer2], axis=1)
            dense1 = Dense(self.dense_layer_size)(concat)
            dense2 = Dense(1, activation='sigmoid')(dense1)
            model = Model(inputs=[input1, input2], outputs=dense2)

            model.compile(loss='binary_crossentropy',
                          optimizer=keras.optimizers.Adam(lr=self.lr, beta_1=0.9, beta_2=0.999, epsilon=1e-08,
                                                          decay=0.0)
                          , metrics=['accuracy'])
            print("models summary:")
            print(model.summary())
            print('training size = %s' % str(x_train1.shape[0]))

            print('training..')
            if self.use_class_weight:
                class_weights = class_weight.compute_class_weight('balanced', np.unique(y_train), y_train)
                print(class_weights)
                print("class_weight is %s" % str(class_weights))
                model.fit(x=[x_train1, x_train2], y=y_train, epochs=self.epochs_model,
                          class_weight=dict(enumerate(class_weights)),
                          batch_size=self.batch_size, shuffle=shuffle,
                          callbacks=callbacks)
            else:
                model.fit(x=[x_train1, x_train2], y=y_train, epochs=self.epochs_model, batch_size=self.batch_size,
                          callbacks=callbacks, shuffle=shuffle)
            try:
                print('saving models..')
                model.save("%s/models.h5" % self.model_path)
                print('models saved')
            except:
                print
                'cant save the models'
        print('testing..')
        system_utils.send_mail_with_attach(files=["auc_total_%s.csv" % self.exp_name])
        return (self.evalute_model(model, x_test1, x_test2, y_test),
                self.evalute_model(model, x_test_cold1, x_test_cold2, y_test_cold))

    def __init__(self, encode_mode=2, use_class_weight=True, max_len_session=10, max_features=None,
                 hidden_lstm_size=150, lr=0.001,
                 embedding_size=50, dense_layer_size=200, epochs_model=1000, batch_size=1024 * 16, model_path="models",
                 print_log=False,
                 predict_path="detailed_prediction/model_predict.csv",
                 exp_name=None):
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
        self.exp_name = exp_name
        # if not self.exp_name == None:
        #     self.predict_path = "%s/%s" % (self.exp_name, predict_path)

    def evalute_model(self, model, x_test1, x_test2, y_test):
        if x_test1 is None or len(x_test1.shape) == 0:
            return -1
        y_pred = model.predict(x=[x_test1, x_test2])
        with open(self.predict_path, "w") as fw:
            fw.write("x_test,y_test,y_pred\n")
            for i in range(y_pred.shape[0]):
                y_pred[i] = y_pred[i][0]
                fw.write("%s,%s,%s\n" % ("NaN",
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
