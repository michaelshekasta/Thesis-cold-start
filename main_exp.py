#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config_model
import system_utils
import yoochose_catalog
from cold_start_exp_val import ColdStartExpVal
from cold_start_session_reader import ColdStartSessionReader
from encode_reader_val import EncodeReaderVal
from item2vec import Item2vec
from session_reader_val import SessionReaderVal
from session_remover import SessionsRemover

if __name__ == "__main__":

    print('starting...')
    print('create catalog...')
    c = yoochose_catalog.Catalog(
        dir_path="catalog", use_german_token=config_model.use_german_tokenizer)
    items = c.get_items()
    print("prepare session file")
    print('create item2vec...')
    if config_model.use_item_emb == True and not (
            system_utils.is_file_exist(config_model.x_train_path) and system_utils.is_file_exist(
        config_model.y_train_path) and system_utils.is_file_exist(
        config_model.x_test_path1) and system_utils.is_file_exist(
        config_model.y_test_path1) and system_utils.is_file_exist(
        config_model.x_test_path2) and system_utils.is_file_exist(
        config_model.y_test_path2)):
        item2vec = Item2vec(catalog=c, embedding_size=config_model.model_embedding_size, hidden_size=10, max_len=10,
                            epoches=config_model.item2vec_epoch)
    else:
        item2vec = None
    s = SessionReaderVal(input_path_session_actions='%s/eventsquance.txt' % config_model.dir_input,
                         input_path_session_info='%s/list session.csv' % config_model.dir_input, items_list=items,
                         val_dates=config_model.dates_for_val,
                         test_dates=config_model.dates_for_test,
                         maxlen=config_model.max_len_session,
                         minlen=config_model.min_len_session,
                         wipe_items_not_in_train=config_model.wipe_items_not_in_train)
    if config_model.remove_items:
        remover = ColdStartSessionReader(c, s.get_train(), test=s.get_test(),
                                         min_item_in_category=config_model.min_item_to_remove,
                                         precent_remove=config_model.percent_to_remove)
    else:
        remover = SessionsRemover(catalog=c, train=s.get_train(), test=s.get_test(),
                                  percent_remove=config_model.percent_to_remove)
    if config_model.use_item_emb:
        if item2vec is None:
            encode_session = EncodeReaderVal(train_df=remover.get_new_train(),
                                             val_df=s.get_val(),
                                             test_df1=remover.get_non_new_item_test_set(),
                                             test_df2=remover.get_new_item_test_set(), catalog=c,
                                             item2vec=None,
                                             encode_mode=2)
        else:
            encode_session = EncodeReaderVal(train_df=remover.get_new_train(),
                                             val_df=s.get_val(),
                                             test_df1=remover.get_non_new_item_test_set(),
                                             test_df2=remover.get_new_item_test_set(), catalog=c,
                                             item2vec=item2vec.item2emb,
                                             encode_mode=2)
    else:
        encode_session = EncodeReaderVal(train_df=remover.get_new_train(),
                                         val_df=s.get_val(),
                                         test_df1=remover.get_non_new_item_test_set(),
                                         test_df2=remover.get_new_item_test_set(), catalog=c,
                                         item2vec=None,
                                         encode_mode=1)

    x_train = encode_session.get_x_train()
    y_train = encode_session.get_y_train()

    x_test1 = encode_session.get_x_test1()
    y_test1 = encode_session.get_y_test1()

    x_test2 = encode_session.get_x_test2()
    y_test2 = encode_session.get_y_test2()

    x_val = encode_session.get_x_val()
    y_val = encode_session.get_y_val()

    if config_model.run_deep_model:
        if config_model.use_item_emb:
            exp = ColdStartExpVal(use_class_weight=config_model.use_class_weight, lr=config_model.lr,
                                  epochs_model=config_model.epochs_model, batch_size=config_model.model_batch_size,
                                  embedding_size=config_model.model_embedding_size,
                                  dense_layer_size=config_model.dense_layer_size,
                                  hidden_lstm_size=config_model.hidden_size_rnn)
        else:
            exp = ColdStartExpVal(use_class_weight=config_model.use_class_weight, encode_mode=1,
                                  max_features=len(items) + 1, lr=config_model.lr,
                                  epochs_model=config_model.epochs_model,
                                  batch_size=config_model.model_batch_size, embedding_size=config_model.model_embedding_size,
                                  dense_layer_size=config_model.dense_layer_size,
                                  hidden_lstm_size=config_model.hidden_size_rnn)
        auc = exp.run_exp(x_train=x_train, y_train=y_train, x_val=x_val, y_val=y_val, x_test=x_test1, y_test=y_test1,
                          x_test_cold=x_test2,
                          y_test_cold=y_test2, use_cnn=config_model.use_cnn, shuffle=config_model.shuffle,
                          validation_split=config_model.validation_split)
        print(auc)
        if not config_model.debug:
            system_utils.send_email(body='our model: %s , remove items: %s with the auc is %s' % (
                str(config_model.use_item_emb), str(config_model.remove_items), str(auc)))
    else:
        if not config_model.debug:
            system_utils.send_email()
