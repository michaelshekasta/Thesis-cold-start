#!/usr/bin/env python
# -*- coding: utf-8 -*-

import yoochose_catalog
import system_utils
from cold_start_exp import ColdStartExp
from cold_start_integrated import ColdStartIntegrated
from cold_start_session_reader import ColdStartSessionReader
from encode_reader_ext import EncodeReaderExt
from item2vec import Item2vec
from session_reader import SessionReader
import config_model
from session_remover import SessionsRemover

if __name__ == "__main__":
    system_utils.redirect_stdout("output.log")
    with open("predictions.csv", "w") as fw2:
        fw2.write("exp_name,model,type_exp,precent,epoch,index,x_test,y_test,y_pred,trash05,right\n")
    print('starting...')
    print('create catalog...')
    c = yoochose_catalog.Catalog(
        dir_path="catalog", use_german_token=config_model.use_german_tokenizer, mode=3)
    items = c.get_items()

    print("prepare session file")
    model_name = 'general'
    system_utils.create_dir('%s' % model_name)
    system_utils.create_dir('%s/data_before_encode' % model_name)
    s = SessionReader(input_path_session_actions='%s/eventsquance.txt' % config_model.dir_input,
                      input_path_session_info='%s/list session.csv' % config_model.dir_input, items_list=items,
                      test_dates=config_model.dates_for_test
                      , maxlen=config_model.max_len_session
                      , minlen=config_model.min_len_session
                      , wipe_items_not_in_train=config_model.wipe_items_not_in_train
                      , encode_dir='%s/data_before_encode' % model_name)

    item2vec = Item2vec(catalog=c, embedding_size=config_model.item2vec_embedding_size, hidden_size=10,
                        max_len=config_model.max_len_item_emb,
                        epoches=config_model.item2vec_epoch)

    for remove_items in [False, True]:
        config_name = 'remove_sessions'
        if remove_items:
            config_name = 'remove_items'
        for percent_to_remove in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            exp_name = '%s_%s' % (
                config_name, str(percent_to_remove).replace(".", ""))
            system_utils.create_dir("%s" % exp_name)
            system_utils.create_dir("%s/data_before_encode" % exp_name)
            system_utils.create_dir("%s/data_after_encode1" % exp_name)
            system_utils.create_dir("%s/data_after_encode2" % exp_name)
            print("starting exp: %s" % exp_name)
            if remove_items:
                new_sessions_set = ColdStartSessionReader(catalog=c, train=s.get_train(),
                                                          test=s.get_test(),
                                                          min_item_in_category=config_model.min_item_to_remove,
                                                          precent_remove=percent_to_remove,
                                                          data_out_path="%s/data_before_encode" % exp_name)
            else:
                new_sessions_set = SessionsRemover(catalog=c, train=s.get_train(), test=s.get_test(),
                                                   percent_remove=percent_to_remove,
                                                   data_out_path="%s/data_before_encode" % exp_name)
            encode_session2 = EncodeReaderExt(train_df=new_sessions_set.get_new_train(),
                                              test_df1=new_sessions_set.get_non_new_item_test_set(),
                                              test_df2=new_sessions_set.get_new_item_test_set(),
                                              encode_path="%s/data_after_encode2" % exp_name,
                                              catalog=c,
                                              item2vec=item2vec.item2emb,
                                              encode_mode=2)
            encode_session1 = EncodeReaderExt(train_df=new_sessions_set.get_new_train(),
                                              test_df1=new_sessions_set.get_non_new_item_test_set(),
                                              test_df2=new_sessions_set.get_new_item_test_set(), catalog=c,
                                              encode_path="%s/data_after_encode1" % exp_name,
                                              item2vec=None,
                                              encode_mode=1)
            for integrated in [False, True]:
                for use_item_emb in [False, True]:
                    if integrated and use_item_emb:
                        continue
                    model_name = 'baseline'
                    if use_item_emb and not integrated:
                        model_name = 'ourmodel'
                    elif integrated:
                        model_name = 'integrated'
                    print('running exps of our model:%s' % str(use_item_emb))
                    exp_name = '%s_%s_%s' % (
                        model_name, config_name, str(percent_to_remove).replace(".", ""))
                    system_utils.create_dir("%s/models" % exp_name)
                    exp_name_path = "%s/output.log" % exp_name
                    system_utils.create_dir("%s/detailed_prediction" % exp_name)
                    system_utils.redirect_stdout(exp_name_path)
                    if integrated and not use_item_emb:
                        exp_name = 'integrated_%s_%s' % (
                            config_name, str(percent_to_remove).replace(".", ""))
                        x_train1 = encode_session1.get_x_train()
                        x_train2 = encode_session2.get_x_train()
                        y_train = encode_session2.get_y_train()
                        x_test11 = encode_session1.get_x_test1()
                        x_test12 = encode_session2.get_x_test1()
                        y_test1 = encode_session2.get_y_test1()
                        x_test21 = encode_session1.get_x_test2()
                        x_test22 = encode_session2.get_x_test2()
                        y_test2 = encode_session2.get_y_test2()
                        exp = ColdStartIntegrated(use_class_weight=config_model.use_class_weight, lr=config_model.lr,
                                                  max_features=len(encode_session1.items_in_train) + 1,
                                                  epochs_model=config_model.epochs_model,
                                                  batch_size=config_model.model_batch_size,
                                                  embedding_size=config_model.item2vec_embedding_size,
                                                  dense_layer_size=config_model.dense_layer_size,
                                                  predict_path='%s/model_predict.csv' % exp_name,
                                                  model_path='%s/models' % exp_name,
                                                  exp_name=exp_name
                                                  )
                        auc = exp.run_exp(x_train1=x_train1, x_train2=x_train2, y_train=y_train,
                                          x_test1=x_test11, x_test2=x_test12, y_test=y_test1,
                                          x_test_cold1=x_test21, x_test_cold2=x_test22,
                                          y_test_cold=y_test2, use_cnn=config_model.use_cnn,
                                          shuffle=config_model.shuffle,
                                          validation_split=config_model.validation_split
                                          )
                    else:
                        if use_item_emb:
                            encode_session = encode_session2
                        else:
                            encode_session = encode_session1
                        x_train = encode_session.get_x_train()
                        y_train = encode_session.get_y_train()
                        x_test1 = encode_session.get_x_test1()
                        y_test1 = encode_session.get_y_test1()
                        x_test2 = encode_session.get_x_test2()
                        y_test2 = encode_session.get_y_test2()
                        if config_model.run_deep_model:
                            if use_item_emb:
                                exp = ColdStartExp(use_class_weight=config_model.use_class_weight, lr=config_model.lr,
                                                   epochs_model=config_model.epochs_model,
                                                   batch_size=config_model.model_batch_size,
                                                   embedding_size=config_model.item2vec_embedding_size,
                                                   dense_layer_size=config_model.dense_layer_size,
                                                   predict_path='%s/model_predict.csv' % exp_name,
                                                   model_path='%s/models' % exp_name,
                                                   exp_name=exp_name
                                                   )
                            else:
                                exp = ColdStartExp(use_class_weight=config_model.use_class_weight, encode_mode=1,
                                                   max_features=len(encode_session1.items_in_train) + 1,
                                                   lr=config_model.lr,
                                                   epochs_model=config_model.epochs_model,
                                                   batch_size=config_model.model_batch_size,
                                                   embedding_size=config_model.model_embedding_size,
                                                   dense_layer_size=config_model.dense_layer_size,
                                                   predict_path='%s/model_predict.csv' % exp_name,
                                                   model_path='%s/models' % exp_name,
                                                   exp_name=exp_name
                                                   )
                            auc = exp.run_exp(x_train=x_train, y_train=y_train, x_test=x_test1, y_test=y_test1,
                                              x_test_cold=x_test2,
                                              y_test_cold=y_test2, use_cnn=config_model.use_cnn,
                                              shuffle=config_model.shuffle,
                                              validation_split=config_model.validation_split
                                              )
                    print(auc)
                    if not config_model.debug:
                        system_utils.send_email(body='exp_name:%s the auc is %s' % (exp_name, str(auc)))
                    else:
                        if not config_model.debug:
                            system_utils.send_email()
