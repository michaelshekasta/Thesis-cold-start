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
    with open("predictions.csv", "w") as fw2:
        fw2.write("exp_name,model,type_exp,precent,epoch,index,x_test,y_test,y_pred,trash05,right\n")
    system_utils.redirect_stdout("output.log")
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
        # for item2vec_embedding_size in [5, 20, 25, 50, 75, 100, 150, 300]:
        #     item2vec = Item2vec(catalog=c, embedding_size=item2vec_embedding_size, hidden_size=10,
        #                         max_len=config_model.max_len_item_emb,
        #                         epoches=config_model.item2vec_epoch)

        # remove_items = False
        config_name = 'remove_sessions'
        if remove_items:
            config_name = 'remove_items'
        for percent_to_remove in [0.1, 0.2, 0.3, 0.4, 0.5, 0.6, 0.7, 0.8]:
            # for percent_to_remove in [0.9, 0.95, 0.995]:
            # for percent_to_remove in [0.1]:
            # for use_item_emb in [False, True]:
            # for use_item_emb in [True]:
            model_name = 'integrated'
            print('running exps of our model:%s' % str("integrated"))
            # exp_name = '%s_%s_%s_%s' % (
            exp_name = '%s_%s_%s' % (
                # model_name, config_name, str(int(10 * percent_to_remove)), str(item2vec_embedding_size))
                model_name, config_name, str(percent_to_remove).replace(".", ""))
            exp_name_path = "%s/output.log" % exp_name
            # if system_utils.is_file_exist(exp_name_path):
            #     continue
            system_utils.create_dir("%s" % exp_name)
            system_utils.create_dir("%s/data_before_encode" % exp_name)
            system_utils.create_dir("%s/data_after_encode1" % exp_name)
            system_utils.create_dir("%s/data_after_encode2" % exp_name)
            system_utils.create_dir("%s/models" % exp_name)
            system_utils.create_dir("%s/detailed_prediction" % exp_name)
            system_utils.redirect_stdout(exp_name_path)
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
                                              encode_mode=1, )

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
                                      max_features=len(items) + 1,
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
                              y_test_cold=y_test2, use_cnn=config_model.use_cnn, shuffle=config_model.shuffle,
                              validation_split=config_model.validation_split
                              )
            print(auc)
            if not config_model.debug:
                system_utils.send_email(body='exp_name:%s the auc is %s' % (exp_name, str(auc)))
            else:
                if not config_model.debug:
                    system_utils.send_email()
