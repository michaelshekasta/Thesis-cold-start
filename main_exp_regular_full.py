#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config_model
import system_utils
import yoochose_catalog
from encode_reader import EncodeReader
from item2vec import Item2vec
from regular_exp import RegularExp
from regular_exp_integreted import RegularExpIntegreted
from session_reader import SessionReader

if __name__ == "__main__":
    with open("predictions.csv", "w") as fw2:
        fw2.write("exp_name,epoch,index,x_test,y_test,y_pred,trash05,right\n")
        mode = 1
        print('starting...')
        print('create catalog...')
        c = yoochose_catalog.Catalog(
            dir_path="catalog", use_german_token=config_model.use_german_tokenizer, mode=mode)
        items = c.get_items()

        print("prepare session file")
        model_name = 'general_%s' % (str(mode))
        system_utils.create_dir('%s' % model_name)
        system_utils.create_dir('%s/data_before_encode' % model_name)
        print("reading sessions")
        s = SessionReader(input_path_session_actions='%s/eventsquance.txt' % config_model.dir_input,
                          input_path_session_info='%s/list session.csv' % config_model.dir_input, items_list=items,
                          test_dates=config_model.dates_for_test
                          , maxlen=config_model.max_len_session
                          , minlen=config_model.min_len_session
                          , wipe_items_not_in_train=config_model.wipe_items_not_in_train
                          , encode_dir='%s/data_before_encode' % model_name)

        items = s.items_in_train
        item2vec = Item2vec(catalog=c, embedding_size=config_model.item2vec_embedding_size, hidden_size=10,
                            max_len=config_model.max_len_item_emb,
                            epoches=config_model.item2vec_epoch)

        print("items =[%s]" % ','.join(items))
        config_name = 'no_cold_start'

        model_name = 'general'
        print('running exps of our model:%s' % model_name)
        exp_name = '%s_%s_%s_%s' % (model_name, config_name, "0", mode)
        exp_name_path = "%s/output.log" % exp_name
        system_utils.create_dir("%s" % exp_name)
        system_utils.create_dir("%s/models" % exp_name)
        system_utils.create_dir("%s/detailed_prediction" % exp_name)
        system_utils.create_dir("%s/data_before_encode" % exp_name)
        system_utils.create_dir("%s/data_after_encode1" % exp_name)
        system_utils.create_dir("%s/data_after_encode2" % exp_name)
        # system_utils.redirect_stdout(exp_name_path)
        print("starting exp: %s" % exp_name)

        encode_session1 = EncodeReader(train_df=s.get_train(),
                                       test_df=s.get_test(),
                                       encode_path="%s/data_after_encode1" % exp_name,
                                       catalog=c,
                                       item2vec=None,
                                       encode_mode=1)
        encode_session2 = EncodeReader(train_df=s.get_train(),
                                       test_df=s.get_test(),
                                       encode_path="%s/data_after_encode2" % exp_name,
                                       catalog=c,
                                       item2vec=item2vec.item2emb,
                                       encode_mode=2)
        for integrated in [False, True]:
            for use_baseline_model in [False, True]:
                if integrated and use_baseline_model:
                    continue
                model_name = 'textmodel'
                if use_baseline_model and not integrated:
                    model_name = 'baseline'
                if integrated and not use_baseline_model:
                    model_name = 'integrated'

                exp_name = '%s_%s_%s_%s' % (model_name, config_name, "0", mode)
                system_utils.create_dir("%s" % exp_name)
                system_utils.create_dir("%s/models" % exp_name)
                system_utils.create_dir("%s/detailed_prediction" % exp_name)
                print("running model %s" % exp_name)
                x_train1 = encode_session1.get_x_train()
                x_train2 = encode_session2.get_x_train()
                y_train = encode_session1.get_y_train()
                x_test1 = encode_session1.get_x_test()
                x_test2 = encode_session2.get_x_test()
                y_test = encode_session1.get_y_test()
                if config_model.run_deep_model:
                    if integrated:
                        exp = RegularExpIntegreted(use_class_weight=config_model.use_class_weight,
                                                   max_features=len(items) + 1, lr=config_model.lr,
                                                   epochs_model=config_model.epochs_model,
                                                   batch_size=config_model.model_batch_size,
                                                   embedding_size=config_model.item2vec_embedding_size,
                                                   dense_layer_size=config_model.dense_layer_size,
                                                   exp_name=exp_name
                                                   )
                        auc = exp.run_exp(x_train1=x_train1, x_train2=x_train2, y_train=y_train, x_test1=x_test1, x_test2=x_test2,
                                          y_test=y_test,
                                          use_cnn=config_model.use_cnn, shuffle=config_model.shuffle,
                                          validation_split=config_model.validation_split
                                          )
                        print(auc)
                    else:
                        if use_baseline_model:
                            x_train = encode_session1.get_x_train()
                            y_train = encode_session1.get_y_train()
                            x_test = encode_session1.get_x_test()
                            y_test = encode_session1.get_y_test()
                            exp = RegularExp(use_class_weight=config_model.use_class_weight, encode_mode=1,
                                             max_features=len(items) + 1, lr=config_model.lr,
                                             epochs_model=config_model.epochs_model,
                                             batch_size=config_model.model_batch_size,
                                             embedding_size=config_model.model_embedding_size,
                                             dense_layer_size=config_model.dense_layer_size,
                                             exp_name=exp_name
                                             )
                        else:
                            x_train = encode_session2.get_x_train()
                            y_train = encode_session2.get_y_train()
                            x_test = encode_session2.get_x_test()
                            y_test = encode_session2.get_y_test()
                            exp = RegularExp(use_class_weight=config_model.use_class_weight, lr=config_model.lr,
                                             epochs_model=config_model.epochs_model,
                                             batch_size=config_model.model_batch_size,
                                             embedding_size=config_model.item2vec_embedding_size,
                                             dense_layer_size=config_model.dense_layer_size,
                                             exp_name=exp_name
                                             )
                        auc = exp.run_exp(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                          use_cnn=config_model.use_cnn, shuffle=config_model.shuffle,
                                          validation_split=config_model.validation_split
                                          )
                    if not config_model.debug:
                        system_utils.send_email(body='exp_name:%s the auc is %s' % (exp_name, str(auc)))
                else:
                    if not config_model.debug:
                        system_utils.send_email()
