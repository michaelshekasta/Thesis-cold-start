#!/usr/bin/env python
# -*- coding: utf-8 -*-

import config_model
import system_utils
import yoochose_catalog
from encode_reader import EncodeReader
from item2vec import Item2vec
from regular_exp import RegularExp
from session_reader import SessionReader

if __name__ == "__main__":
    with open("predictions.csv", "w") as fw2:
        fw2.write("exp_name,epoch,index,x_test,y_test,y_pred,trash05,right\n")
    # for mode in [1, 2, 3, 4, 5, 6, 7]:
    for mode in [1]:
        # for clean_words in [False, True]:
        for clean_words in [False]:
            system_utils.redirect_stdout("output_%s.log" % str(mode))
            print('starting...')
            print('create catalog...')
            c = yoochose_catalog.Catalog(
                dir_path="catalog", use_german_token=config_model.use_german_tokenizer, mode=mode,
                clean_words_punctuation=clean_words)
            items = c.get_items()

            print("prepare session file")
            model_name = 'general_%s' % (str(mode))
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

            config_name = 'no_cold_start'

            for use_item_emb in [False, True]:
            # for use_item_emb in [True]:
            # for use_item_emb in [False]:
                model_name = 'baseline'
                if use_item_emb:
                    model_name = 'ourmodel'
                print('running exps of our model:%s' % str(use_item_emb))
                exp_name = '%s_%s_%s_%s_%s' % (model_name, config_name, "0", mode, str(clean_words))
                exp_name_path = "%s/output.log" % exp_name
                system_utils.create_dir("%s" % exp_name)
                system_utils.create_dir("%s/data_before_encode" % exp_name)
                system_utils.create_dir("%s/data_after_encode" % exp_name)
                system_utils.create_dir("%s/models" % exp_name)
                system_utils.create_dir("%s/detailed_prediction" % exp_name)
                system_utils.redirect_stdout(exp_name_path)
                print("starting exp: %s" % exp_name)
                if use_item_emb:
                    encode_session = EncodeReader(train_df=s.get_train(),
                                                  test_df=s.get_test(),
                                                  encode_path="%s/data_after_encode" % exp_name,
                                                  catalog=c,
                                                  item2vec=item2vec.item2emb,
                                                  encode_mode=2)
                else:
                    encode_session = EncodeReader(train_df=s.get_train(),
                                                  test_df=s.get_test(),
                                                  encode_path="%s/data_after_encode" % exp_name,
                                                  catalog=c,
                                                  item2vec=None,
                                                  encode_mode=1)
                x_train = encode_session.get_x_train()
                y_train = encode_session.get_y_train()
                x_test = encode_session.get_x_test()
                y_test = encode_session.get_y_test()
                if config_model.run_deep_model:
                    if use_item_emb:
                        exp = RegularExp(use_class_weight=config_model.use_class_weight, lr=config_model.lr,
                                         epochs_model=config_model.epochs_model,
                                         batch_size=config_model.model_batch_size,
                                         embedding_size=config_model.item2vec_embedding_size,
                                         dense_layer_size=config_model.dense_layer_size,
                                         exp_name=exp_name
                                         )
                    else:
                        exp = RegularExp(use_class_weight=config_model.use_class_weight, encode_mode=1,
                                         max_features=len(items) + 1, lr=config_model.lr,
                                         epochs_model=config_model.epochs_model,
                                         batch_size=config_model.model_batch_size,
                                         embedding_size=config_model.model_embedding_size,
                                         dense_layer_size=config_model.dense_layer_size,
                                         exp_name=exp_name
                                         )
                    auc = exp.run_exp(x_train=x_train, y_train=y_train, x_test=x_test, y_test=y_test,
                                      use_cnn=config_model.use_cnn, shuffle=config_model.shuffle,
                                      validation_split=config_model.validation_split
                                      )
                    print(auc)
                    if not config_model.debug:
                        system_utils.send_email(body='exp_name:%s the auc is %s' % (exp_name, str(auc)))
                else:
                    if not config_model.debug:
                        system_utils.send_email()
