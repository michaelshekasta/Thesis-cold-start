#!/usr/bin/env python
# -*- coding: utf-8 -*-
import random

import yoochose_catalog
import system_utils
from cold_start_session_reader import ColdStartSessionReader
from encode_reader import EncodeReader
from item2vec import Item2vec
from regular_exp import RegularExp
from session_reader import SessionReader

random.seed(0)

if __name__ == "__main__":
    # check_data()
    # debug = True
    use_item_emb = True
    use_class_weight = True
    percent_to_remove = 0.1
    min_item_to_remove = int(1 / percent_to_remove)
    use_german_tokenizer = False
    lr = 0.0001

    min_len_session = 1
    max_len_session = 10
    hidden_size_gru = 150
    embedding_size = 50
    dense_layer_size = 200
    item2vec_epoch = 30
    epochs_model = 20

    dir_input = 'data'

    print('starting...')
    print('create catalog...')
    c = yoochose_catalog.Catalog(
        dir_path="catalog", use_german_token=use_german_tokenizer)
    items = c.get_items()
    print("prepare session file")
    dates_for_test = ['2016-08-31']
    print('create item2vec...')
    s = SessionReader(input_path_session_actions='%s/eventsquance.txt' % dir_input,
                      input_path_session_info='%s/list session.csv' % dir_input, items_list=items,
                      test_dates=dates_for_test
                      , maxlen=max_len_session
                      , minlen=min_len_session)
