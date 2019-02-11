import random
import system_utils
import numpy as np

debug = False
debug = True

# ranodm issues
SEED = 11
np.random.seed(SEED)
random.seed(SEED)
try:
    import tensorflow as tf

    tf.set_random_seed(SEED)
    # from keras.backend import manual_variable_initialization
    # manual_variable_initialization(True)
except:
    pass

try:
    import os

    os.environ['PYTHONHASHSEED'] = '0'
except:
    pass

# model params
use_item_emb = False
use_class_weight = True
remove_items = True
percent_to_remove = 0.1
min_item_to_remove = int(1 / percent_to_remove)
use_german_tokenizer = True
max_len_item_emb = 10
run_deep_model = True
lr = 0.001
model_batch_size = 256

use_cnn = False
shuffle = True
epochs_model = 20

# file parms
delete_files = True

# use_item_emb = False
# min_len_session = None

# session embedding params
min_len_session = 2
max_len_session = 10
hidden_size_rnn = 150
model_embedding_size = 150
item2vec_embedding_size = 150
dense_layer_size = 20
item2vec_epoch = 1000
wipe_items_not_in_train = False

# val
validation_split = None
# validation_split = 0.1
# dates_for_val = ['2016-08-28','2016-08-29']

# test
dates_for_test = ['2016-08-30', '2016-08-31']
# dates_for_test = ['2016-08-29','2016-08-31']

# file params
dir_input = 'data'
x_train_path = "data_after_encode/x_train.npy"
y_train_path = "data_after_encode/y_train.npy"
x_val_path = "data_after_encode/x_val.npy"
y_val_path = "data_after_encode/y_val.npy"
x_test_path1 = "data_after_encode/x_test1.npy"
y_test_path1 = "data_after_encode/y_test1.npy"
x_test_path2 = "data_after_encode/x_test2.npy"
y_test_path2 = "data_after_encode/y_test2.npy"

model_path = "models/models.h5"
best_model_path = "models/best_model.h5"

if debug:
    if delete_files:
        system_utils.delete_file_no_exp(x_train_path)
        system_utils.delete_file_no_exp(y_train_path)
        system_utils.delete_file_no_exp(x_test_path1)
        system_utils.delete_file_no_exp(y_test_path1)
        system_utils.delete_file_no_exp(x_test_path2)
        system_utils.delete_file_no_exp(y_test_path2)
        system_utils.delete_file_no_exp(x_val_path)
        system_utils.delete_file_no_exp(y_val_path)
        system_utils.delete_file_no_exp(model_path)
        system_utils.delete_file_no_exp(best_model_path)
    hidden_size_rnn = 3
    item2vec_epoch = 5
    epochs_model = 1
    model_embedding_size = 5
    item2vec_embedding_size = 5
    dense_layer_size = 5
    item2vec_epoch = 1
