# import dask.dataframe as pd
import pandas as pd


def buy_item_to_item(item):
    if str(item)[0] == '2':
        item = int('1%s' % str(item)[1:])
    return item


def read_session_actions(input_path, items_list, maxlen, minlen=None):
    data = []
    i=0
    with open(input_path) as f_reader:
        for line in f_reader:
            i+=1
            line = line.replace("\n", "").replace("\r", "")
            split_fields = line.split(" ")
            session_id = int(split_fields[0])
            actions = split_fields[1:]
            actions = [action for action in actions if str(action)[0] == '1']
            to_insert = True
            if not items_list is None:
                for item in actions:
                    # if not long(item) in items_list:
                    if not int(item) in items_list:
                        to_insert = False
                        break
            if not minlen is None and len(actions) < minlen:
                to_insert = False
            if to_insert:
                if not maxlen is None:
                    actions = actions[-maxlen:]
                data.append((session_id, actions))
    return pd.DataFrame(data, columns=['sessionid', 'actions'])


def read_session_info(input_path):
    return pd.read_csv(input_path, delimiter=',')


class SessionReader(object):
    def __init__(self, input_path_session_actions, input_path_session_info, maxlen=None, minlen=None, test_percent=-1.0,
                 test_dates=None, encode_dir='data_before_encode',
                 items_list=None, wipe_items_not_in_train=False):
        self.items = items_list
        self.train = None
        self.test = None
        self.encode_dir = encode_dir
        temp_session_id = read_session_info(input_path_session_info)
        print('done reading info')
        print('reading actions..')
        temp_action = read_session_actions(input_path_session_actions, items_list=items_list, maxlen=maxlen,
                                           minlen=minlen)
        print('done reading actions')
        df_merge = pd.merge(left=temp_session_id, right=temp_action, on=['sessionid'])
        print('done merging..')
        sessionids = df_merge.sessionid.values
        # if test_percent > -1:
        #     last_train_session = sessionids[int(len(sessionids) * test_percent)]
        #     self.train = df_merge.loc[df_merge['sessionid'] <= last_train_session]
        #     self.test = df_merge.loc[df_merge['sessionid'] > last_train_session]
        # else:
        self.train = df_merge.loc[~df_merge['dayofsession'].isin(test_dates)]
        self.train.to_csv('%s/train.csv' % self.encode_dir, index_label='index')
        self.test = df_merge.loc[df_merge['dayofsession'].isin(test_dates)]
        self.test.to_csv('%s/test.csv' % self.encode_dir, index_label='index')
        self.test.reset_index(drop=True).to_csv('%s/test.csv' % self.encode_dir, index_label='index')
        self.new_test_only_new_items = None
        self.items_in_train = set()
        for index, row in self.train.iterrows():
            self.items_in_train = self.items_in_train.union(row.actions)
        if wipe_items_not_in_train:
            print(self.items_in_train)
            new_items_session_id = []
            for index, row in self.test.iterrows():
                items = set(row.actions)
                if len(items.difference(self.items_in_train)) > 0:
                    new_items_session_id.append(row.sessionid)

                self.new_test_only_new_items = self.test.loc[self.test.sessionid.isin(
                    new_items_session_id)]  # df_merge.loc[~df_merge['dayofsession'].isin(test_dates)]
                self.test = self.test.loc[~self.test.sessionid.isin(new_items_session_id)]

        print('training set = %s' % str(self.train.shape[0]))
        print('test set = %s' % str(self.test.shape[0]))
        if wipe_items_not_in_train:
            print('out of test set (cold start) = %s' % str(self.new_test_only_new_items.shape[0]))
        print('done divide train test..')

    def get_train(self):
        return self.train

    def get_test(self):
        return self.test

    def get_items(self):
        return self.items

    def set_train(self, train):
        self.train = train

    def set_test(self, test):
        self.test = test
