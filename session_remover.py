"""
This class wil represent exp for cold start issue
"""
import random
import pandas as pd


class SessionsRemover(object):
    def __init__(self, catalog, train, test, data_out_path='data_before_encode', percent_remove=0.2,
                 by_dist_class=False):
        self.catalog = catalog
        self.train = train
        self.test = test
        catalog_df = catalog.catalog_df
        # categories = catalog_df[u'categorie'].unique()
        self.items_to_del = set()
        random_generator = random.Random(0)
        # self.train_new = train.reindex()

        items_in_train = self.get_items_from_df(self.train)

        if by_dist_class:
            # df_merge.loc[df_merge['sessionid'] <= last_train_session]
            buy_col_name = u'buy'
            train_df0 = self.train.loc[self.new_train[buy_col_name] == 0]
            train_df1 = self.train.loc[self.new_train[buy_col_name] != 0]
            # train_remove_session0 = random_generator.sample(range(0, len(self.train_new)), int(len(self.train_new) * percent_remove))
            train_remove_session0 = random_generator.sample(train_df0.sessionid.values,
                                                            int(len(train_df0) * percent_remove))
            train_remove_session1 = random_generator.sample(train_df1.sessionid.values,
                                                            int(len(train_df1) * percent_remove))
            train_remove_sessions = train_remove_session0 + train_remove_session1
            # self.train_new = self.train_new.drop(self.train_new.index[train_remove_sessions])
            self.new_train.loc[~self.new_train.sessionid.isin(train_remove_sessions)]
        else:
            train_remove_sessions = random_generator.sample(list(self.train.sessionid.values),
                                                            int(len(self.train) * percent_remove))
            self.new_train = self.train.loc[~self.train.sessionid.isin(train_remove_sessions)]

        items_in_new_train = self.get_items_from_df(self.new_train)

        self.item_to_del = items_in_train.difference(items_in_new_train)

        new_items_session_id = []
        for index, row in self.test.iterrows():
            items = set(row.actions)
            if len(items.difference(items_in_new_train)) > 0:
                new_items_session_id.append(row.sessionid)

        self.new_item_test_set = self.test.loc[self.test.sessionid.isin(
            new_items_session_id)]  # df_merge.loc[~df_merge['dayofsession'].isin(test_dates)]
        self.non_new_item_test_set = self.test.loc[
            ~self.test.sessionid.isin(new_items_session_id)]
        # dump to file
        self.new_train.to_csv('%s/train_new.csv' % data_out_path)
        self.non_new_item_test_set.to_csv('%s/non_new_item_test_set.csv' % data_out_path)
        self.new_item_test_set.to_csv('%s/new_item_test_set.csv' % data_out_path)

        # items_to_csv = self.item_to_del
        items_to_csv = []
        # for long_item in self.item_to_del:
        #     items_to_csv.append(str(long_item))
        # df_items = pd.DataFrame.from_records(data=list(items_to_csv), columns=['items'])
        # self.item_to_del
        # df_items = pd.DataFrame.from_dict({'items:', items_to_csv})
        df_items = pd.DataFrame.from_dict({'items': list(self.item_to_del)})
        df_items.to_csv('%s/items_removed.csv' % data_out_path)

    def get_items_from_df(self, df):
        items_in_new_train = set()
        for index, row in df.iterrows():
            items_in_new_train = items_in_new_train.union(row.actions)
        return items_in_new_train

    def get_new_train(self):
        return self.new_train

    def get_non_new_item_test_set(self):
        return self.non_new_item_test_set

    def get_new_item_test_set(self):
        return self.new_item_test_set
