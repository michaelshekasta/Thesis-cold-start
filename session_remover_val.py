"""
This class wil represent exp for cold start issue
"""
import random


class SessionsRemoverVal(object):
    def __init__(self, catalog, train, test, data_file_path='data_before_encode', percent_remove=0.2,
                 by_dist_class=False):
        self.catalog = catalog
        self.train = train
        catalog_df = catalog.catalog_df
        categories = catalog_df[u'categorie'].unique()
        self.items_to_del = set()
        random_generator = random.Random(0)
        self.train_new = train.reindex()

        if by_dist_class:
            # df_merge.loc[df_merge['sessionid'] <= last_train_session]
            train_df0 = self.train_new.loc[self.train_new[u'buy'] == 0]
            train_df1 = self.train_new.loc[self.train_new[u'buy'] != 0]
            # train_remove_session0 = random_generator.sample(range(0, len(self.train_new)), int(len(self.train_new) * percent_remove))
            train_remove_session0 = random_generator.sample(train_df0.sessionid.values,
                                                            int(len(train_df0) * percent_remove))
            train_remove_session1 = random_generator.sample(train_df1.sessionid.values,
                                                            int(len(train_df1) * percent_remove))
            train_remove_sessions = train_remove_session0 + train_remove_session1
            # self.train_new = self.train_new.drop(self.train_new.index[train_remove_sessions])
            self.train_new.loc[~self.train_new.sessionid.isin(train_remove_sessions)]
        else:
            train_remove_sessions = random_generator.sample(self.train_new.sessionid.values,
                                                            int(len(self.train_new) * percent_remove))
            self.train_new = self.train_new.loc[~self.train_new.sessionid.isin(train_remove_sessions)]

        items_in_new_train = set()
        for index, row in self.train_new.iterrows():
            items_in_new_train = items_in_new_train.union(row.actions)

        self.non_new_item_test_set = test
        new_items_session_id = []
        for index, row in self.non_new_item_test_set.iterrows():
            items = set(row.actions)
            if len(items.difference(items_in_new_train)) > 0:
                new_items_session_id.append(row.sessionid)

        self.new_item_test_set = self.non_new_item_test_set.loc[self.non_new_item_test_set.sessionid.isin(
            new_items_session_id)]  # df_merge.loc[~df_merge['dayofsession'].isin(test_dates)]
        self.non_new_item_test_set = self.non_new_item_test_set.loc[~self.non_new_item_test_set.sessionid.isin(new_items_session_id)]
        # dump to file
        self.train_new.to_csv('%s/train_new.csv' % data_file_path, sep=';')
        self.non_new_item_test_set.to_csv('%s/non_new_item_test_set.csv' % data_file_path, sep=';')
        self.new_item_test_set.to_csv('%s/new_item_test_set.csv' % data_file_path, sep=';')

    def get_new_train(self):
        return self.train_new

    def get_non_new_item_test_set(self):
        return self.non_new_item_test_set

    def get_new_item_test_set(self):
        return self.new_item_test_set
