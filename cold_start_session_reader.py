"""
This class wil represent exp for cold start issue
"""
import random
import pandas as pd

class ColdStartSessionReader(object):
    def __init__(self, catalog, train, test, data_out_path='data_before_encode', min_item_in_category=5,
                 precent_remove=0.2):
        self.catalog = catalog
        self.train = train
        self.test = test
        categorie_col_name = u'categorie'

        catalog_df = catalog.catalog_df
        categories = catalog_df[categorie_col_name].unique()
        self.items_to_del = set()
        random_generator = random.Random(0)
        for category in categories:
            items_in_category = catalog_df.loc[catalog_df[categorie_col_name] == category].product_id.values.tolist()
            amount_items = len(items_in_category)
            if amount_items >= min_item_in_category:
                for i in range(int(precent_remove * amount_items)):
                    item_del = random_generator.choice(items_in_category)
                    items_in_category.remove(item_del)
                    self.items_to_del.add(str(item_del))

        new_items_session_id_train = []
        for index, row in self.train.iterrows():
            items = set(row.actions)
            if len(self.items_to_del.intersection(items)) > 0:
                new_items_session_id_train.append(row.sessionid)

        self.train_new = self.train.loc[~self.train.sessionid.isin(new_items_session_id_train)]

        new_items_session_id_test = []
        for index, row in self.test.iterrows():
            items = set(row.actions)
            if len(self.items_to_del.intersection(items)) > 0:
                new_items_session_id_test.append(row.sessionid)

        self.new_item_test_set = self.test.loc[self.test.sessionid.isin(
            new_items_session_id_test)]  # df_merge.loc[~df_merge['dayofsession'].isin(test_dates)]
        self.non_new_item_test_set = self.test.loc[~self.test.sessionid.isin(new_items_session_id_test)]
        # dump to file
        self.train_new.to_csv('%s/train_new.csv' % data_out_path)
        self.non_new_item_test_set.to_csv('%s/test_new.csv' % data_out_path)
        self.new_item_test_set.to_csv('%s/new_test_only_new_items.csv' % data_out_path)

        df_items = pd.DataFrame.from_dict({'items': list(self.items_to_del)})
        df_items.to_csv('%s/items_removed.csv' % data_out_path)

    def get_new_train(self):
        return self.train_new

    def get_non_new_item_test_set(self):
        return self.non_new_item_test_set

    def get_new_item_test_set(self):
        return self.new_item_test_set
