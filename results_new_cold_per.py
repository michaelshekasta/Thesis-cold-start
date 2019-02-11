#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io

import pandas as pd
from sklearn import metrics

# csv_file_path = "C:\\Users\\Michael\\Desktop\\results with category utf8.csv"

with io.open("auc results new items per.csv", 'w', encoding='utf8') as fw:
    fw.write(
        u"type,percent,length,avg new items,total,purchase,notpurchase,auc baseline,auc textmodel,auc integration\n")
    # for type in ['remove_items', 'remove_sessions']:
    for type in ['remove_items']:
        for percent in range(1, 9):
            csv_file_path = "C:\\Users\\Michael\\Documents\\study\\Thesis\\4 - results\\final results\\%s_%d.csv" % (
                type, percent)
            df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf8')
            df[u'new_item_per'] = (df[u'cold_start_items'] * 1.0) / df[u'clicks']
            # for per_new_items in [0, 0.1, 0.111111111, 0.125, 0.142857143, 0.166666667, 0.2, 0.222222222, 0.25, 0.285714286, 0.3, 0.333333333, 0.375, 0.4, 0.428571429, 0.444444444, 0.5, 0.555555556, 0.571428571, 0.6, 0.625, 0.666666667, 0.7, 0.714285714, 0.75, 0.777777778, 0.8, 0.833333333, 0.857142857, 0.875, 0.888888889, 0.9, 1]:
            for per_new_items in [0.5]:
                percent_cond = (df[u'new_item_per'] >= per_new_items)
                a = df.loc[(percent_cond)]
                if a.empty:
                    continue
                y_test = a.y_test.values
                y_baseline = a.y_baseline.values
                y_textmodel = a.y_textmodel.values
                y_integration = a.y_integrated.values

                number_sample = a.shape[0]
                number_sample_no_purchase = a.loc[df[u'buy'] == 0].shape[0]
                number_sample_with_purchase = a.loc[df[u'buy'] > 0].shape[0]
                try:
                    auc_baseline = metrics.roc_auc_score(y_test, y_baseline)
                except:
                    auc_baseline = -1
                try:
                    auc_textmodel = metrics.roc_auc_score(y_test, y_textmodel)
                except:
                    auc_textmodel = -1
                try:
                    auc_integrated = metrics.roc_auc_score(y_test, y_integration)
                except:
                    auc_integrated = -1
                fw.write(u"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                    str(type), str(percent),
                    str(per_new_items), str(a[u'clicks'].mean()),
                    str(number_sample), str(number_sample_no_purchase), str(number_sample_with_purchase),
                    str(auc_baseline), str(auc_textmodel), str(auc_integrated)))
