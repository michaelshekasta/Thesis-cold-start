#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io

import pandas as pd
from sklearn import metrics

# csv_file_path = "C:\\Users\\Michael\\Desktop\\results with category utf8.csv"
import system_utils

count = 0
with io.open("auc results_thesis_2.csv", 'w', encoding='utf8') as fw:
    fw.write(
        u"type,percent,category,length,new_items,percent new,total,purchase,notpurchase,baseline,textmodel,integration,gap12,gap13,gap23\n")
    for type in ['remove_items', 'remove_sessions']:
        for percent in range(1, 9):
            csv_file_path = "C:\\Users\\Michael\\Documents\\study\\Thesis\\4 - results\\final results\\%s_%d.csv" % (
                type, percent)
            df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf8')
            for category in [u'Angebote', u'Baumarkt', u'Drogerie & Gesundheit', u'Garten & Grillen',
                             u'Haushalt & Küche', u'Hobby & Freizeit', u'Kinderwelt', u'Lebensmittel',
                             u'Möbel & Einrichtung', u'Mode', u'Multimedia & Technik', u'Schönheit & Pflege',
                             u'Sport', u'all']:  # * is all
                if category == u'all':
                    cat_cond = (df[u'category'] <> u'all')
                else:
                    cat_cond = (df[u'category'] == category)
                for length in range(2, 12):  # 11 is no filter
                    if length == 11:
                        length = 'all'
                        length_cond = (df[u'clicks'] > -1)
                    else:
                        length_cond = (df[u'clicks'] == length)
                    for new_items in range(12):  # 11 no filter
                        if new_items == 11:
                            new_items = 'all'
                            new_items_cond = (df[u'cold_start_items'] > -1)
                        else:
                            new_items_cond = (df[u'cold_start_items'] == new_items)
                        if new_items <> u'all' and length <> u'all' and new_items > length:
                            continue
                        a = df.loc[(cat_cond) & (length_cond) & (new_items_cond)]
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
                        if new_items == 'all' or length == 'all':
                            items_length = 'NaN'
                        else:
                            items_length = float(new_items) / length
                        fw.write(u"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                            str(type), str(percent),
                            category, str(length),
                            str(new_items), str(items_length), str(number_sample), str(number_sample_with_purchase),
                            str(number_sample_no_purchase), str(auc_baseline), str(auc_textmodel),
                            str(auc_integrated), str(abs(auc_baseline - auc_textmodel)),
                            str(abs(auc_baseline - auc_integrated)),
                            str(abs(auc_textmodel - auc_integrated))))
                        count += 1
                        if count % 100 == 0:
                            print('count=%d' % count)
system_utils.send_email()
