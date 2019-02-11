#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io

import pandas as pd
from sklearn import metrics

# csv_file_path = "C:\\Users\\Michael\\Desktop\\results with category utf8.csv"
with io.open("auc results.csv", 'w', encoding='utf8') as fw:
    fw.write(
        u"type,percent,onlyone,category,gt_length,length,gt_new_item,new_items,total,purchase,notpurchase,baseline,textmodel,integration,gap12,gap13,gap23\n")
    for type in ['remove_items', 'remove_sessions']:
        for percent in range(1, 9):
            csv_file_path = "C:\\Users\\Michael\\Documents\\study\\Thesis\\4 - results\\final results\\%s_%d.csv" % (
                type, percent)
            df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf8')
            for onlyone in [False, True]:
                for category in [u'Angebote', u'Baumarkt', u'Drogerie & Gesundheit', u'Garten & Grillen',
                                 u'Haushalt & Küche', u'Hobby & Freizeit', u'Kinderwelt', u'Lebensmittel',
                                 u'Möbel & Einrichtung', u'Mode', u'Multimedia & Technik', u'Schönheit & Pflege',
                                 u'Sport', u'all']:  # * is all
                    if category == u'all':
                        cat_cond = (df[u'category'] <> u'all')
                    else:
                        if onlyone:
                            cat_cond = (df[u'category'] == category)
                        else:
                            cat_cond = (df[u'category'] != category)
                    for gt_length in [False, True]:
                        for length in range(12):  # 11 is no filter
                            if length == 11:
                                cat_length = (df[u'clicks'] >= 0)
                            else:
                                if gt_length:
                                    length_cond = (df[u'clicks'] > length)
                                else:
                                    length_cond = (df[u'clicks'] < length)
                            for gt_new_items in [False, True]:
                                for new_items in range(12):  # 11 no filter
                                    if new_items == 11:
                                        new_items_cond = (df[u'cold_start_items'] >= 0)
                                    else:
                                        if gt_new_items:
                                            new_items_cond = df[u'cold_start_items'] > new_items
                                        else:
                                            new_items_cond = df[u'cold_start_items'] < new_items
                                        a = df.loc[(cat_cond) & (length_cond) & (new_items_cond)]
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

                                        fw.write(u"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                                            str(type), str(percent),
                                            str(onlyone), category, str(gt_length), str(length), str(gt_new_items),
                                            str(new_items), str(number_sample), str(number_sample_with_purchase),
                                            str(number_sample_no_purchase), str(auc_baseline), str(auc_textmodel),
                                            str(auc_integrated), str(abs(auc_baseline - auc_textmodel)),
                                            str(abs(auc_baseline - auc_integrated)),
                                            str(abs(auc_textmodel - auc_integrated))))