#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io

import pandas as pd
from sklearn import metrics

# csv_file_path = "C:\\Users\\Michael\\Desktop\\results with category utf8.csv"

with io.open("auc results length.csv", 'w', encoding='utf8') as fw:
    fw.write(
        u"type,percent,length,sign,avg new items,total,purchase,notpurchase,auc baseline,auc textmodel,auc integration\n")
    for type in ['remove_items', 'remove_sessions']:
        for percent in range(1, 9):
            csv_file_path = "C:\\Users\\Michael\\Documents\\study\\Thesis\\4 - results\\final results\\%s_%d.csv" % (
                type, percent)
            df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf8')
            for length in range(1, 11):
                for gt_sign in [True, False]:
                    if gt_sign:
                        length_cond = (df[u'clicks'] >= length)
                    else:
                        length_cond = (df[u'clicks'] <= length)
                    a = df.loc[(length_cond)]
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
                    if gt_sign:
                        sign ='>'
                    else:
                        sign = '<'
                    fw.write(u"%s,%s,%s,%s,%s,%s,%s,%s,%s,%s,%s\n" % (
                        str(type),str(percent),
                        str(length), str(sign), str(a[u'cold_start_items'].mean()),
                        str(number_sample),str(number_sample_no_purchase),str(number_sample_with_purchase),
                        str(auc_baseline), str(auc_textmodel),str(auc_integrated)))
