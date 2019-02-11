#!/usr/bin/env python
# -*- coding: utf-8 -*-
import io

import pandas as pd
from sklearn import metrics

# csv_file_path = "C:\\Users\\Michael\\Desktop\\results with category utf8.csv"
with io.open("auc results category.csv", 'w', encoding='utf8') as fw:
    fw.write(
        u"category,auc baseline,auc textmodel,auc integration\n")
    csv_file_path = "C:\\Users\\Michael\\Downloads\\results-q1.csv"
    df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf8')
    for category in [u'Angebote', u'Baumarkt', u'Drogerie & Gesundheit', u'Garten & Grillen',
                     u'Haushalt & Küche', u'Hobby & Freizeit', u'Kinderwelt', u'Lebensmittel',
                     u'Möbel & Einrichtung', u'Mode', u'Multimedia & Technik', u'Schönheit & Pflege',
                     u'Sport', u'all']:  # * is all
        cat_cond = (df[u'category'] == category)
        a = df.loc[(cat_cond)]
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

        fw.write(u"%s,%s,%s,%s\n" % (
            category, str(auc_baseline), str(auc_textmodel),
            str(auc_integrated)))
