#!/usr/bin/env python
# -*- coding: utf-8 -*-

import pandas as pd


def replace_umlauts(text):
    res = text
    res = res.replace(u'ä', 'ae')
    res = res.replace(u'ö', 'oe')
    res = res.replace(u'ü', 'ue')
    res = res.replace(u'Ä', 'Ae')
    res = res.replace(u'Ö', 'Oe')
    res = res.replace(u'Ü', 'Ue')
    res = res.replace(u'ß', 'ss')
    return res


csv_file_path = "C:\\Users\\Michael\\Documents\\study\\Thesis\\4 - results\\final results\\regular exp\\results-q1.csv"
df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf8')
for category in [u'Angebote', u'Baumarkt', u'Drogerie & Gesundheit', u'Garten & Grillen',
                 u'Haushalt & Küche', u'Hobby & Freizeit', u'Kinderwelt', u'Lebensmittel',
                 u'Möbel & Einrichtung', u'Mode', u'Multimedia & Technik', u'Schönheit & Pflege',
                 u'Sport', u'all']:  # * is all
    cat_cond = (df[u'category'] == category)
    a = df.loc[(cat_cond)]
    a.to_csv("results category\\%s.csv" % replace_umlauts(category), sep=',', encoding='utf8')
