#!/usr/bin/env python
# -*- coding: utf-8 -*-
from sklearn import metrics
from sklearn.metrics import roc_curve
import matplotlib.pyplot as plt
import pandas as pd

# csv_file_path = "C:\\Users\\Michael\\Documents\\study\\Thesis\\4 - results\\final results\\remove_items_4.csv"
csv_file_path = "C:\\Users\\Michael\\Documents\\study\\Thesis\\4 - results\\final results\\regular exp\\results-q1.csv"
# csv_file_path = "C:\\Users\\Michael\\Desktop\\results with category utf8.csv"
df = pd.read_csv(csv_file_path, delimiter=',', encoding='utf8')
a = df
# a = df.loc[(df['clicks'] < 3) & (df[u'cold_start_items'] > 1)]
# a = df.loc[(df[u'category'] == u'Baumarkt')]
# a = df.loc[(df[u'cold_start_items'] > 1) & (df['clicks'] < 5) & (df[u'category'] == u'Baumarkt')]
# a = df.loc[(df[u'cold_start_items'] > 1) & (df['clicks'] < 5)]
# a = df
#
y_test = a.y_test.values
y_baseline = a.y_baseline.values
y_textmodel = a.y_textmodel.values
y_integration = a.y_integrated.values

number_sample = a.shape[0]
print("count samples = %d" % number_sample)
number_sample_no_purchase = a.loc[df[u'buy'] == 0].shape[0]
print("count samples end without purchase = %d" % number_sample_no_purchase)
number_sample_with_purchase = a.loc[df[u'buy'] > 0].shape[0]
print("count samples end with purchase = %d" % number_sample_with_purchase)
auc_baseline = metrics.roc_auc_score(y_test, y_baseline)
print("y_baseline auc = %s" % str(auc_baseline))
auc_textmodel = metrics.roc_auc_score(y_test, y_textmodel)
print("y_textmodel auc = %s" % str(auc_textmodel))
auc_integrated = metrics.roc_auc_score(y_test, y_integration)
print("y_integration auc = %s" % str(auc_integrated))

fpr_grd1, tpr_grd1, _ = roc_curve(y_test, y_baseline)
fpr_grd2, tpr_grd2, _ = roc_curve(y_test, y_textmodel)
fpr_grd3, tpr_grd3, _ = roc_curve(y_test, y_integration)
plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
plt.plot(fpr_grd1, tpr_grd1, label='baseline (auc=%.4f)' % auc_baseline)
plt.plot(fpr_grd2, tpr_grd2, label='content-based model (auc=%.4f)' % auc_textmodel)
plt.plot(fpr_grd3, tpr_grd3, label='integration model (auc=%.4f)' % auc_integrated)
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.text(0.5, 0.25, "count samples=%d\nwithout purchase=%d\nwith purchase=%d" % (
    number_sample, number_sample_no_purchase, number_sample_with_purchase))
fig1 = plt.gcf()
plt.savefig("roc.png", dpi=100)
plt.show()
