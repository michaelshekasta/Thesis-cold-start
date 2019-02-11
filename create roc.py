import math

import numpy as np
from sklearn import metrics

np.random.seed(10)
import pandas as pd

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve

train1 = pd.read_csv(
    "C:\\Users\\Michael\\Documents\\study\\Thesis\\4 - results\\cold_start\\deep analysis\\30\\comapre.csv")
# y_pred3 = train1['y_pred13'].values # y_pred12,y_pred7,y_test
y_pred15 = train1['y_pred15'].values  # y_pred12,y_pred7,y_test
y_pred16 = train1['y_pred16'].values  # y_pred12,y_pred7,y_test
real_y = train1['y_test'].values
# fpr_grd0, tpr_grd0, _ = roc_curve(real_y, y_pred3)
fpr_grd1, tpr_grd1, _ = roc_curve(real_y, y_pred15)
fpr_grd2, tpr_grd2, _ = roc_curve(real_y, y_pred16)

# using ntxt

# print metrics.roc_auc_score(real_y, hmm)
# print metrics.roc_auc_score(real_y, y_pred3)
print
metrics.roc_auc_score(real_y, y_pred15)
print
metrics.roc_auc_score(real_y, y_pred16)

plt.figure(1)
plt.plot([0, 1], [0, 1], 'k--')
# plt.plot(fpr_grd1, tpr_grd1, label='hmm')
# plt.plot(fpr_grd0, tpr_grd0, label='3')
plt.plot(fpr_grd1, tpr_grd1, label='15')
plt.plot(fpr_grd2, tpr_grd2, label='16')
plt.xlabel('False positive rate')
plt.ylabel('True positive rate')
plt.title('ROC curve')
plt.legend(loc='best')
plt.show()

# write_sol_file()
# write_file_to_roc()
# eval()
