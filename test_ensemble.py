from sklearn.metrics import roc_curve
from sklearn.metrics import auc
colors = ['black', 'orange', 'blue', 'green']
linestyles = [':', '--', '-.', '-' ]
for clf, label, clr, ls in zip(all_clf, clf_labels, colors, linestyles):