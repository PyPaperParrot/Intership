import numpy as np

from analisys import Q_SNR_plot


SNR_borders = (-30, 5)
n_steps = 15

clf_names = [
    'GaussianNB', 'LogisticRegression',
    'RandomForestClassifier',
    'KNeighborsClassifier', 'SVC', 'MLP'
]

SNR = np.linspace(SNR_borders[0], SNR_borders[1], n_steps)
SNR = np.flip(SNR)

for clf in clf_names:
    Q_SNR_plot('quantiles_%s.txt' % clf, SNR, clf)
