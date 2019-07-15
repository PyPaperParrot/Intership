import numpy as np


from sklearn.naive_bayes import GaussianNB
from sklearn.linear_model import LogisticRegression
from sklearn.svm import SVC
from sklearn.ensemble import RandomForestClassifier
from sklearn.neighbors import KNeighborsClassifier


from keras.models import Sequential
from keras.layers import Dense, Dropout


from analisys import Quantile, NN_Quantile


clf_array = [
    GaussianNB(), LogisticRegression(),
    RandomForestClassifier(),
    KNeighborsClassifier(), SVC()
]

clf_names = [
    'GaussianNB', 'LogisticRegression',
    'RandomForestClassifier',
    'KNeighborsClassifier', 'SVC'
]


batch_size = 10
epochs = 200


def my_model():
    model = Sequential()
    model.add(Dense(50, input_dim=128, activation='relu'))
    model.add(Dropout(0.2))
    model.add(Dense(3, activation='sigmoid'))
    model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
    return model


for i in range(len(clf_array)):
    quantiles = Quantile(clf_array[i], 'Data/Datasets/sigma_')
    np.savetxt('quantiles_%s.txt' % clf_names[i], quantiles)


quantiles = NN_Quantile(my_model, batch_size, epochs, 'Data/Datasets/sigma_')
np.savetxt('quantiles_MLP.txt', quantiles)
