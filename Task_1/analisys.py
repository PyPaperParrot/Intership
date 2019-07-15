import numpy as np
import h5py
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.utils import shuffle
from sklearn.metrics import f1_score


from keras.utils import np_utils


def read_Data(file_path):
    data = h5py.File(file_path, 'r')
    X_train = data['trainSignal'][:]
    y_train = data['trainTarget_class'][:]
    X_test = data['testSignal'][:]
    y_test = data['testTarget_class'][:]
    data.close()

    X_train, y_train = shuffle(X_train, y_train, random_state=0)
    X_test, y_test = shuffle(X_test, y_test, random_state=0)

    return X_train, X_test, y_train, y_test


def Classifier(classifier, train_data, train_target, test_data, test_target):
    classifier.fit(train_data, train_target)
    prediction = classifier.predict(test_data)
    f1 = f1_score(test_target, prediction, average='micro')

    return f1


def Quantile(classifier, dir_path):
    IQR_array = []
    for i in range(15):
        f1 = []
        for j in range(30):
            path = dir_path + '%s/data_%s.hdf5' % (i, j)
            X_train, X_test, y_train, y_test = read_Data(path)
            f1.append(Classifier(classifier, X_train, y_train, X_test, y_test))
        IQR_array.append(f1)

    return IQR_array


def MLP(model, X_train, y_train, X_test, y_test, batch_size, epochs):
    clf = model()
    y_train = np_utils.to_categorical(y_train, 3)
    clf.fit(X_train, y_train, batch_size=batch_size, epochs=epochs, verbose=1)
    predictions = clf.predict_classes(X_test)
    f1 = f1_score(y_test, predictions, average='micro')
    return f1


def NN_Quantile(model, batch_size, epochs, dir_path):
    IQR_array = []
    for i in range(15):
        f1 = []
        for j in range(30):
            path = dir_path + '%s/data_%s.hdf5' % (i, j)
            X_train, X_test, y_train, y_test = read_Data(path)
            f1.append(MLP(model, X_train, y_train, X_test, y_test, batch_size, epochs))
        IQR_array.append(f1)

    return IQR_array


def Q_SNR_plot(quantiles_path, SNR, name):
    quantiles = np.loadtxt(quantiles_path)
    quantiles = quantiles.transpose()
    df = pd.DataFrame(quantiles, columns=SNR)
    ax = df.plot.box(title=name)
    ax.set_xlabel('SNR (дБ)')
    ax.set_ylabel('f1_metric')
    plt.show()
