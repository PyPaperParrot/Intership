import numpy as np
from sklearn.preprocessing import MinMaxScaler
import h5py

from chirpGen import signal, nonChirpSignal, noising


def trainDataGen(frequency_list, C, size=1200, T=128, amplitude=1):
    '''amount of objects for each class'''
    N = int(size / (len(frequency_list) + 1))
    data = []

    for V in frequency_list:
#        data.append(np.array([signal(V, T, amplitude, np.random.uniform(0, 2 * np.pi)) for i in range(N)]))
        data.append(np.array([signal(V, T, amplitude) for i in range(N)]))
# Non-chirp data generator
    if C == 'random':
        obj = np.array([nonChirpSignal(T) for i in range(N)])
        norm = np.linalg.norm(obj)
        obj = obj / norm
        data.append(obj)

    data = np.array(data)
    data = np.reshape(data, (size, T))
    target = np.concatenate(tuple(i * np.ones(N, dtype=int) for i in range((len(frequency_list) + 1))))

    return data, target


def testDataGen(data, sigma):
    noisy_data = np.array([noising(d, sigma) for d in data])
    scaler = MinMaxScaler(feature_range=(-1, 1))
    scaler.fit(noisy_data)
    noisy_data = scaler.transform(noisy_data)

    return noisy_data


# 'size' must be tuple(train_size, test_size)
def DatasetGen(frequency_list, C, size, T, amplitude, sigma, filename, path=None):
    trainData, trainTarget = trainDataGen(frequency_list, C, size[0], T, amplitude)
    testData, testTarget = trainDataGen(frequency_list, C, size[1], T, amplitude)
    testData = testDataGen(testData, sigma)

    with h5py.File('%s/data_%s.hdf5' % (path, filename), 'w') as f:
        f.create_dataset('trainSignal', data=trainData)
        f.create_dataset('trainTarget_class', data=trainTarget)
        f.create_dataset('testSignal', data=testData)
        f.create_dataset('testTarget_class', data=testTarget)


def variance(SNR_borders, n_steps):
    SNR = np.linspace(SNR_borders[0], SNR_borders[1], n_steps)
    SNR = np.flip(SNR)
    variance_array = np.array([10 ** (-snr / 10) for snr in SNR])
    return variance_array

#Test
#DatasetGen(freq, 'random', (1080, 120), T, amplitude, 1, 1, 'Datasets/')

