import os
from badDataGen import DatasetGen, variance


V1 = 0.0001
V2 = 0.0001* 5
freq = (V1, V2)
N = 1200
amplitude = 1
T = 128
SNR_borders = (-30, 5)
n_steps = 15

path = os.getcwd()


variance = variance(SNR_borders, n_steps)

for i in range(len(variance)):
    dir_name = 'sigma_%i' % i
    current_path = path + '/' + dir_name
    os.mkdir(current_path)
    for j in range(30):
        DatasetGen(freq, 'random', (1080, 120), T, amplitude, variance[i], j, dir_name)

