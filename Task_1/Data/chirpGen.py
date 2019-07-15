import numpy as np


def signal(freq, T=None, amplitude=1, omega=0):
    """Generator for signal with linear frequency.
    parameters:
        + V -- radian frequency of the sinusoid
        + T -- time
        + amplitude -- signal's amplitude
        + omega --
    """
    sig = np.array([amplitude * np.sin(2 * np.pi * freq * t * t + omega) for t in range(T)])
    return sig


def nonChirpSignal(T=None):
    return np.array([np.random.normal(0, 1) for t in range(T)])


def noising(signal=None, sigma=1):
    """
    Noising the given signal
    """
    nsig = signal + np.random.normal(0, sigma, len(signal))
    return nsig







'''

dataV1 = signal(0.001, 128, 1, np.random.uniform(0, 2*np.pi))
dataV2 = signal(0.001*1.5, 128, 1, np.random.uniform(0, 2*np.pi))

plt.plot(np.arange(128), dataV1)
plt.plot(np.arange(128), dataV2)
plt.show()

noisy_dataV1 = noising(dataV1, 1)
plt.plot(np.arange(128), noisy_dataV1)
plt.show()
'''


'''a = np.array([signal(0.001, 128, 1, np.random.uniform(0, 2*np.pi)) for i in range(10)])
for i in a:
	plt.plot(np.arange(128), i)
plt.show()
print(dataV1)'''