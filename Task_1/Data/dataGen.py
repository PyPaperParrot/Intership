import numpy as np
import luigi

from chirpGen import signal, noising

'''class GenerateSignalData(luigi.Task):
	N = luigi.IntParameter()
	cls_freq = luigi.Parameter()

	def requires(self):
		return []

	def output(self):
		return 

	def run(self):
		data = np.array([])
		for f in cls_freq:
			data = np.array([signal(f, T, amplitude, omega=np.random.uniform(0, 2*np.pi)) for i in range(int(self.N/3))]
			

class GenerateNoisedSignalData(luigi.Task):
	pass'''


a = np.concatenate(tuple(np.ones(5, dtype=int)*i for i in range(5)))

print(a)
'''
a = np.array([])
b = np.array([[4, 5, 6]])
c = np.array([a, b])
d = np.array([b, a])
e = np.array([c, d])
print(np.concatenate((a, b), axis=1))
'''