import numpy as np
import matplotlib.pyplot as plt

def sigmoid(x, derivative=False):
	sigm = 1. / (1. + np.exp(-x))
	if derivative:
		return sigm * (1. - sigm)
	else:
		return sigm

def squareLoss(y_pred, y_test):
	return np.mean(np.square(y_test - y_pred))


class NeuralNetwork:

	#data initialization
	def __init__(self, X, y):
		self.input = X
		self.weights1 = np.random.rand(self.input.shape[1], 4)
		self.weights2 = np.random.rand(4, 1)
		self.y = y
		self.output = np.zeros(y.shape)

	#feedforward weights
	def feedforward(self):
		self.layer1 = sigmoid(np.dot(self.input, self.weights1))
		self.layer2 = sigmoid(np.dot(self.layer1, self.weights2))

		return self.layer2

	#backpropagation
	def backprop(self):
		d_weights2 = np.dot(self.layer1.T, 2*(self.y - self.output) * sigmoid(self.output, derivative=True))
		d_weights1 = np.dot(self.input.T, (np.dot(2*(self.y - self.output) * sigmoid(self.output, derivative=True), self.weights2.T) * sigmoid(self.layer1, True)))

		self.weights1 += d_weights1
		self.weights2 += d_weights2 

	#train neural network
	def train(self, X, y):
		self.output = self.feedforward()
		self.backprop()

#feature matrix
X = np.array([[0, 0, 1], [0, 1, 1], [1, 0, 1], [1, 1, 1]], dtype=float)
#target
y = np.array([[0], [1], [1], [0]], dtype=float)

NN = NeuralNetwork(X, y)

#print(NN.feedforward())
loss = []
for i in range(2001):
	if i % 100 == 0:
		print('for' + str(i) + ' iteration')
		print('Input: \t' + str(X))
		print('Acturl Output: \t' + str(y))
		print('Predicted Output: \t' + str(NN.feedforward()))
		print('Loss: \t' + str(squareLoss(NN.feedforward(), y)))
		loss.append(squareLoss(NN.feedforward(), y))
		print('\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\\n')
	NN.train(X, y)
print('///////////////////////////////////////////////\n')
print('Predicted: \n', NN.feedforward(), '\nActual: \n', y)
plt.plot([i for i in range(0, 2001, 100)], loss)
plt.show()
