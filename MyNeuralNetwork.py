import numpy as np

# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers, activation_function, perc, dataset, units, epochs, learning_rate, momentum):
    self.L = len(layers)                  # number of layers
    self.n = units                        # number of neurons in each layer
    self.h = []                           # field values.
    for lay in range(self.L):
      self.h.append(np.zeros(layers[lay]))

    self.theta = []                       # threshold values
    for lay in range(1, self.L):
      self.theta.append(np.zeros(layers[lay]))

    self.delta = []                       # delta values
    for lay in range(self.L):
      self.delta.append(np.zeros(layers[lay]))
    
    self.d_w = []                         # delta weights
    self.d_w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.d_w.append(np.zeros((layers[lay], layers[lay - 1])))

    self.d_theta = []                     # delta thresholds
    for lay in range(1, self.L):
      self.d_theta.append(np.zeros(layers[lay]))

    self.d_w_prev = []                    # previous delta weights
    self.d_w_prev.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.d_w_prev.append(np.zeros((layers[lay], layers[lay - 1])))


    self.d_theta_prev = []                # previous delta thresholds
    for lay in range(1, self.L):
      self.d_theta_prev.append(np.zeros(layers[lay]))
    
    self.xi = []                          # node values
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = []                           # edge weights
    self.w.append(np.zeros((1, 1)))
    for lay in range(1, self.L):
      self.w.append(np.zeros((layers[lay], layers[lay - 1])))

    self.perc = perc                      # percentage of training data
    self.epochs = epochs                  # number of epochs
    self.learning_rate = learning_rate    # learning rate
    self.momentum = momentum              # momentum
    self.dataset = dataset                # dataset
    self.fact = activation_function       # activation function

    def fit(self, X, y):
      print("fit")

    def predict(self, X):
      print("predict")

    def loss_epochs(self):
      print("loss_epochs")

# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]

# fact is a list of activation functions coded as lambda functions
fact = []
fact[0] = lambda x: 1 / (1 + np.exp(-x))
fact[1] = lambda x: np.maximum(0, x)
fact[2] = lambda x: np.tanh(x)

nn = MyNeuralNetwork(layers, fact[0])

print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("wh = ", nn.w, end="\n")
print("wh[1] = ", nn.w[1], end="\n")

print("theta = ", nn.theta, end="\n")
print("theta[1] = ", nn.theta[1], end="\n")

print("h = ", nn.h, end="\n")
print("h[1] = ", nn.h[1], end="\n")

print("delta = ", nn.delta, end="\n")
print("delta[1] = ", nn.delta[1], end="\n")

print("d_w = ", nn.d_w, end="\n")
print("d_w[1] = ", nn.d_w[1], end="\n")

print("d_theta = ", nn.d_theta, end="\n")
print("d_theta[1] = ", nn.d_theta[1], end="\n")

print("d_w_prev = ", nn.d_w_prev, end="\n")
print("d_w_prev[1] = ", nn.d_w_prev[1], end="\n")

print("d_theta_prev = ", nn.d_theta_prev, end="\n")
print("d_theta_prev[1] = ", nn.d_theta_prev[1], end="\n")

print("fact = ", nn.fact, end="\n")

