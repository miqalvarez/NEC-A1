import numpy as np

# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers, activation_function, perc, dataset, epochs, learning_rate, momentum):
    self.L = len(layers)                  # number of layers
    self.n = layers.copy()                        # number of neurons in each layer
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
    self.n_epochs = epochs                # number of epochs
    self.learning_rate = learning_rate    # learning rate
    self.momentum = momentum              # momentum
    self.dataset = dataset                # dataset
    self.fact = activation_function       # activation function
    
    # Initialize all weights and thresholds randomly
    for lay in range(1, self.L):
      self.w[lay] = np.random.rand(self.n[lay], self.n[lay - 1]) - 0.5
      self.theta[lay] = np.random.rand(self.n[lay]) - 0.5
    
    if self.perc != 0:
      self.training_set = int(self.perc * len(self.dataset))
          
      self.train_set = self.dataset[:self.training_set]
      self.valid_set = self.dataset[self.training_set:]

    else:
      self.training_set = len(self.dataset)

    def fit(self, X, y):
      for epoch in range(1, self.n_epochs):
        for pat in range(self.training_set):
          # Choose a random pattern xµ from the training set
          xu = np.random.randint(0, self.training_set)

          feed_forward(X[xu])
          back_propagation()
          update_weights()

        # todo: Feed−forward all training patterns and calculate their prediction quadratic error
        # todo: Feed−forward all validation patterns and calculate their prediction quadratic error

    def feed_forward(self, X):
      # Feed−forward propagation of pattern xµ to obtain the output o(xµ)
      for neuron in range(self.n[0]):
        self.xi[0][neuron] = X[neuron]

      for lay in range(1, self.L):
        for neuron in range(self.n[lay]):
          for j in range(self.n[lay - 1]):
            self.h[lay][neuron] += self.w[lay][neuron][j] * self.xi[lay - 1][j]

          self.h[lay][neuron] -= self.theta[lay][neuron]
          self.xi[lay][neuron] = self.fact(self.h[lay][neuron])

    def back_propagation(self, y):
      # todo: Back−propagation of the error to obtain the delta values
      pass
      
    def update_weights(self):
      # todo: Update the weights and thresholds
      pass

    def predict(self, X):
      pass

    def loss_epochs(self):
      pass

# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]

# list of activation functions coded as lambda functions
fact = [
    lambda x: 1 / (1 + pow(2.71828, -x)),   # Sigmoid
    lambda x: max(0, x),                    # ReLU
    lambda x: x,                            # Linear
    lambda x: (pow(2.71828, x) - pow(2.71828, -x)) / (pow(2.71828, x) + pow(2.71828, -x))  # Tanh
]

# todo: add read parameters from args

nn = MyNeuralNetwork()  

# todo: execute part

# Check if the neural network is correctly initialized
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

