import numpy as np

# Neural Network class
class MyNeuralNetwork:
  def __init__(self, layers, activation_function, derivative_function, perc, dataset, epochs, learning_rate, momentum):
    self.L = len(layers)                  # number of layers
    self.n = layers.copy()                # number of neurons in each layer
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
    self.derivative = derivative_function
    
    # Initialize all weights and thresholds randomly
    for lay in range(1, self.L):
      self.w[lay] = np.random.rand(self.n[lay], self.n[lay - 1]) - 0.5
      self.theta[lay] = np.random.rand(self.n[lay]) - 0.5
    
    if self.perc != 0:
      self.n_train = int(self.perc * len(self.dataset))
          
      self.training_set = self.dataset[:self.n_train]
      self.training_set = self.dataset[self.n_train:]

    else:
      self.training_set = len(self.dataset)

    def fit(self, X, y):
      errors_train = []
      errors_valid = []
      for epoch in range(1, self.n_epochs):
        used = []
        for pat in range(self.n_train):
          # Choose a random pattern xµ from the training set
          i = np.random.randint(0, self.n_train)
          while i in used:
            i = np.random.randint(0, self.n_train)

          used.append(i)

          xu = self.training_set[i]

          __feed_forward(xu)
          __back_propagation(y[i])
          __update_weights()

        # Feed−forward all training patterns and calculate their prediction quadratic error
        for pat in self.training_set:
          __feed_forward(pat)

          # Calculate error_train
          errors_train[epoch] = 0
          for neuron in range(self.n[self.L - 1]):
            errors_train[epoch] += pow(self.xi[self.L - 1][neuron] - y[pat], 2)
        
        errors_train[epoch] /= 2

        # Feed−forward all validation patterns and calculate their prediction quadratic error
        for pat in self.validation_set:
          __feed_forward(pat)

          # Calculate error_valid
          errors_valid[epoch] = 0
          
          for neuron in range(self.n[self.L - 1]):
            errors_valid[epoch] += pow(self.xi[self.L - 1][neuron] - y[pat], 2)
          errors_valid[epoch] /= 2

    # Feed−forward propagation of pattern xµ to obtain the output o(xµ)
    def __feed_forward(self, xu):
      for neuron in range(self.n[0]):
        self.xi[0][neuron] = xu[neuron]

      for lay in range(1, self.L):
        for neuron in range(self.n[lay]):
          for j in range(self.n[lay - 1]):
            self.h[lay][neuron] += self.w[lay][neuron][j] * self.xi[lay - 1][j]

          self.h[lay][neuron] -= self.theta[lay][neuron]
          self.xi[lay][neuron] = self.fact(self.h[lay][neuron])
      
      return self.xi[self.L - 1]

    # Back−propagation of the error to obtain the delta values
    def __back_propagation(self, y):
      # Compute delta in the output layer
      for neuron in range(self.n[self.L - 1]):
        self.delta[self.L - 1][neuron] = (self.xi[self.L - 1][neuron] - y[neuron]) * self.derivative(self.h[self.L - 1][neuron])

      # Back-propagate delta to the rest of the network
      for lay in range(self.L - 2, 0, -1):
        for neuron in range(self.n[lay]):
          for j in range(self.n[lay + 1]):
            self.delta[lay][neuron] += self.w[lay + 1][j][neuron]
          
          self.delta[lay][neuron] *= self.derivative(self.h[lay][neuron])

    # Update the weights and thresholds
    def __update_weights(self):
      for lay in range(1, self.L):
        for neuron in range(self.n[lay]):
          for j in range(self.n[lay - 1]):
            self.d_w[lay][neuron][j] = -self.learning_rate * self.delta[lay][neuron] * self.xi[lay - 1][j] + self.momentum * self.d_w_prev[lay][neuron][j]
            self.w[lay][neuron][j] += self.d_w[lay][neuron][j]
            self.d_w_prev[lay][neuron][j] = self.d_w[lay][neuron][j]

          self.d_theta[lay][neuron] = self.learning_rate * self.delta[lay][neuron] + self.momentum * self.d_theta_prev[lay][neuron]
          self.theta[lay][neuron] -= self.d_theta[lay][neuron]
          self.d_theta_prev[lay][neuron] = self.d_theta[lay][neuron]

    def predict(self, X):
      predictions = []

      for pat in X:
        predictions.append(__feed_forward(pat))

      return predictions
      
    def loss_epochs(self):
      return self.errors_train, self.errors_valid
      

# layers include input layer + hidden layers + output layer
layers = [4, 9, 5, 1]

# list of activation functions coded as lambda functions
fact = [
    lambda x: 1 / (1 + pow(2.71828, -x)),   # Sigmoid
    lambda x: max(0, x),                    # ReLU
    lambda x: x,                            # Linear
    lambda x: (pow(2.71828, x) - pow(2.71828, -x)) / (pow(2.71828, x) + pow(2.71828, -x))  # Tanh
]

derivatives = [
    lambda x: 1 / (1 + pow(2.71828, -x)) * 1 / (1 + pow(2.71828, -x)),  # Derivative of Sigmoid
    lambda x: 1 if x > 0 else 0,                                        # Derivative of ReLU
    lambda x: 1,                                                        # Derivative of Linear
    lambda x: 1 - ((pow(2.71828, x) - pow(2.71828, -x)) / (pow(2.71828, x) + pow(2.71828, -x)))**2                      # Derivative of Tanh
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

