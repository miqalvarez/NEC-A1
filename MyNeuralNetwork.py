from matplotlib import pyplot as plt
import numpy as np

# Neural Network class
class MyNeuralNetwork:
  ### CONSTRUCTOR ###
  def __init__(self, layers, activation_function, derivative_function, perc, epochs, learning_rate, momentum):
    self.L = len(layers)                  # number of layers
    self.n = layers.copy()                # number of neurons in each layer
    
    self.h = []                           # field values.
    for lay in range(self.L):
      self.h.append(np.zeros(layers[lay]))

    self.theta = []                       # threshold values
    for lay in range(self.L):
      self.theta.append(np.zeros(layers[lay]))

    self.delta = []                       # delta values
    for lay in range(self.L):
      self.delta.append(np.zeros(layers[lay]))
    
    self.d_w = [np.zeros((1, 1))] + [np.zeros((layers[lay], layers[lay - 1])) for lay in range(1, self.L)]

    self.d_theta = []                     # delta thresholds
    for lay in range(self.L):
      self.d_theta.append(np.zeros(layers[lay]))

    self.d_w_prev = [np.zeros((1, 1))] + [np.zeros((layers[lay], layers[lay - 1])) for lay in range(1, self.L)]


    self.d_theta_prev = []                # previous delta thresholds
    for lay in range(self.L):
      self.d_theta_prev.append(np.zeros(layers[lay]))
    
    self.xi = []                          # node values
    for lay in range(self.L):
      self.xi.append(np.zeros(layers[lay]))

    self.w = [np.zeros((1, 1))] + [np.zeros((layers[lay], layers[lay - 1])) for lay in range(1, self.L)]


    self.perc = perc                      # percentage of training data
    self.n_epochs = epochs                # number of epochs
    self.learning_rate = learning_rate    # learning rate
    self.momentum = momentum              # momentum
    self.fact = activation_function       # activation function
    self.derivative = derivative_function # derivative of activation function
    self.errors_train = []
    self.errors_valid = []
    
    
    # Initialize weights and thresholds
    for lay in range(self.L):
        for neuron in range(self.n[lay]):
            # Si es la capa de entrada, no hay pesos para inicializar
            if lay > 0:
                for j in range(self.n[lay - 1]):
                    # Asegúrate de que las dimensiones de self.w sean correctas
                    self.w[lay][neuron][j] = np.random.uniform(-1, 1)
                    self.d_w_prev[lay][neuron][j] = 0

            # Solo inicializa los umbrales para las capas ocultas y de salida
            if lay > 0 and lay < self.L - 1:
                self.theta[lay][neuron] = np.random.uniform(-1, 1)
                self.d_theta_prev[lay][neuron] = 0

    
  
  ### FUNCTIONS ###
  # Train the neural network
  # X: training samples
  # y: target values
  def fit(self, X, y):
    self.errors_train = np.zeros((self.n_epochs, 2))
    self.errors_valid = np.zeros((self.n_epochs, 2))
    print("Training...")
    
    if self.perc != 0:
        n_train = int(self.perc * len(X))
        training_set = X[:n_train]
        y_train = y[:n_train]
        validation_set = X[n_train:]
        y_valid = y[n_train:]
    else:
        training_set = X
        y_train = y
        validation_set = None
        y_valid = None

    for epoch in range(0, self.n_epochs):
      # print("Epoch: ", epoch+1)
      used = []
      for pat in range(n_train):
        # Choose a random pattern xµ from the training set
        i = np.random.randint(0, n_train)
        while i in used:
          i = np.random.randint(0, n_train)

        used.append(i)

        xu = training_set[i]
        yu = y_train[i]

        self.__feed_forward(xu)
        #print("Prediction: ", self.xi[self.L - 1][:])
        #print("Target: ", yu)      
        self.__back_propagation(yu)
        self.__update_weights()
      
      # Feed−forward all training patterns and calculate their prediction quadratic error
      for index, pat in enumerate(training_set):
        self.__feed_forward(pat)
        
        # Calculate quadratic error_train
        self.errors_train[epoch][0] = epoch
        self.errors_train[epoch][1] = 0

        for neuron in range(self.n[self.L - 1]):
          self.errors_train[epoch][1] += pow(self.xi[self.L - 1][neuron] - y_train[index], 2)

      self.errors_train[epoch][1] /= 2

      # Feed−forward all validation patterns and calculate their prediction quadratic error
      if validation_set.all != None:
        for index, pat in enumerate(validation_set):
          self.__feed_forward(pat)
          
          # Calculate quadratic error_valid
          self.errors_valid[epoch][0] = epoch
          self.errors_valid[epoch][1] = 0

          for neuron in range(self.n[self.L - 1]):
            self.errors_valid[epoch][1] += pow(self.xi[self.L - 1][neuron] - y_valid[index], 2)

        self.errors_valid[epoch][1] /= 2

      else:
        self.errors_valid[epoch] = None

  # Feed−forward propagation of pattern xµ to obtain the output o(xµ)
  def __feed_forward(self, xu):
    for neuron in range(self.n[0]):
      self.xi[0][neuron] = xu[neuron]

    for lay in range(1, self.L):
      for neuron in range(self.n[lay]):
        self.h[lay][neuron] = 0
        for j in range(self.n[lay - 1]):
          self.h[lay][neuron] += self.w[lay][neuron][j] * self.xi[lay - 1][j]
        
        self.h[lay][neuron] -= self.theta[lay][neuron]     
        self.xi[lay][neuron] = self.fact(self.h[lay][neuron])
    
    return self.xi[self.L - 1]

  # Back−propagation of the error to obtain the delta values
  def __back_propagation(self, y):
    # Compute delta in the output layer
    for neuron in range(self.n[self.L - 1]):
      self.delta[self.L - 1][neuron] = self.derivative(self.h[self.L - 1][neuron]) * (self.xi[self.L - 1][neuron] - y)

    # Back-propagate delta to the rest of the network
    for lay in range(self.L - 2, 0, -1):
      for neuron in range(self.n[lay]):
        self.delta[lay][neuron] = 0
        for j in range(self.n[lay + 1]):
          self.delta[lay][neuron] += self.delta[lay + 1][j] * self.w[lay + 1][j][neuron]
          self.delta[lay][neuron] *= self.derivative(self.h[lay][neuron])

  # Update the weights and thresholds
  def __update_weights(self):
    for lay in range(self.L - 1, 0, -1):
      for neuron in range(self.n[lay]):
        for j in range(self.n[lay - 1]):
          self.d_w[lay][neuron][j] = -self.learning_rate * self.delta[lay][neuron] * self.xi[lay - 1][j] + self.momentum * self.d_w_prev[lay][neuron][j]
          self.d_w_prev[lay][neuron][j] = self.d_w[lay][neuron][j]
          self.w[lay][neuron][j] += self.d_w[lay][neuron][j]

        self.d_theta[lay][neuron] = self.learning_rate * self.delta[lay][neuron] + self.momentum * self.d_theta_prev[lay][neuron]
        self.d_theta_prev[lay][neuron] = self.d_theta[lay][neuron]
        self.theta[lay][neuron] += self.d_theta[lay][neuron]
        

  # Predict the output for a given input
  # X: samples
  # return: predictions
  def predict(self, X):
    predictions = []

    for pat in X:
      predictions.append(self.__feed_forward(pat))

    return predictions
    
  # Return the errors of the training and validation sets
  def loss_epochs(self):
    return self.errors_train, self.errors_valid
      

# layers include input layer + hidden layers + output layer
layers = [4, 9, 6, 1]

# list of activation functions coded as lambda functions
fact = [
    lambda x: 1 / (1 + np.exp(-x)),         # Sigmoid
    lambda x: max(0, x),                    # ReLU
    lambda x: x,                            # Linear
    lambda x: (pow(2.71828, x) - pow(2.71828, -x)) / (pow(2.71828, x) + pow(2.71828, -x))  # Tanh
]

derivatives = [
    lambda x: (1 / (1 + np.exp(-x)))*1-(1 / (1 + np.exp(-x))),  # Derivative of Sigmoid
    lambda x: 1 if x > 0 else 0,                                        # Derivative of ReLU
    lambda x: 1,                                                        # Derivative of Linear
    lambda x: 1 - ((pow(2.71828, x) - pow(2.71828, -x)) / (pow(2.71828, x) + pow(2.71828, -x)))**2                      # Derivative of Tanh
]


# todo: add read parameters from args

nn = MyNeuralNetwork(layers, fact[0], derivatives[0],perc=0.85, epochs=1000, learning_rate=0.1, momentum=0.9)  

# Read data and get train, validation and test sets (85% train and validation ,15% test)
data = np.loadtxt('A1-turbine-normalized.csv', delimiter=',', skiprows=1)
np.random.shuffle(data)

percentage_train_validation = 0.85
index_split = int(percentage_train_validation * len(data))
train_validation_set = data[:index_split]
test_set = data[index_split:]

# Train the neural network
nn.fit(train_validation_set[0:-1], train_validation_set[-1]) 

# Get errors and plot them
errors_train, errors_valid = nn.loss_epochs()
plt.plot(errors_train[:, 0], errors_train[:, 1], label="train")
plt.plot(errors_valid[:, 0], errors_valid[:, 1], label="validation")
plt.legend()
plt.show()

# Predict
predictions = nn.predict(test_set[0:-1])

# Plot predictions vs real values
plt.plot(y, label="real")
plt.plot(predictions, label="prediction")
plt.legend()
plt.show()


'''
# Check if the neural network is correctly executed
print("L = ", nn.L, end="\n")
print("n = ", nn.n, end="\n")

print("h = ", nn.h, end="\n")
print("h[1] = ", nn.h[1], end="\n")

print("xi = ", nn.xi, end="\n")
print("xi[0] = ", nn.xi[0], end="\n")
print("xi[1] = ", nn.xi[0], end="\n")

print("w = ", nn.w, end="\n")
print("w[0] = ", nn.w[0], end="\n")
print("w[1] = ", nn.w[1], end="\n")

print("theta = ", nn.theta, end="\n")
print("theta[1] = ", nn.theta[1], end="\n")

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
'''
