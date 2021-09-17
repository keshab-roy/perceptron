import numpy as np

class Perceptron:
    def __init__(self, eta, epochs):
        self.weights = np.random.randn(3) * 1e-4 #Small weight init
        print(f"Initial weights before training: {self.weights}")
        self.eta = eta #learning rate
        self.epochs = epochs

    def activationFunction(self, inputs, weights):
        z = np.dot(inputs, weights) #Z = W * X
        return np.where(z > 0, 1, 0) #Condition, true val, false val

    def fit(self, x, y):
        self.x = x
        self.y = y
        x_with_bias = np.c_[self.x, -np.ones((len(self.x)))] #Concat
        print(f"x_with_bias:\n {x_with_bias}")

        for epoch in range(self.epochs):
            print("--" * 10)
            print(f"Epoch: {epoch}")
            print("--" * 10)

            y_hat = self.activationFunction(x_with_bias, self.weights) #Forward propagation
            print(f"Predicted value after the forward pass: {y_hat}")
            self.error = self.y - y_hat
            print(f"Error:\n {self.error}")
            self.weights = self.weights + self.eta * np.dot(x_with_bias.T, self.error) #Backward propagation
            print(f"Updated weights after epoch:\n {self.weights}")
            print("###########" * 10)

    def predict(self, x):
        x_with_bias = np.c_[x, -np.ones(len(x))]
        return self.activationFunction(x_with_bias, self.weights)

    def total_loss(self):
        total_loss = np.sum(self.error)
        print(f"Total Loss: {total_loss}")
        return total_loss
