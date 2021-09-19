import numpy as np
import logging
from tqdm import tqdm


class Perceptron:
    def __init__(self, eta, epochs):
        self.weights = np.random.randn(3) * 1e-4  # Small weight init
        logging.info(f"Initial weights before training: {self.weights}")
        self.eta = eta  # learning rate
        self.epochs = epochs

    def activationFunction(self, inputs, weights):
        z = np.dot(inputs, weights)  # Z = W * X
        return np.where(z > 0, 1, 0)  # Condition, true val, false val

    def fit(self, x, y):
        self.x = x
        self.y = y
        x_with_bias = np.c_[self.x, -np.ones((len(self.x)))]  # Concat
        logging.info(f"x_with_bias:\n {x_with_bias}")

        for epoch in tqdm(range(self.epochs), total=self.epochs, desc="Training the data"):
            logging.info("--" * 10)
            logging.info(f"Epoch: {epoch}")
            logging.info("--" * 10)

            y_hat = self.activationFunction(
                x_with_bias, self.weights)  # Forward propagation
            logging.info(f"Predicted value after the forward pass: {y_hat}")
            self.error = self.y - y_hat
            logging.info(f"Error:\n {self.error}")
            self.weights = self.weights + self.eta * \
                np.dot(x_with_bias.T, self.error)  # Backward propagation
            logging.info(f"Updated weights after epoch:\n {self.weights}")
            logging.info("###########" * 10)

    def predict(self, x):
        x_with_bias = np.c_[x, -np.ones(len(x))]
        return self.activationFunction(x_with_bias, self.weights)

    def total_loss(self):
        total_loss = np.sum(self.error)
        logging.info(f"Total Loss: {total_loss}")
        return total_loss
