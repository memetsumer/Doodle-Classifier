import numpy as np
import scipy.special


def activation(x):
    return scipy.special.expit(x)  # Sigmoid function 


class NeuralNetwork:
    def __init__(self, n_input, n_hidden, n_output, learning_rate, lambda_val):
        self.n_input = n_input
        self.n_hidden = n_hidden
        self.n_output = n_output
        self.learning_rate = learning_rate
        self.lambda_val = lambda_val

        self.ih_weights = np.random.normal(0.0, pow(self.n_hidden, -0.5), (self.n_hidden, self.n_input))
        self.ho_weights = np.random.normal(0.0, pow(self.n_output, -0.5), (self.n_output, self.n_hidden))
        self.ih_bias = np.random.normal(0.0, pow(self.n_hidden, -0.5), (self.n_hidden, 1))
        self.ho_bias = np.random.normal(0.0, pow(self.n_output, -0.5), (self.n_output, 1))

    def train(self, training_inputs, training_outputs):
        z2 = np.dot(self.ih_weights, training_inputs)
        hidden_outputs = activation(z2 + self.ih_bias)
        z3 = np.dot(self.ho_weights, hidden_outputs)
        outputs = activation(z3 + self.ho_bias)

        output_errors = training_outputs - outputs

        hidden_errors = np.transpose(self.ho_weights).dot(output_errors)

        delta_ho_weights = self.learning_rate * np.dot(output_errors * outputs * (1 - outputs),
                                                       np.transpose(hidden_outputs))

        delta_ih_weights = self.learning_rate * np.dot(hidden_errors * hidden_outputs * (1 - hidden_outputs),
                                                       np.transpose(training_inputs))

        delta_ih_bias = self.learning_rate * hidden_errors * hidden_outputs * (1 - hidden_outputs)
        delta_ho_bias = self.learning_rate * output_errors * outputs * (1 - outputs)

        self.ho_weights += delta_ho_weights + (self.lambda_val / self.n_input) * delta_ho_weights
        self.ih_weights += delta_ih_weights + (self.lambda_val / self.n_hidden) * delta_ih_weights
        self.ih_bias += delta_ih_bias
        self.ho_bias += delta_ho_bias

    def predict(self, test_inputs):
        z2 = np.dot(self.ih_weights, test_inputs)
        hidden_outputs = activation(z2 + self.ih_bias)
        z3 = np.dot(self.ho_weights, hidden_outputs)
        outputs = activation(z3 + self.ho_bias)
        return outputs


