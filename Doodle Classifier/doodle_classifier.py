import numpy as np
from neural_network import NeuralNetwork


class DoodleClassifier:
    def __init__(self):
        # Length of each dataset
        self.length = 5000

        # Loading datasets
        # Dataset names are just samples. To use different datasets, refactor names of .npy files and variables.
        self.icecream_dataset = np.load('full-numpy_bitmap-ice cream.npy')[:self.length]
        self.basketball_dataset = np.load('full-numpy_bitmap-basketball.npy')[:self.length]
        self.eiffel_dataset = np.load('full-numpy_bitmap-The Eiffel Tower.npy')[:self.length]

        # Defining labels for each dataset
        self.label_icecream = np.zeros((self.length, 1)) + 0
        self.label_car = np.zeros((self.length, 1)) + 1
        self.label_eiffel = np.zeros((self.length, 1)) + 2

        # Creating each training datasets
        self.icecream_training_set = np.concatenate((self.label_icecream, self.icecream_dataset), axis=1)
        self.basketball_training_set = np.concatenate((self.label_car, self.basketball_dataset), axis=1)
        self.eiffel_training_set = np.concatenate((self.label_eiffel, self.eiffel_dataset), axis=1)

        # Merging training sets and shuffling them
        self.training_set = np.concatenate(
            (self.icecream_training_set, self.basketball_training_set, self.eiffel_training_set), axis=0)\
            .reshape(self.length * 3, -1, 1)
        np.random.shuffle(self.training_set)

        self.nn = NeuralNetwork(784, 128, 3, 0.0005, 0.0000)
        self.epochs = 10

    def __train_nn(self):
        for _ in range(self.epochs):

            # Training network (for one epoch)
            for record in self.training_set:
                inputs = (record[1:] / 255.0 * 0.99) + 0.01

                targets = np.zeros(3).reshape((-1, 1)) + 0.01
                targets[int(record[0])] = 0.99
                self.nn.train(inputs, targets)

                pass
            pass

    def train(self):
        print("Training started...")
        self.__train_nn()
        print("Training finished!")

    def predict_doodle(self, drawing):
        return self.nn.predict(drawing)



