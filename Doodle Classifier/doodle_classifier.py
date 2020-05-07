import numpy as np
from neural_network import NeuralNetwork


class DoodleClassifier:
    def __init__(self):
        # Length of each dataset
        self.length = 2000

        # Loading datasets
        self.icecream_dataset = np.load('full_numpy_bitmap_ice_cream.npy')[:self.length]
        self.cat_dataset = np.load('full_numpy_bitmap_cat.npy')[:self.length]
        self.eiffel_dataset = np.load('full_numpy_bitmap_The_Eiffel_Tower.npy')[:self.length]

        # Defining labels for each dataset
        self.label_icecream = np.zeros((self.length, 1)) + 0
        self.label_cat = np.zeros((self.length, 1)) + 1
        self.eiffel_pizza = np.zeros((self.length, 1)) + 2

        # Creating each training datasets
        self.icecream_training_set = np.concatenate((self.label_icecream, self.icecream_dataset), axis=1)
        self.cat_training_set = np.concatenate((self.label_cat, self.cat_dataset), axis=1)
        self.eiffel_training_set = np.concatenate((self.eiffel_pizza, self.eiffel_dataset), axis=1)

        # Merging training sets and shuffling them
        self.training_set = np.concatenate((self.icecream_training_set, self.cat_training_set, self.eiffel_training_set), axis=0).reshape(self.length * 3, -1, 1)
        np.random.shuffle(self.training_set)

        self.nn = NeuralNetwork(784, 64, 3, 0.20, 0.02)
        self.epochs = 5

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

    def predict_doodle(self, drawing):
        self.__train_nn()
        return self.nn.predict(drawing)



