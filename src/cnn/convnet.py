import tensorflow as tf
from tensorflow.keras import datasets, layers, models
import numpy as np
import matplotlib.pyplot as plt


class MNISTConvNet:
    """
    A CNN model for the MNIST dataset
    """
    def __init__(self):
        self.history = None
        (self.X_train, self.y_train), (self.X_test, self.y_test) = datasets.mnist.load_data()
        self.X_train, self.X_test = self.X_train / 255.0, self.X_test / 255.0

        self.model = models.Sequential([
            layers.Conv2D(32, (3, 3), activation='relu', input_shape=(28, 28, 1)),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.MaxPooling2D((2, 2)),
            layers.Conv2D(64, (3, 3), activation='relu'),
            layers.Flatten(),
            layers.Dense(64, activation='relu'),
            layers.Dense(10)
        ])

        self.model.summary()

        self.model.compile(optimizer='adam',
                           loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
                           metrics=['accuracy'])

    def train(self):
        self.history = self.model.fit(self.X_train, self.y_train, epochs=5, validation_data=(self.X_test, self.y_test))

    def evaluate(self):
        return self.model.evaluate(self.X_test, self.y_test, verbose=2)

    def plot_history(self):
        plt.plot(self.history.history['accuracy'], label='accuracy')
        plt.plot(self.history.history['val_accuracy'], label='val_accuracy')
        plt.xlabel('Epoch')
        plt.ylabel('Accuracy')
        plt.ylim([0.8, 1.1])
        plt.legend(loc='lower right')
        plt.show()

    def predict(self, x):
        return np.argmax(self.model.predict(x))

