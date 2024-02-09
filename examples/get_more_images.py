import tensorflow as tf
from PIL import Image
import random

if __name__ == '__main__':
    # Load the MNIST dataset
    mnist = tf.keras.datasets.mnist
    (train_images, train_labels), (test_images, test_labels) = mnist.load_data()

    # Convert the first image in the training set to a PIL image and save
    for _ in range(100):
        i = random.randint(0, len(train_images))
        image = Image.fromarray(train_images[i])
        image.save(f'img/mnist_image_keras_28x28_{i}.png')
