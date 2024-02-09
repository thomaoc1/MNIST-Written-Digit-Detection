from PIL import Image

from src.mnist import MNIST
import matplotlib.pyplot as plt
from sklearn.metrics import confusion_matrix, ConfusionMatrixDisplay, classification_report
import numpy as np
import os


def confusion_matrix_custom_display(model):
    cm = confusion_matrix(model.y_test, model.get_model().predict(model.X_test))
    cm_normalized = cm.astype('float') / cm.sum(axis=1)[:, np.newaxis]
    cm_display = np.where(cm_normalized < 0.01, 0, np.round(cm_normalized, 2))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm_display, display_labels=MNIST.get_target_names())
    disp.plot(cmap=plt.cm.Blues)


def recursive_predict(model, directory_path):
    """
    Predicts the labels of all the images in the directory and displays them with their predictions

    :param model: The model to use for the predictions
    :param directory_path: The path to the directory containing the images
    """

    if not os.path.isdir(directory_path):
        raise ValueError("Path is not a directory")

    fig, axs = plt.subplots(nrows=1, ncols=len(os.listdir(directory_path)), figsize=(20, 4))
    if len(os.listdir(directory_path)) == 1:
        axs = [axs]

    for i, filename in enumerate(os.listdir(directory_path)):
        if filename.endswith('.png'):  # Check if the file is an image
            full_path = os.path.join(directory_path, filename)

            # Load and normalize the image
            image_array = Image.open(full_path).convert('L')

            # Predict the image label
            prediction = model.predict_from_file(full_path)[0]

            # Display the image with the prediction
            axs[i].imshow(image_array, cmap='grey')  # Adjust the reshape if necessary
            axs[i].title.set_text(f'Prediction: {prediction}')
            axs[i].axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def predict_from_file(model, file_path):
    image = Image.open(file_path)
    image_gray = image.convert('L')
    image_array = np.array(image_gray)
    normalized_image_array = image_array / 255.0

    reshaped_image_array = np.expand_dims(normalized_image_array, axis=0)  # Add batch dimension
    reshaped_image_array = np.expand_dims(reshaped_image_array, axis=-1)

    return model.predict(reshaped_image_array)


def rec_predict_cnn(model, directory_path):
    """
    Predicts the labels of all the images in the directory and displays them with their predictions

    :param model: the model
    :param directory_path: the path to the directory containing the images
    :return:
    """
    if not os.path.isdir(directory_path):
        raise ValueError("Path is not a directory")

    fig, axs = plt.subplots(nrows=1, ncols=len(os.listdir(directory_path)), figsize=(20, 4))
    if len(os.listdir(directory_path)) == 1:
        axs = [axs]

    for i, filename in enumerate(os.listdir(directory_path)):
        if filename.endswith('.png'):  # Check if the file is an image
            full_path = os.path.join(directory_path, filename)

            # Load and normalize the image
            image_array = Image.open(full_path).convert('L')

            # Predict the image label
            prediction = predict_from_file(model, full_path)

            # Display the image with the prediction
            axs[i].imshow(image_array, cmap='grey')  # Adjust the reshape if necessary
            axs[i].title.set_text(f'Prediction: {prediction}')
            axs[i].axis('off')

    # Adjust layout and show the plot
    plt.tight_layout()
    plt.show()


def report(model):
    print(classification_report(model.y_test, model.get_model().predict(model.X_test)))
