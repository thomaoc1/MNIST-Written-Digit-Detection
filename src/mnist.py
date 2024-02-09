import tensorflow as tf
import tensorflow_datasets as tfds
from sklearn.decomposition import PCA
import matplotlib.pyplot as plt
import numpy as np


class MNIST:
    def __init__(self, pca_threshold: float = 0.95, data_set_size: float = 1.0):
        self._pca_threshold: float = pca_threshold
        self._data_set_size: int = int(data_set_size * 70000)
        self._pca = None
        self._components: int = 28 * 28
        self._train_ds: tf.data.Dataset = tfds.load('mnist', split='train', as_supervised=True)
        self._test_ds: tf.data.Dataset = tfds.load('mnist', split='test', as_supervised=True)
        self._normalize_dataset()

    @staticmethod
    def get_target_names():
        return [str(i) for i in range(10)]

    """
    Normalises the dataset [0, 1]
    """
    def _normalize_dataset(self):
        def normalize_img(image, label):
            return tf.cast(image, tf.float32) / 255.0, label

        self._train_ds = self._train_ds.map(normalize_img)
        self._test_ds = self._test_ds.map(normalize_img)

    def _pca_cumulative_variance(self) -> (np.ndarray, int):
        images, _, _, _ = self.get_train_test_as_numpy()
        pca = PCA().fit(images)
        cumulative_variance = np.cumsum(pca.explained_variance_ratio_)
        components_99 = np.argmax(cumulative_variance >= self._pca_threshold) + 1
        return cumulative_variance, components_99

    def get_train_test_as_numpy(self) -> (np.ndarray, np.ndarray, np.ndarray, np.ndarray):
        if self._data_set_size == 70000:
            X_train, y_train = ds_to_numpy(self._train_ds)
            X_test, y_test = ds_to_numpy(self._test_ds)
        else:
            train: int = int(self._data_set_size * 0.8)
            test: int = self._data_set_size - train
            small_train_ds, small_test_ds = self._get_smaller_datasets(train, test)
            X_train, y_train = ds_to_numpy(small_train_ds)
            X_test, y_test = ds_to_numpy(small_test_ds)

        return X_train, y_train, X_test, y_test

    def _get_smaller_datasets(self, train_size: int, test_size: int) -> (tf.data.Dataset, tf.data.Dataset):
        small_train_ds = self._train_ds.shuffle(len(self._train_ds)).take(train_size)
        small_test_ds = self._test_ds.shuffle(len(self._train_ds)).take(test_size)
        return small_train_ds, small_test_ds

    def get_train_test_reduced(self):
        train_images, train_labels, test_images, test_labels = self.get_train_test_as_numpy()
        _, components = self._pca_cumulative_variance()
        self._components = components

        self._pca = PCA(n_components=components)

        print("Reducing features with PCA...", end=" ")

        reduced_train_images = self._pca.fit_transform(train_images)
        reduced_test_images = self._pca.transform(test_images)

        print(f"Features reduced ({train_images.shape[1]} -> {reduced_train_images.shape[1]})...", end=" ")

        return reduced_train_images, train_labels, reduced_test_images, test_labels

    def display_pca(self) -> None:
        cumulative_variance, components = self._pca_cumulative_variance()
        plt.figure(figsize=(10, 7))
        plt.plot(cumulative_variance, marker='o', linestyle='--', label='Cumulative Explained Variance')
        plt.axhline(y=self._pca_threshold, color='r', linestyle='-',
                    label=f'{self._pca_threshold * 100}% Explained Variance')

        plt.axvline(x=components, color='g', linestyle='-', label=f'Components = {components}')
        plt.title('Explained Variance by Components')
        plt.xlabel('Number of Components')
        plt.ylabel('Cumulative Explained Variance')
        plt.legend()
        plt.show()

    def transform(self, param) -> np.ndarray:
        if self._pca is None:
            return param
        return self._pca.transform([param])


def ds_to_numpy(ds: tf.data.Dataset) -> (np.ndarray, np.ndarray):
    images = []
    labels = []
    for image, label in ds:
        images.append(image.numpy().flatten())
        labels.append(label.numpy())
    return np.array(images), np.array(labels)


if __name__ == '__main__':
    mnist = MNIST()

