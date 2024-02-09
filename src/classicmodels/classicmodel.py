from sklearn.model_selection import RandomizedSearchCV, GridSearchCV
from PIL import Image
import numpy as np

from src.mnist import MNIST


class ClassicModel:
    def __init__(self,
                 base_model,
                 default_hparams: dict = None,
                 param_grid: dict = None,
                 pca_threshold: float = None,
                 hparam_tuning_method: str = None,
                 data_set_size: float = 1.0
                 ):

        print('Loading MNIST dataset...', end=' ')
        self._pca_threshold = pca_threshold
        if self._pca_threshold is not None:
            self.mnist: MNIST = MNIST(pca_threshold=self._pca_threshold, data_set_size=data_set_size)
            self.X_train, self.y_train, self.X_test, self.y_test = self.mnist.get_train_test_reduced()
        else:
            self.mnist: MNIST = MNIST(data_set_size=data_set_size)
            self.X_train, self.y_train, self.X_test, self.y_test = self.mnist.get_train_test_as_numpy()
        print('Done!')

        self._default_hparams = default_hparams
        self._param_grid = param_grid
        self._base_model = base_model

        self._best_param_dict = self._hyperparameter_tuning(hparam_tuning_method=hparam_tuning_method)

        self._final_model = None

    def __str__(self):
        raise NotImplementedError('Method __str__() must be implemented by subclass')

    def __repr__(self):
        return self.__str__()

    def _hyperparameter_tuning(self, hparam_tuning_method: str):
        if hparam_tuning_method == 'random':
            search = RandomizedSearchCV(self._base_model, self._param_grid, cv=3, scoring='accuracy', verbose=2)
        elif hparam_tuning_method == 'grid':
            search = GridSearchCV(self._base_model, self._param_grid, cv=3, scoring='accuracy', verbose=2)
        elif hparam_tuning_method is None:
            return self._default_hparams
        else:
            raise ValueError("hparam_tuning_method must be 'random' or 'grid'")

        search.fit(self.X_train, self.y_train)

        print('Best parameters found: ', search.best_params_)
        print('Best cross-validation score: {:.2f}'.format(search.best_score_))

        return search.best_params_

    def train(self) -> None:
        if self._final_model is None:
            raise ValueError('Model has not been trained yet')

        print('Training ...')
        self._final_model.fit(self.X_train, self.y_train)
        print('Done!')

    def evaluate(self) -> float:
        if self._final_model is None:
            raise ValueError('Model has not been trained yet')

        print('Evaluating ...', end=' ')
        return self._final_model.score(self.X_test, self.y_test)

    def get_model(self):
        return self._final_model

    def predict_from_file(self, file_path) -> int:
        image = Image.open(file_path)
        image_gray = image.convert('L')
        image_array = np.array(image_gray)
        normalized_image_array = image_array / 255.0
        transformed_array = self.mnist.transform(normalized_image_array.flatten())
        return self.predict(transformed_array)

    def predict(self, x) -> int:
        if self._final_model is None:
            raise ValueError('Model has not been trained yet')

        return self._final_model.predict(x)
