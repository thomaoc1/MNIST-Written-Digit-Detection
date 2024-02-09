from sklearn.neighbors import KNeighborsClassifier

from src.classicmodels.classicmodel import ClassicModel


class KNN(ClassicModel):
    """
    K-Nearest Neighbors model

    :param hparam_tuning_method: The type of cross validation to use
    :param pca_threshold: The PCA threshold to use
    """
    def __init__(self, hparam_tuning_method: str = None, pca_threshold: float = None, data_set_size: float = 1.0):
        super().__init__(
            base_model=KNeighborsClassifier(),
            default_hparams={'n_neighbors': 1},
            param_grid={'n_neighbors': [1, 2, 3, 4]},
            hparam_tuning_method=hparam_tuning_method,
            pca_threshold=pca_threshold,
            data_set_size=data_set_size
        )

        self.best_n_neighbors = self._best_param_dict['n_neighbors']
        self.final_model = KNeighborsClassifier(n_neighbors=self.best_n_neighbors)

    def __str__(self):
        return f'KNN(n_neighbors={self.best_n_neighbors})'

