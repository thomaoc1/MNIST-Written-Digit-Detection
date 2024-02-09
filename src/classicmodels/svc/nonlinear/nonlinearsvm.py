from sklearn.svm import SVC

from src.classicmodels.classicmodel import ClassicModel


class NonLinearSVM(ClassicModel):
    """
    Non-linear SVM model

    :param hparam_tuning_method: The type of cross validation to use
    :param pca_threshold: The PCA threshold to use
    """
    def __init__(self, hparam_tuning_method: str = None, pca_threshold: float = None):
        super().__init__(
            base_model=SVC(kernel='rbf'),
            default_hparams={'C': 10.0, 'gamma': 0.01},
            param_grid={'C': [0.01, 0.1, 1, 10], 'gamma': [0.01, 0.1, 1, 10]},
            hparam_tuning_method=hparam_tuning_method,
            pca_threshold=pca_threshold,
        )

        self.best_c = self._best_param_dict['C']
        self.best_gamma = self._best_param_dict['gamma']
        self._final_model = SVC(kernel='rbf', C=self.best_c, gamma=self.best_gamma)

    def __str__(self):
        return f'NonLinearSVM(C={self.best_c}, gamma={self.best_gamma})'
