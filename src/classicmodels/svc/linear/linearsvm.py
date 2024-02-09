from sklearn.svm import LinearSVC

from src.classicmodels.classicmodel import ClassicModel


class LinearSVM(ClassicModel):
    """
    Linear SVM model

    :param hparam_tuning_method: The type of cross validation to use
    :param pca_threshold: The PCA threshold to use
    """
    def __init__(self, hparam_tuning_method: str = None, pca_threshold: float = None):
        super().__init__(
            base_model=LinearSVC(dual=False),
            default_hparams={'C': 1.0},
            param_grid={'C': [0.01, 0.1, 1, 10]},
            pca_threshold=pca_threshold,
            hparam_tuning_method=hparam_tuning_method,
        )

        self.best_c = self._best_param_dict['C']
        self._final_model = LinearSVC(C=self.best_c, dual=False)

    def __str__(self):
        return f'LinearSVM(C={self.best_c})'
