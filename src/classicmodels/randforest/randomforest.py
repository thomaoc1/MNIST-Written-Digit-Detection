from src.classicmodels.classicmodel import ClassicModel
from sklearn.ensemble import RandomForestClassifier


class RandomForest(ClassicModel):
    """
    Random Forest model

    :param hparam_tuning_method: The type of cross validation to use
    :param pca_threshold: The PCA threshold to use
    """
    def __init__(self, hparam_tuning_method: str = None, pca_threshold: float = None, data_set_size: float = 1.0):
        super().__init__(
            base_model=RandomForestClassifier(),
            default_hparams={'n_estimators': 100, 'max_depth': None},
            param_grid={'n_estimators': [10, 100, 1000], 'max_depth': [10, 100]},
            hparam_tuning_method=hparam_tuning_method,
            pca_threshold=pca_threshold,
            data_set_size=data_set_size
        )

        self.best_n_estimators = self._best_param_dict['n_estimators']
        self.best_max_depth = self._best_param_dict['max_depth']

        self._final_model = RandomForestClassifier(n_estimators=self.best_n_estimators, max_depth=self.best_max_depth)

    def __str__(self):
        return f'RandomForest(n_estimators={self.best_n_estimators}, max_depth={self.best_max_depth})'
