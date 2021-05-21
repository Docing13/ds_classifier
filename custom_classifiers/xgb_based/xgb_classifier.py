from custom_classifiers.base.base import CustomClassifierBase
from sklearn.model_selection import StratifiedKFold, GridSearchCV
from xgboost.sklearn import XGBClassifier
import numpy as np

from custom_classifiers.xgb_based.settings.constants import GRID_PARAMS, \
    N_SPLIT, OBJECTIVE, NUM_CLASS, SCORING


class CustomXGBClassifier(CustomClassifierBase):
    def __init__(self):
        self.__model = self.__new_estimator
        self.__estimator = self.__new_estimator

    @property
    def __new_estimator(self) -> XGBClassifier:
        estimator = XGBClassifier(objective=OBJECTIVE,
                                  use_label_encoder=False,
                                  num_class=NUM_CLASS,
                                  nthreads=8,
                                  verbosity=0)
        return estimator

    def fit(self, x_train, y_train) -> None:
        y_train = y_train-1
        cv = StratifiedKFold(n_splits=N_SPLIT, shuffle=True)

        grid = GridSearchCV(estimator=self.__estimator,
                            param_grid=GRID_PARAMS,
                            cv=cv,
                            scoring=SCORING)

        grid.fit(x_train, y_train)

        self.__model = grid.best_estimator_

    def predict(self, x) -> np.ndarray:
        ans = self.__model.predict(x)
        ans += 1
        return ans
