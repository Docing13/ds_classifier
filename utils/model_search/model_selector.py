import matplotlib.pyplot as plt
import seaborn as sns
import pickle

from sklearn.ensemble import RandomForestClassifier
from sklearn.svm import SVC
from sklearn.tree import DecisionTreeClassifier

from sklearn.metrics import accuracy_score

import pandas as pd

from custom_classifiers.nn_based.nn_classifier import CustomNNClassifier
from custom_classifiers.xgb_based.xgb_classifier import CustomXGBClassifier
from settings.constants import LOG_COLS, MODEL, ACCURACY, CLASSIFIER, NAME, \
    COLOR, PLOT_TITLE


class ModelSelector:
    """
    Fit models and find the best
    """

    def __init__(self):

        self.__classifiers = [
            SVC(),
            DecisionTreeClassifier(),
            RandomForestClassifier(),
            CustomXGBClassifier(),
            CustomNNClassifier()
        ]

        self.__log: pd.DataFrame = pd.DataFrame(columns=LOG_COLS)

    def fit(self, x, y):

        for classifier in self.__classifiers:
            classifier.fit(x, y)

    def verbose(self, x, y):

        for classifier in self.__classifiers:

            y_pred = classifier.predict(x)
            score = accuracy_score(y_true=y,
                                   y_pred=y_pred)

            classifier_name = classifier.__class__.__name__
            data = pd.DataFrame([[classifier_name, score, classifier]],
                                columns=LOG_COLS)

            self.__log = self.__log.append(data)

    def save_best_model(self):
        with open(MODEL, 'wb') as f:
            model = self.best_model
            pickle.dump(model, f)

    @property
    def best_model(self):
        best_acc = self.__log[ACCURACY].max()
        best_row = self.__log[self.__log[ACCURACY] == best_acc]

        model = best_row[CLASSIFIER].item()
        return model

    @property
    def models_data(self) -> pd.DataFrame:
        return self.__log.loc[:, [NAME, ACCURACY]]

    def models_plot(self):
        data: pd.DataFrame = self.__log.loc[:,  [NAME, ACCURACY]]

        plt.xlabel(ACCURACY)
        plt.title(PLOT_TITLE)

        sns.barplot(x=ACCURACY, y=NAME, data=data, color=COLOR)
        plt.show()
