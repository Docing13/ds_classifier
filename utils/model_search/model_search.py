from settings.constants import TRAIN_CSV, VAL_CSV
from utils.csv_prerprocessing.csv_preprocessing import CSVPreprocessor
from utils.model_search.model_selector import ModelSelector


class ModelSearcher:
    """
    Facade class for fit and store best model
    """

    def __init__(self):
        self.__model_selector = ModelSelector()

    def fit(self):

        x_train, y_train = CSVPreprocessor.to_x_y(TRAIN_CSV)
        x_test, y_test = CSVPreprocessor.to_x_y(VAL_CSV)

        self.__model_selector.fit(x_train, x_test)
        self.__model_selector.verbose(y_train, y_test)
        self.__model_selector.save_best_model()

    @property
    def models(self):
        return self.__model_selector.models_data
