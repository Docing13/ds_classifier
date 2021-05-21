import pickle
from settings.constants import MODEL


class Predictor:
    """
    Model interface class
    """
    def __init__(self):
        self.__model = self.__predictor

    def predict(self, data):
        return self.__model.predict(data)

    def load(self, path=None):
        if path:
            self.__model = self.__load_from_path(path)
        else:
            self.__model = self.__predictor

    @staticmethod
    def __load_from_path(path):
        try:
            predictor = pickle.load(open(path, 'rb'))
            return predictor

        except Exception:
            raise FileNotFoundError("Can't load model")

    @property
    def __predictor(self):
        return self.__load_from_path(MODEL)
