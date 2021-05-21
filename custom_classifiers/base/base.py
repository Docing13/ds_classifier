from abc import ABC, abstractmethod
import numpy as np


class CustomClassifierBase(ABC):

    @abstractmethod
    def fit(self, x_train, y_train) -> None:
        pass

    @abstractmethod
    def predict(self, x) -> np.ndarray:
        pass
