import pandas as pd
from typing import Optional
from settings.constants import NUMERIC_FEATURES, DROP_FEATURES, LABELS
from sklearn.preprocessing import MinMaxScaler


class DataLoader:
    """
    Loading and precessing data, standard flow for data:
    Drop useless columns, MinMax standardization,
    Bring label column to 0-(max-1) range
    """

    def __init__(self):
        self.__dataset: Optional[pd.DataFrame] = None

    def fit(self, dataset: pd.DataFrame) -> None:
        self.__dataset = dataset.copy()

    def load_data(self) -> pd.DataFrame:
        if self.__dataset:
            # drop useless fields
            data = self.__dataset.drop(columns=DROP_FEATURES)

            # numeric values standardisation
            minmax = MinMaxScaler()
            data[NUMERIC_FEATURES] = minmax.fit_transform(data[NUMERIC_FEATURES])

            # set labels to diapason 0 - (max-min)
            data[LABELS] -= data[LABELS].min()

            return data

        else:
            return self.__dataset
