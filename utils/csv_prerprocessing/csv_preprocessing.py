import pandas as pd

from settings.constants import LAST_FEATURE, LABELS
from utils.dataloader import DataLoader


class CSVPreprocessor:

    @staticmethod
    def to_x_y(csv: str) -> (pd.DataFrame, pd.DataFrame):

        df = CSVPreprocessor.to_df(csv)
        x = df.loc[:, :LAST_FEATURE]
        y = df.loc[:, LABELS]

        return x, y

    @staticmethod
    def to_df(csv: str) -> pd.DataFrame:
        data = pd.read_csv(csv)

        loader = DataLoader()
        loader.fit(data)
        return loader.load_data()
