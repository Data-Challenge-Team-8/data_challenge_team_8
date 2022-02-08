import pandas as pd


class Classifier:

    def __init__(self):
        pass

    def train(self, x_data, y_data):
        raise NotImplementedError

    def predict(self, x_data) -> pd.DataFrame:
        raise NotImplementedError

    def test(self, x_data, y_data):
        """
        Makes a prediction over x_data and compares with y_data
        :param x_data:
        :param y_data:
        :return:
        """
        raise NotImplementedError
