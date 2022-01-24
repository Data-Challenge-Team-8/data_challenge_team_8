import pandas as pd

from classifier.timeseries.time_series_classifier import TimeSeriesClassifier
from objects.training_set import TrainingSet, Patient


class ShapeletClassifier(TimeSeriesClassifier):

    def __init__(self, data_set: TrainingSet, target_label: str = None):
        super().__init__(data_set=data_set, custom_label=target_label)

    def transform_data_set(self, data_set, label_set) -> pd.DataFrame:
        pass

    def setup(self):
        pass

    def train(self, train_set: pd.DataFrame = None):
        pass

    def predict(self, test_set: pd.DataFrame = None):
        pass