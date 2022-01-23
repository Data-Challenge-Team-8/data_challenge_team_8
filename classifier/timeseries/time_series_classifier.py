import numpy as np
import pandas as pd

from objects.training_set import TrainingSet


class TimeSeriesClassifier:
    """
    Parent class for a common interface between all TimeSeries classifiers
    """

    def __init__(self, data_set: TrainingSet, train_fraction: float = 0.8, custom_label=None):
        self.data_set = data_set
        self.label_set = custom_label
        self.model = None
        self.__train_fraction = train_fraction

    def transform_data_set(self, data_set, label_set) -> pd.DataFrame:
        """
        Transform the given TrainingSet into a format used by the classifier
        :param data_set:
        :param label_set:
        :return:
        """
        raise NotImplementedError

    def setup(self):
        """
        Do setup tasks for the classifier like transforming the given data set
        :return:
        """
        raise NotImplementedError

    def train(self, train_set: pd.DataFrame = None):
        """
        Train this model with the training set derived from the given data_set if train_set is None
        else use the given train_set.

        See train_fraction to adjust the size
        :param train_set: training data set as transformed by self.transform_data_set()
        :return:
        """
        raise NotImplementedError

    def predict(self, test_set: pd.DataFrame = None):
        """
        Test this instances model against the test set derived from the given data_set if test_set is None
        else use the given test_set.

        The size of this test set is  1 - train_fraction * len(data_set)
        :param test_set: test data set as transformed by self.transform_data_set()
        :return:
        """
        raise NotImplementedError
