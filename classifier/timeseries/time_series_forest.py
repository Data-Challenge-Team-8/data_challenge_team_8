from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.model_selection import train_test_split
from sktime.classification.interval_based import TimeSeriesForestClassifier
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from objects.training_set import TrainingSet, Patient
from classifier.timeseries.time_series_classifier import TimeSeriesClassifier


class TimeSeriesForest(TimeSeriesClassifier):

    def __init__(self, data_set: TrainingSet, train_fraction: float = 0.8, feature: str = "HR"):
        super().__init__(data_set=data_set, train_fraction=train_fraction)
        self.selected_feature = feature
        self.model = TimeSeriesForestClassifier(random_state=1337, n_jobs=-1)
        self.train_data = None
        self.test_data = None

    def transform_data_set(self, data_set: TrainingSet, label_set: pd.DataFrame) -> tuple:
        transformed_data_set = data_set.get_timeseries_df(use_interpolation=True, fix_missing_values=True,
                                                          limit_to_features=[self.selected_feature])
        transformed_labels = []
        for patient_id, label in transformed_data_set.index:
            transformed_labels.append(label_set[patient_id][0])

        return transformed_data_set, transformed_labels

    def setup(self, use_balancing: str = None):
        """
        Setup the train and test data and apply balancing if asked to.
        :param use_balancing: None for no balancing, "SMOTE" for SMOTE oversampling, "NEARMISS" for NEARMISS undersampling.
        Please note the versions for Nearmiss ("Nearmiss-1", "Nearmiss-2", "Nearmiss-3")
        :return:
        """
        self.label_set = self.data_set.get_sepsis_label_df().transpose() if self.label_set is None else self.label_set
        transformed_data, transformed_labels = self.transform_data_set(self.data_set, self.label_set)
        X_train, X_test, y_train, y_test = train_test_split(transformed_data, transformed_labels, random_state=1337)
        if use_balancing == "SMOTE":
            print(f"Applying SMOTE to \"{self.data_set.name}\" data set ...")
            smote = SMOTE(random_state=1337)
            X_train, y_train = smote.fit_resample(X_train, y_train)
            X_test, y_test = smote.fit_resample(X_test, y_test)
        elif "NEARMISS-" in use_balancing:
            version = int(use_balancing[-1])
            print(f"Applying Nearmiss v{version} to \"{self.data_set.name}\" data set ...")
            nm = NearMiss(version=version)
            X_train, y_train = nm.fit_resample(X_train, y_train)
            X_test, y_test = nm.fit_resample(X_test, y_test)


        self.train_data = X_train.to_numpy().reshape(len(X_train), -1), y_train
        self.test_data = X_test.to_numpy().reshape(len(X_test), -1), y_test

    def train(self, train_set: Tuple[pd.DataFrame, np.ndarray] = None):
        if train_set is None:
            train_set = self.train_data
        X_train = train_set[0]
        y_train = pd.Series(train_set[1])

        print("shape", X_train.shape)
        self.model.fit(X_train, y_train)

    def predict(self, x_data=None):
        if x_data is None:
            x_data = self.test_data[0]
        elif isinstance(x_data, pd.DataFrame):
            x_data = x_data.to_numpy()

        y_prediction = self.model.predict(x_data)
        return y_prediction




