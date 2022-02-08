from typing import List, Tuple

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.pipeline import Pipeline
from sklearn.ensemble import RandomForestClassifier
from sktime.transformations.panel.shapelets import ContractedShapeletTransform
from sklearn.model_selection import train_test_split

from classifier.timeseries.time_series_classifier import TimeSeriesClassifier
from objects.training_set import TrainingSet, Patient


class ShapeletClassifier(TimeSeriesClassifier):

    def __init__(self, data_set: TrainingSet, target_label: str = None, train_fraction: float = 0.8, features=None,
                 time_limit=1):
        """

        :param data_set: TrainingSet instance to be used as a data set for training and/or testing
        :param target_label: The label we want to predict and test with
        :param train_fraction: fraction of the data_set (0.0 to 1.0) used for training
        :param features: list of features that are trained with. Untested with multiple.
        :param time_limit: time limit for the shapelet search (in minutes)
        """
        super().__init__(data_set=data_set, custom_label=target_label, train_fraction=train_fraction)
        self.features = ["HR"] if features is None else features
        if len(self.features) > 1:
            raise ValueError("[WARNING] Using more than one feature with ShapeletClassifier might have terrible results!")
        # this (instead of list expression as default argument) prevents it to be mutable (serious bug)

        self.model: Pipeline = Pipeline([
            (
                "st",
                ContractedShapeletTransform(
                    random_state=1337,
                    time_contract_in_mins=time_limit,
                    num_candidates_to_sample_per_case=10,
                    verbose=2,  # False or 2
                ),
            ),
            ("rf", RandomForestClassifier(n_estimators=100)),
        ])

    def transform_data_set(self, data_set, label_set):
        transformed_data_set = data_set.get_timeseries_df(use_interpolation=True, fix_missing_values=True,
                                                          limit_to_features=self.features)

        transformed_labels = []
        for patient_id, label in transformed_data_set.index:
            transformed_labels.append(label_set[patient_id][0])

        return transformed_data_set, transformed_labels

    def setup(self):
        self.label_set = self.data_set.get_sepsis_label_df().transpose() if self.label_set is None else self.label_set
        transformed_data, transformed_labels = self.transform_data_set(self.data_set, self.label_set)

        X_train, X_test, y_train, y_test = train_test_split(transformed_data, transformed_labels, random_state=1337)
        self.train_data = X_train.to_numpy().reshape(len(X_train), -1), y_train
        self.test_data = X_test.to_numpy().reshape(len(X_test), -1), y_test

    def train(self, train_set: Tuple[pd.DataFrame, np.ndarray] = None):
        if train_set is None:
            train_set = self.train_data
        X_train = train_set[0]
        y_train = pd.Series(train_set[1])

        self.model.fit(X_train, y_train)

    def predict(self, x_data=None):
        if x_data is None:
            x_data = self.test_data[0]

        y_prediction = self.model.predict(x_data)
        return y_prediction

    def display_shapelets(self):
        """
        Display plots with the shapelets we've found!
        :return:
        """
        fig, axs = plt.subplots(len(self.model[0].shapelets))
        for i in range(len(self.model[0].shapelets)):
            s = self.model[0].shapelets[i]
            axs[i].plot(self.train_data[0].iloc[s.series_id, 0], "gray")
            axs[i].plot(list(range(s.start_pos, (s.start_pos + s.length))),
                        self.train_data[0].iloc[s.series_id, 0][s.start_pos : s.start_pos + s.length],
                        "r",
                        linewidth=3.0)

        plt.show()
