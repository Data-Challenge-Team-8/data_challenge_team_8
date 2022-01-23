from typing import Tuple

import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from sklearn.model_selection import train_test_split
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
from sktime.classification.interval_based import TimeSeriesForestClassifier

from objects.training_set import TrainingSet, Patient
from classifier.timeseries.time_series_classifier import TimeSeriesClassifier


class TimeSeriesForest(TimeSeriesClassifier):

    def __init__(self, data_set: TrainingSet, train_fraction: float = 0.8, label: str = "HR"):
        super().__init__(data_set=data_set, train_fraction=train_fraction)
        self.selected_label = label
        self.model = TimeSeriesForestClassifier(random_state=1337, n_jobs=-1)
        self.train_data = None
        self.test_data = None

    def transform_data_set(self, data_set: TrainingSet, label_set: pd.DataFrame) -> tuple:
        transformed_data_set = data_set.get_timeseries_df(use_interpolation=True, fix_missing_values=True,
                                                          limit_to_features=[self.selected_label])
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

        print("shape", X_train.shape)
        self.model.fit(X_train, y_train)

    def predict(self, x_data=None):
        if x_data is None:
            x_data = self.test_data[0]

        y_prediction = self.model.predict(x_data)
        return y_prediction

    def test(self, x_data, y_data):
        return self.get_confusion_matrix(y_data, self.predict(x_data))

    def get_confusion_matrix(self, y_data, y_predicted) -> np.ndarray:
        cm = confusion_matrix(y_data, y_predicted)
        return cm

    def get_classification_report(self, x_data, y_data):
        report = classification_report(y_data, self.predict(x_data))
        return report

    def display_confusion_matrix(self, test_set=None, plotting: bool = False):
        if test_set is None:
            test_set = self.test_data
        x_test, y_test = test_set

        # classification_report
        report = self.get_classification_report(x_test, y_test)
        print(report)
        # confusion matrix plot
        if plotting:
            cm: np.ndarray = self.test(x_test, y_test)
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sepsis", "Sepsis"])
            disp.plot()
            # funktioniert leider nicht mit title
            # temp_text_obj = Text(x=100, y=50, text=f"Confusion Matrix for version: {version}")
            # disp.ax_.title = temp_text_obj
            plt.show()


