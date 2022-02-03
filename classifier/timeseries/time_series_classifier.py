import os.path
from typing import Tuple

import numpy as np
import pandas as pd
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
import matplotlib.pyplot as plt

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
        self.train_data: Tuple[pd.DataFrame, np.ndarray] = None
        self.test_data: Tuple[pd.DataFrame, np.ndarray] = None

    def transform_data_set(self, data_set, label_set):
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

    def train(self, train_set: Tuple[pd.DataFrame, np.ndarray] = None):
        """
        Train this model with the training set derived from the given data_set if train_set is None
        else use the given train_set.

        See train_fraction to adjust the size
        :param train_set: training data set as transformed by self.transform_data_set() (X_data, y_data)
        :return:
        """
        raise NotImplementedError

    def predict(self, test_set: Tuple[pd.DataFrame, np.ndarray] = None):
        """
        Test this instances model against the test set derived from the given data_set if test_set is None
        else use the given test_set.

        The size of this test set is  1 - train_fraction * len(data_set)
        :param test_set: test data set as transformed by self.transform_data_set() (X_data, y_data)
        :return:
        """
        raise NotImplementedError

    def test(self, x_data, y_data):
        return self.get_confusion_matrix(y_data, self.predict(x_data))

    def get_confusion_matrix(self, y_data, y_predicted) -> np.ndarray:
        cm = confusion_matrix(y_data, y_predicted)
        return cm

    def get_classification_report(self, x_data, y_data):
        report = classification_report(y_data, self.predict(x_data))
        return report

    def display_confusion_matrix(self, test_set=None, plotting: bool = False, save_to_file: bool = False, save_name_postfix: str = ""):
        if test_set is None:
            test_set = self.test_data
        x_test, y_test = test_set

        # classification_report
        report = self.get_classification_report(x_test, y_test)
        print(report)
        if save_to_file:
            save_name_postfix.replace(" ", "_")
            file_path = os.path.join(os.getcwd(), "output", f"tsf_{self.data_set.name}_{save_name_postfix}")
            file = open(file_path+".txt", "w")
            file.write(str(report))
        # confusion matrix plot
        cm: np.ndarray = self.test(x_test, y_test)
        print(cm)
        if plotting:
            disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sepsis", "Sepsis"])
            disp.plot()
            # funktioniert leider nicht mit title
            # temp_text_obj = Text(x=100, y=50, text=f"Confusion Matrix for version: {version}")
            # disp.ax_.title = temp_text_obj
            if save_to_file:
                print(f"Saving plot to file \"{file_path}\"")
                plt.savefig(file_path+".png")
            else:
                plt.show()
