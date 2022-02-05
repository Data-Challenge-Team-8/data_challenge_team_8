import pandas as pd
import numpy as np
from sklearn.ensemble import RandomForestClassifier
from sklearn.metrics import confusion_matrix, classification_report, roc_auc_score, plot_roc_curve
from classifier.classifier import Classifier


class RandomForest(Classifier):

    def __init__(self, max_depth: int = 5, tree_count: int = 100, use_bootstrapping: bool = True):
        super().__init__()
        self.__model = RandomForestClassifier(n_estimators=tree_count, max_depth=max_depth, bootstrap=use_bootstrapping,
                                              random_state=1337)

    def train(self, x_data, y_data):
        self.__model.fit(x_data, y_data)

    def predict(self, x_data) -> pd.DataFrame:
        return self.__model.predict(x_data)

    def test(self, x_data, y_data):
        return self.get_confusion_matrix(y_data, self.predict(x_data))

    def get_confusion_matrix(self, y_data, y_predicted) -> np.ndarray:
        cm = confusion_matrix(y_data, y_predicted)
        return cm

    def get_classification_report(self, x_data, y_data):
        report = classification_report(y_data, self.predict(x_data))
        return report

    def test_df(self, x_data, y_data):
        return self.get_confusion_matrix_df(y_data, self.predict(x_data))

    def get_confusion_matrix_df(self, y_data, y_predicted) -> pd.DataFrame:
        cm = confusion_matrix(y_data, y_predicted)
        cm_df = pd.DataFrame({
            "predicts_false": {"is_false": cm[0][0], "is_true": cm[1][0]},
            "predicts_true": {"is_false": cm[0][1], "is_true": cm[1][1]}
        })
        return cm_df

    def plot_roc_curve(self, X_test, y_test, title: str = ""):
        return plot_roc_curve(self.__model, X_test, y_test, name=title)

    def get_roc_auc_score(self, X_test, y_test):
        roc = roc_auc_score(y_test, self.__model.predict_proba(X_test)[:, 1])
        return roc

