import pandas as pd
from matplotlib import pyplot as plt
from numpy import ndarray
from pandas import DataFrame
from sklearn import tree
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix, classification_report

from classifier.classifier import Classifier


class DecisionTree(Classifier):

    def __init__(self, max_depth: int = 5, class_weight=None):
        super().__init__()
        self.__model = DecisionTreeClassifier(max_depth=max_depth, random_state=1337, class_weight=class_weight)

    def train(self, x_data, y_data):
        self.__model.fit(x_data, y_data)

    def predict(self, x_data) -> pd.DataFrame:
        return self.__model.predict(x_data)

    def test(self, x_data, y_data):
        return self.get_confusion_matrix(y_data, self.predict(x_data))

    def get_confusion_matrix(self, y_data, y_predicted) -> ndarray:
        cm = confusion_matrix(y_data, y_predicted)
        return cm

    def get_classification_report(self, x_data, y_data):
        report = classification_report(y_data, self.predict(x_data))
        return report

    def test_df(self, x_data, y_data):
        return self.get_confusion_matrix_df(y_data, self.predict(x_data))

    def get_confusion_matrix_df(self, y_data, y_predicted) -> DataFrame:
        cm = confusion_matrix(y_data, y_predicted)
        cm_df = pd.DataFrame({
            "predicts_false": {"is_false": cm[0][0], "is_true": cm[1][0]},
            "predicts_true": {"is_false": cm[0][1], "is_true": cm[1][1]}
        })
        return cm_df

    def plot_tree(self, max_depth: int = 10):
        print("Plotting the Tree:")
        tree.plot_tree(decision_tree=self.__model,
                       max_depth=max_depth,
                       # feature_names=self.__model.feature_names_in_,
                       class_names=["no_sepsis", "sepsis"],
                       filled=True)
        plt.show()
        # tree_text = tree.export_text(self.__model, feature_names=self.__model.feature_names_in_)
        # print("------ Text report --------")
        # print(tree_text)
        # print("----- END Text report ------")
