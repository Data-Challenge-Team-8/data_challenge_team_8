import pandas as pd
from sklearn.tree import DecisionTreeClassifier
from sklearn.metrics import confusion_matrix

from classifier.classifier import Classifier


class DecisionTree(Classifier):

    def __init__(self, max_depth: int = 5, class_weight=None):
        super().__init__()
        self.__model = DecisionTreeClassifier(max_depth=max_depth, random_state=1337, class_weight=class_weight)

    def train(self, x_data, y_data, max_depth: int = 5):
        self.__model.fit(x_data, y_data)

    def predict(self, x_data) -> pd.DataFrame:
        return self.__model.predict(x_data)

    def test(self, x_data, y_data):
        return self.get_confusion_matrix(y_data, self.predict(x_data))

    def get_confusion_matrix(self, y_data, y_predicted):
        cm = confusion_matrix(y_data, y_predicted)
        cm_df = pd.DataFrame({
            "predicts_false": {"is_false": cm[0][0], "is_true": cm[0][1]},
            "predicts_true": {"is_false": cm[1][0], "is_true": cm[1][1]}
        })
        return cm_df
