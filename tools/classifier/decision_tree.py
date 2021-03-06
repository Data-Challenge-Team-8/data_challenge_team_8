import matplotlib.pyplot as plt
from typing import List

from matplotlib.text import Text
from numpy import ndarray
from pandas import DataFrame
from sklearn import tree
from sklearn.metrics import silhouette_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

import tools
from classifier.decisiontree.decisiontree import DecisionTree
from tools.imbalance_methods import get_near_miss_for_split_data, get_smote_for_split_data


def implement_decision_tree(avg_df, sepsis_df):
    """
    Used for the actual implementation of decision tree classification.
    """
    clf = DecisionTree()
    x_train, x_test, y_train, y_test = train_test_split(avg_df.transpose(), sepsis_df, test_size=0.2, random_state=1337)
    clf.train(x_data=x_train, y_data=y_train)
    print("Classification Report for complete (imbalanced) dataset:")
    display_confusion_matrix(clf, x_test, y_test, plotting=True, version="complete dataset")
    clf.plot_tree(max_depth=6)

    # Oversampling: SMOTE
    x_train_smote, y_train_smote = get_smote_for_split_data(x_matrix=x_train, y_label=y_train)
    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(x_train_smote, y_train_smote, test_size=0.2,
                                                                        random_state=1337)
    clf.train(x_train_smote, y_train_smote)
    print("Classification Report for SMOTE oversampling")
    display_confusion_matrix(clf, new_x_test, new_y_test, plotting=True, version="SMOTE oversampling")
    clf.plot_tree(max_depth=12)

    # Undersampling: NearMiss
    versions: List = [1, 2, 3]
    for version in versions:
        x_train_nearmiss, y_train_nearmiss = get_near_miss_for_split_data(version=version, x_matrix=x_test, y_label=y_test)
        new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(x_train_nearmiss, y_train_nearmiss, test_size=0.2,
                                                                            random_state=1337)
        clf.train(x_data=new_x_train, y_data=new_y_train)
        print("Classification Report for NearMiss Version:", version)
        display_confusion_matrix(clf, new_x_test, new_y_test, plotting=True, version=str(version))


def display_confusion_matrix(clf, x_test, y_test, version: str, plotting: bool = False):
    # confusion matrix as df (old version for print)
    cm_df: DataFrame = clf.test_df(x_test, y_test)
    print(cm_df)
    # classification_report
    report = clf.get_classification_report(x_test, y_test)
    print(report)
    # confusion matrix plot
    if plotting:
        cm: ndarray = clf.test(x_test, y_test)
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sepsis", "Sepsis"])
        disp.plot()
        # funktioniert leider nicht mit title
        # temp_text_obj = Text(x=100, y=50, text=f"Confusion Matrix for version: {version}")
        # disp.ax_.title = temp_text_obj
        plt.show()
