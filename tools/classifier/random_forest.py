import matplotlib.pyplot as plt
from typing import List

from matplotlib.text import Text
from numpy import ndarray
from pandas import DataFrame
from sklearn.metrics import silhouette_score, ConfusionMatrixDisplay, classification_report
from sklearn.model_selection import train_test_split
from imblearn.over_sampling import SMOTE
from imblearn.under_sampling import NearMiss

from classifier.decisiontree.random_forest import RandomForest
from objects.training_set import TrainingSet


def implement_random_forest(set: TrainingSet, use_interpolation: bool = True, fix_missing_values: bool = True):
    """
    Used for the actual implementation of decision tree classification.
    """
    avg_df = set.get_average_df(use_interpolation=use_interpolation, fix_missing_values=fix_missing_values)
    sepsis_df = set.get_sepsis_label_df()
    clf = RandomForest()
    x_train, x_test, y_train, y_test = train_test_split(avg_df.transpose(), sepsis_df, test_size=0.2, random_state=1337)
    clf.train(x_data=x_train, y_data=y_train)
    print("Classification Report for complete (imbalanced) dataset:")
    display_confusion_matrix(clf, x_test, y_test, plotting=True, version="complete dataset", set=set)
    display_roc_auc_curve(clf,x_test, y_test, version="complete dataset", plotting=True, set=set)

    # Oversampling: SMOTE
    smote = SMOTE(random_state=1337)
    x_train_smote, y_train_smote = smote.fit_resample(x_train, y_train)
    new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(x_train_smote, y_train_smote, test_size=0.2,
                                                                        random_state=1337)
    clf.train(x_train_smote, y_train_smote)
    print("Classification Report for SMOTE oversampling")
    display_confusion_matrix(clf, new_x_test, new_y_test, plotting=True, version="SMOTE oversampling", set=set)
    display_roc_auc_curve(clf, new_x_test, new_y_test, version="SMOTE oversampling", plotting=True, set=set)

    # Undersampling: NearMiss
    versions: List = [1, 2, 3]
    for version in versions:
        near_miss = NearMiss(version=version)
        new_x, new_y = near_miss.fit_resample(x_test, y_test)
        new_x_train, new_x_test, new_y_train, new_y_test = train_test_split(new_x, new_y, test_size=0.2,
                                                                            random_state=1337)
        clf.train(x_data=new_x_train, y_data=new_y_train)
        print("Classification Report for NearMiss Version:", version)
        display_confusion_matrix(clf, new_x_test, new_y_test, plotting=True, version="NearMiss"+str(version), set=set)
        display_roc_auc_curve(clf, new_x_test, new_y_test, version="NearMiss"+str(version), plotting=True, set=set)


def display_confusion_matrix(clf, x_test, y_test, version: str, set: TrainingSet, plotting: bool = False):
    # confusion matrix as df (old version for print)
    cm_df: DataFrame = clf.test_df(x_test, y_test)
    print(cm_df)
    # classification_report
    report = clf.get_classification_report(x_test, y_test)
    print(report)
    # confusion matrix plot
    if plotting:
        cm: ndarray = clf.test(x_test, y_test)
        fig, ax = plt.subplots()
        disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=["No Sepsis", "Sepsis"])
        disp.plot(ax=ax)
        ax.set_title(f"RandomForest {version} {set.name}")
        plt.show()


def display_roc_auc_curve(clf, x_test, y_test, version: str, set: TrainingSet, plotting: bool = False):
    auc = clf.get_roc_auc_score(x_test, y_test)
    fig, ax = plt.subplots()
    disp = clf.plot_roc_curve(x_test, y_test, title=f"RF {version}")
    if plotting:
        disp.plot(ax=ax)
        ax.set_title(f"RandomForest {version} ({set.name} AUC: {round(auc, 4)})")
        plt.show()
