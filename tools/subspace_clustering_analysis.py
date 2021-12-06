from typing import Tuple
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import sklearn.base
from sklearn import cluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import consensus_score

from objects.training_set import TrainingSet
from IO.data_reader import FIGURE_OUTPUT_FOLDER


def implement_bicluster(training_set: TrainingSet, save_to_file: bool = False):
    data, model = calculate_bicluster(training_set)
    plot_biclustering(data, model, training_set, save_to_file=save_to_file)


def calculate_bicluster(training_set: TrainingSet) -> Tuple[np.ndarray, cluster.SpectralCoclustering]:
    avg_np = training_set_to_data(training_set)
    X = avg_np
    model = cluster.SpectralCoclustering(n_clusters=3, random_state=0)          # 2 = bicluster?
    model.fit(X)

    return X, model


def plot_biclustering(data_raw: np.ndarray, model: sklearn.base.BiclusterMixin, training_set: TrainingSet,
                      save_to_file: bool = False):
    """
    Sorts the data according to the biclusters and then plots the biclustering.
    :param data_raw: the unsorted data to be plotted
    :param model: Biclustering model used for getting the bicluster data
    :param training_set: training_set that contained the data for the set name
    :param save_to_file:
    :return:
    """
    data_row = data_raw[np.argsort(model.row_labels_)]
    data = data_row[:, np.argsort(model.column_labels_)]

    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(4)
    ax = fig.add_subplot()

    ax.matshow(data, cmap=plt.cm.Blues)
    ax.set_aspect(len(data[0]) / len(data))
    ax.set_xlabel('Features')
    ax.set_ylabel('Patients')
    ax.set_title(f'"{training_set.name}" after biclustering; rearranged to show biclusters')

    if not save_to_file:
        plt.show()
    else:
        plt.savefig(os.path.join(f"bicluster-{training_set.name}"))
    plt.close()


def training_set_to_data(training_set: TrainingSet, use_interpolation: bool = False) -> np.ndarray:
    avg_df = training_set.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation)
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    return avg_np
