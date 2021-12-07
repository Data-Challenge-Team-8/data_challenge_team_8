from typing import Tuple
import math
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import sklearn.base
from sklearn import cluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import consensus_score

from objects.training_set import TrainingSet
from IO.data_reader import FIGURE_OUTPUT_FOLDER


def implement_bicluster_spectralcoclustering(training_set: TrainingSet, save_to_file: bool = False, use_interpolation: bool = False):
    data, model, avg_df = calculate_bicluster_spectralcoclustering(training_set, use_interpolation=use_interpolation)
    plot_biclustering(data, model, avg_df, training_set, save_to_file=save_to_file, feature_names=avg_df.index.to_numpy(),
                      used_interpolation=use_interpolation)

    return model.row_labels_, model.column_labels_


def implement_bicluster_spectralbiclustering(training_set: TrainingSet, save_to_file: bool = False,
                                             use_interpolation: bool = False):
    data, model, avg_df = calculate_bicluster_spectralbiclustering(training_set, use_interpolation=use_interpolation)
    plot_biclustering(data, model, avg_df, training_set, save_to_file=save_to_file, feature_names=avg_df.index.to_numpy(),
                      used_interpolation=use_interpolation)

    return model.row_labels_, model.column_labels_


def calculate_bicluster_spectralcoclustering(training_set: TrainingSet, use_interpolation: bool = False) -> Tuple[np.ndarray, cluster.SpectralCoclustering, pd.DataFrame]:
    avg_np, avg_df = training_set_to_data(training_set, use_interpolation=use_interpolation)
    X = avg_np
    model = cluster.SpectralCoclustering(n_clusters=3, random_state=0)
    model.fit(X)

    return X, model, avg_df


def calculate_bicluster_spectralbiclustering(training_set: TrainingSet, use_interpolation: bool = False) -> Tuple[np.ndarray, cluster.SpectralBiclustering, pd.DataFrame]:
    avg_np, avg_df = training_set_to_data(training_set, use_interpolation=use_interpolation)
    X = avg_np
    model = cluster.SpectralBiclustering(n_clusters=3, random_state=0)
    model.fit(X)

    return X, model, avg_df


def plot_biclustering(data_raw: np.ndarray, model: sklearn.base.BiclusterMixin, avg_df: pd.DataFrame,
                      training_set: TrainingSet, save_to_file: bool = False, used_interpolation: bool = False,
                      feature_names: np.ndarray = None):
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
    feature_names = feature_names[np.argsort(model.column_labels_)]

    clustering_type = ""  # for getting the biclustering method recorded
    if isinstance(model, cluster.SpectralCoclustering):
        clustering_type = "SpectralCoclustering"
    elif isinstance(model, cluster.SpectralBiclustering):
        clustering_type = "SpectralBiclustering"

    fig = plt.figure()
    fig.set_figheight(4)
    fig.set_figwidth(4)

    ax = plt.subplot2grid((10, 8), (0, 0), 7, 7)
    ax1 = plt.subplot2grid((10, 8), (8, 0), 2, 2)
    ax2 = plt.subplot2grid((10, 8), (8, 3), 2, 2)
    ax3 = plt.subplot2grid((10, 8), (8, 6), 2, 2)

    ax.matshow(data, cmap=plt.cm.Blues)
    ax.set_aspect(len(data[0]) / len(data))  # this makes the plot quadratic and prevents extreme stretching
    ax.set_xlabel('Features')
    ax.set_ylabel('Patients')

    if feature_names is not None:
        ax.set_xticklabels(labels=list(feature_names))
        props = {"rotation": 90}
        plt.setp(ax.get_xticklabels(), **props)
    ax.set_xticks([i for i in range(len(data[0]))])

    if not used_interpolation:
        ax.set_title(f'"{training_set.name}" after biclustering; rearranged to show biclusters \n'
                     f'({clustering_type})')
    else:
        ax.set_title(f'"{training_set.name}" after biclustering; rearranged to show biclusters \n'
                     f'({clustering_type}, interpolation, quadratic)')

    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=plt.cm.Blues,
                                              norm=matplotlib.colors.Normalize(vmin=min([min(v) for v in data]),
                                                                               vmax=max([max(v) for v in data]))),
                 ax=ax)

    # Tfeature, target & demographics data per cluster
    sepsis_per_row_cluster = [[] for y in range(max(model.row_labels_)+1)]
    gender_per_row_cluster = [[] for y in range(max(model.row_labels_)+1)]
    age_per_row_cluster = [[] for y in range(max(model.row_labels_)+1)]

    for i in range(len(avg_df.columns)):
        sepsis_per_row_cluster[model.row_labels_[i]].append(training_set.data[avg_df.columns[i]].sepsis_label.max())
        gender_per_row_cluster[model.row_labels_[i]].append(training_set.data[avg_df.columns[i]].gender.max())
        age_per_row_cluster[model.row_labels_[i]].append(training_set.data[avg_df.columns[i]].age.max())

    for cluster_nr in range(max(model.row_labels_)+1):
        ax1.bar(cluster_nr + 0.00, sum(sepsis_per_row_cluster[cluster_nr])/len(sepsis_per_row_cluster[cluster_nr]),
                color='g', label="Sepsis", width=0.4)
        ax1.bar(cluster_nr + 0.40, sum(gender_per_row_cluster[cluster_nr])/len(gender_per_row_cluster[cluster_nr]),
                color='b', label="Gender", width=0.4)

        ax2.bar(cluster_nr + 0.00, sum(age_per_row_cluster[cluster_nr]) / len(age_per_row_cluster[cluster_nr]))
        ax3.bar(cluster_nr, len(sepsis_per_row_cluster[cluster_nr]))

    ax1.set_xticks([x + 0.2 for x in range(max(model.row_labels_)+1)])
    ax1.set_xticklabels(labels=[f"Row Cluster #{x}" for x in range(max(model.row_labels_)+1)])
    props = {"rotation": 45}
    plt.setp(ax1.get_xticklabels(), **props)

    ax2.set_xticks([x + 0.2 for x in range(max(model.row_labels_)+1)])
    ax2.set_xticklabels(labels=[f"Row Cluster #{x}" for x in range(max(model.row_labels_) + 1)])
    props = {"rotation": 45}
    plt.setp(ax2.get_xticklabels(), **props)

    ax1.set_title("Relative Sepsis & Avg. Gender")
    ax2.set_title("Average Age")
    ax3.set_title("Total row cluster size")

    fig.tight_layout()
    if not save_to_file:
        plt.show()
    else:
        if used_interpolation:
            plt.savefig(os.path.join(FIGURE_OUTPUT_FOLDER, f"bicluster-{clustering_type}-{training_set.name}-interpolated.png"))
        else:
            plt.savefig(os.path.join(FIGURE_OUTPUT_FOLDER, f"bicluster-{clustering_type}-{training_set.name}.png"))
    plt.close()


def training_set_to_data(training_set: TrainingSet, use_interpolation: bool = False) -> Tuple[np.ndarray, pd.DataFrame]:
    avg_df = training_set.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation)
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    return avg_np, avg_df
