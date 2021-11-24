import numpy as np
from typing import Dict, List
import pandas as pd
import os
from sklearn.cluster import DBSCAN, KMeans

from objects.training_set import TrainingSet


def calculate_cluster_dbscan(training_set: TrainingSet, eps: float, min_samples: int):
    """
    density Based Spatial Clustering of Applications with Noise. Dateninstanzen der selben 'dichten' Region werden geclustert.
    dichte Region := Radius eps mit Mindestanzahl min_samples

    Based on TrainingSet.get_average_df() and the fix_missing_values flag
    :param training_set:
    :param eps: needed for DBSCAN clustering regions
    :param min_samples: needed for minimum amount of neighbors in DBSAN clustering regions
    :return: list of responding clusters in the same order as patients list
    """
    avg_df = training_set.get_average_df(fix_missing_values=True)
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    # we could use weights per label to give imputed labels less weights?
    clustering_obj = DBSCAN(eps=eps, min_samples=min_samples).fit(avg_np)
    clustering_labels_list = clustering_obj.labels_
    return clustering_labels_list


def calculate_cluster_kmeans(training_set: TrainingSet, n_clusters: int):
    """
    k-means clustering: choose amount n_clusters to calculate k centroids for these clusters

    Based on TrainingSet.get_average_df() and the fix_missing_values flag
    :param training_set:
    :param n_clusters: amount of k clusters
    :return: list of responding clusters in the same order as patients list
    """
    avg_df = training_set.get_average_df(fix_missing_values=True)
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    # we could use weights per label to give imputed labels less weights?
    kmeans_obj = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0, max_iter=350, verbose=True).fit(avg_np)
    clustering_labels_list = kmeans_obj.labels_
    return clustering_labels_list
