from typing import List
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn import cluster
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from objects.training_set import TrainingSet
from IO.data_reader import FIGURE_OUTPUT_FOLDER
from tools.pacmap_analysis import plot_pacmap2D, calculate_pacmap

def implement_bicluster(training_set):
    calculate_bicluster(training_set)


def calculate_bicluster(training_set):
    avg_np = training_set_to_data(training_set)
    X = avg_np
    model = cluster.SpectralCoclustering(n_clusters=2)          # 2 = bicluster?
    model.fit(X)
    fit_X = X[np.argsort(model.row_labels_)]
    fit_X = fit_X[:, np.argsort(model.column_labels_)]

    plt.matshow(fit_X, cmap="cool")


def training_set_to_data(training_set: TrainingSet, use_interpolation: bool = False) -> np.ndarray:
    avg_df = training_set.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation)
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    return avg_np
