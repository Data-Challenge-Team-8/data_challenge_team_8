import pickle

import numpy as np
from typing import Dict, List

from pandas import DataFrame

from IO.data_reader import FIGURE_OUTPUT_FOLDER

import os
import pacmap
from matplotlib import pyplot as plt
import matplotlib


def training_set_to_data(training_set, use_interpolation: bool = False) -> np.ndarray:
    avg_df = training_set.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation)
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    return avg_np


def calculate_pacmap(training_set, dimension: int = 2, use_interpolation: bool = False):
    """
    Calculate a PaCMAP transformation of the given TrainingSet.

    Based on TrainingSet.get_average_df() and the fix_missing_values flag
    :param use_interpolation:
    :param avg_np:
    :param dimension: dimension of the resulting PaCMAP mapping
    :return: data as returned by the PaCMAP algorithm
    """
    avg_df = training_set.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation)
    avg_np = avg_df.transpose().to_numpy()
    # avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    embedding = pacmap.PaCMAP(n_dims=dimension, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, verbose=True, random_state=1)
    data_transformed = embedding.fit_transform(avg_np, init="pca")

    patient_ids = avg_df.columns.tolist()

    return data_transformed, patient_ids

def calculate_pacmap_on_avg_df(avg_df: DataFrame, dimension: int = 2):
    """
    Calculate a PaCMAP transformation of the given avg_df.

    Based on TrainingSet.get_average_df()
    :param avg_df:
    :param dimension: dimension of the resulting PaCMAP mapping
    :return: data as returned by the PaCMAP algorithm
    """
    avg_np = avg_df.transpose().to_numpy()
    # avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    embedding = pacmap.PaCMAP(n_dims=dimension, n_neighbors=None, MN_ratio=0.5, FP_ratio=2.0, verbose=True, random_state=1)
    data_transformed = embedding.fit_transform(avg_np, init="pca")

    patient_ids = avg_df.columns.tolist()

    return data_transformed, patient_ids


def plot_pacmap2D_sepsis(plot_title: str, data: np.ndarray, patient_ids: List[str], training_set, save_to_file: bool = False):
    """
    Plots the given 2D PaCMAP data using matplotlib with sepsis coloring
    :param patient_ids: ordering is expected to be in sync with data
    :param plot_title:
    :param data: result of pacmap calculation
    :param training_set: TrainingSet for access to additional Patient data
    :param save_to_file:
    :return:
    """
    sepsis_list = []
    try:
        for patient_id in patient_ids:
            sepsis_list.append(training_set.data[patient_id].sepsis_label.sum() > 0)  # if the patient has sepsis or not
    except KeyError:
        pass
    plot_pacmap2D(plot_title, data, sepsis_list, color_map="cool", save_to_file=save_to_file)


def plot_pacmap3D_sepsis(plot_title: str, data: np.ndarray, patient_ids: List[str], training_set, save_to_file: bool = False):
    """
    Plots the given 3D PaCMAP data using matplotlib with sepsis coloring
    :param plot_title:
    :param data:
    :param patient_ids:
    :param training_set:
    :param save_to_file:
    :return:
    """
    sepsis_list = []
    for patient_id in patient_ids:
        sepsis_list.append(training_set.data[patient_id].sepsis_label.sum() > 0)  # if the patient has sepsis or not

    plot_pacmap3D(plot_title, data, sepsis_list, color_map="cool", save_to_file=save_to_file)


def plot_pacmap2D(plot_title: str, data: np.ndarray, coloring: List[float], color_map: str, save_to_file: bool):
    """
    Plots the given 2D PaCMAP data using matplotlib
    :param coloring: list of floats to be used for coloring with the color map
    :param color_map: string name of the color map to use. See matplotlib for options
    :param plot_title:
    :param data: result of pacmap calculation
    :param save_to_file:
    :return:
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1)
    ax1.set_title(plot_title)

    ax1.scatter(data[:, 0], data[:, 1], cmap=color_map, c=coloring, s=0.6, label="Patient")

    plt.legend()
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=color_map,
                                              norm=matplotlib.colors.Normalize(vmin=min(coloring),
                                                                               vmax=max(coloring))),
                                                                                ax=ax1)

    if not save_to_file:
        plt.show()
    else:
        if not os.path.exists(FIGURE_OUTPUT_FOLDER):
            os.mkdir(FIGURE_OUTPUT_FOLDER)

        f = os.path.join(FIGURE_OUTPUT_FOLDER, "pacmap-" + plot_title.replace(" ", "_") + ".png")
        print(f"Saving figure \"{plot_title}\" to file {f}")
        plt.savefig(f)
    plt.close()


def plot_pacmap3D(plot_title: str, data: np.ndarray, coloring: List[float], color_map: str, save_to_file: bool):
    """
    Plots the given 3D PaCMAP data using matplotlib with sepsis coloring
    :param coloring: list of floats to be used for coloring with the color map
    :param color_map: string name of the color map to use. See matplotlib for options
    :param plot_title:
    :param data: result of pacmap calculation
    :param save_to_file:
    :return:
    """
    fig = plt.figure()
    ax1 = fig.add_subplot(1, 1, 1, projection='3d')
    ax1.set_title(plot_title)

    ax1.scatter(data[:, 0], data[:, 1], data[:, 2], cmap=color_map, c=coloring, s=0.6, label="Patient")

    plt.legend()
    fig.colorbar(matplotlib.cm.ScalarMappable(cmap=color_map,
                                              norm=matplotlib.colors.Normalize(vmin=min(coloring), vmax=max(coloring))),
                 ax=ax1)

    if not save_to_file:
        plt.show()
    else:
        if not os.path.exists(FIGURE_OUTPUT_FOLDER):
            os.mkdir(FIGURE_OUTPUT_FOLDER)

        f = os.path.join(FIGURE_OUTPUT_FOLDER, "pacmap-" + plot_title.replace(" ", "_") + ".png")
        print(f"Saving figure \"{plot_title}\" to file {f}")
        plt.savefig(f)
    plt.close()
