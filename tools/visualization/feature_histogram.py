import math
import os.path
from typing import List
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd

from objects.training_set import TrainingSet
from IO.data_reader import FIGURE_OUTPUT_FOLDER


def plot_histogram_count(set: TrainingSet, features: List[str], label: str = "SepsisLabel", n_bins: int = None,
                         show_plot: bool = True, save_to_file: bool = False):
    """
    Plot histograms with all features as x-axis and the amount of label at the X as the height of the bar.
    :param set:
    :param features:
    :param label:
    :param n_bins: number of bins used, default None, if None the length of the current feature is used
    :param show_plot: flag if the plot will be showed by matplotlib
    :param save_to_file: flag if the plot will be saved to file
    :return:
    """
    fig, axs = plt.subplots(len(features))
    if not isinstance(axs, list) and not isinstance(axs, np.ndarray):
        axs = [axs]

    label_data = set.get_feature(label)
    for i in range(len(features)):
        feature_data = set.get_feature(features[i])
        feature_series = feature_data.mean()

        n_bins = n_bins if n_bins is not None else len(range(int(feature_data.max().max()+1)))
        label_counts = []
        feature_counts = []
        for k in range(len(feature_series)):
            if np.isnan(feature_series[k]):
                continue
            if label_data.loc[feature_series.index[k]] != 0:
                label_counts.append(int(feature_series[k]))
            feature_counts.append(int(feature_series[k]))

        print("feature counts", feature_counts)
        print("label counts", label_counts)
        print("n_bins", n_bins)
        axs[i].hist(feature_counts, bins=n_bins//2, label=features[i])
        axs[i].hist(label_counts, bins=n_bins//2, label=label)
        #axs[i].grid()
        axs[i].set_title(f"\"{label}\" distribution over \"{features[i]}\"")
        axs[i].legend()

    plt.show()
    plt.clf()


def plot_sepsis_distribution(sepsis_df: pd.DataFrame, set_name: str, title_postfix: str = "", save_to_file: bool = False):
    """
    Plot the distribution of sepsis across the given data
    :param sepsis_df:
    :param set_name:
    :param title_postfix:
    :return:
    """
    one_count = sepsis_df.sum()[0]
    zero_count = sepsis_df[sepsis_df == 0].count()[0]

    fig, ax = plt.subplots()
    absolute_sum = one_count + zero_count
    avg = absolute_sum / 2

    color = plt.get_cmap("RdYlGn").reversed()
    ax.bar([0, 1], [zero_count, one_count],
           color=[color(abs(value - avg) / (absolute_sum-avg)) for value in [zero_count, one_count]])
    ax.set_xticks([0, 1])
    ax.hlines([avg], label="Average", xmin=ax.get_xlim()[0], xmax=ax.get_xlim()[1], color='b', linestyles='dashed')
    ax.legend()

    title = f"Distribution of \"SepsisLabel\" across \"{set_name}\""+" "+title_postfix
    ax.set_title(title)

    if not save_to_file:
        plt.show()
    else:
        file_path = os.path.join(FIGURE_OUTPUT_FOLDER, title.lower().replace(" ", "_").replace("\"", ""))
        print(f"Saving figure to: {file_path}")
        plt.savefig(file_path)
    plt.clf()



