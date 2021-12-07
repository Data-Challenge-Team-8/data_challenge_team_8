import numpy as np

import os
import pacmap
import seaborn as sb
from matplotlib import pyplot as plt
from typing import Dict, List

from objects.patient import Patient
from objects.training_set import TrainingSet
from IO.data_reader import FIGURE_OUTPUT_FOLDER


def get_and_plot_sepsis_correlation(training_set, fix_missing_values=True, use_interpolation=True):
    avg_df = training_set.get_average_df(fix_missing_values=fix_missing_values, use_interpolation=use_interpolation)       # TODO: unterschiedliche df testen
    avg_df_corr = avg_df.transpose().corr()
    # feature_names = np.argsort(avg_df_corr.columns)         # feature_names: [ 6  4  7  0  8  9  3  1  5  2 10]
    feature_names = avg_df_corr.columns
    avg_df_corr_without_nan = avg_df_corr.fillna(0)       # Aus irgend einem grund ist EtCO2 NaN
    sepsis_corr = avg_df_corr_without_nan["SepsisLabel"]
    sorted_sepsis_corr = sepsis_corr.sort_values(ascending=False)

    # Plotten von Correlation zu SepsisLabel
    # print("Sepsis Correlation:")
    plot_sepsis_corr = sorted_sepsis_corr.drop("SepsisLabel")
    ax1 = plot_sepsis_corr.plot.bar(x='Features')
    plt.title(f"Correlation to SepsisLabel, fixed values={fix_missing_values}")
    plt.xticks(rotation=-45)
    plt.show()

    # # Heatmap über alle Labels
    # # print("Heatmap:")
    ax2 = sb.heatmap(data=avg_df_corr_without_nan.to_numpy(), vmin=-1, vmax=1, linewidths=0.5,
                     cmap='bwr', yticklabels=feature_names)
    plt.title(f"Correlations in {training_set.name}, fixed values={fix_missing_values}")
    plt.show()

    # Pairplot von ausgewählten Labels zu Sepsis und zueinander
    important_features = sorted_sepsis_corr.index[:3].tolist()
    important_features.extend(sorted_sepsis_corr.index[-3:].tolist())
    selected_labels_df = avg_df.transpose().filter(important_features, axis=1)
    avg_df_small = selected_labels_df.iloc[:100]                                     # scatter plot nur 100 patients
    sb.set_style('darkgrid')
    pairplot = sb.pairplot(avg_df_small)
    plt.show()
    return sorted_sepsis_corr


def training_set_to_data(training_set: TrainingSet, use_interpolation: bool = False) -> np.ndarray:
    avg_df = training_set.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation)
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    return avg_np


# def plot_correlation(plot_title: str, data: np.ndarray, patient_ids: List[str], training_set: TrainingSet, save_to_file: bool = False):
#     """
#     Plots the given 2D PaCMAP data using matplotlib with sepsis coloring
#     :param patient_ids: ordering is expected to be in sync with data
#     :param plot_title:
#     :param data: result of pacmap calculation
#     :param training_set: TrainingSet for access to additional Patient data
#     :param save_to_file:
#     :return:
#     """
#     sepsis_list = []
#     for patient_id in patient_ids:
#         sepsis_list.append(training_set.data[patient_id].sepsis_label.sum() > 0)  # if the patient has sepsis or not
#
#     plot_pacmap2D(plot_title, data, sepsis_list, color_map="cool", save_to_file=save_to_file)
