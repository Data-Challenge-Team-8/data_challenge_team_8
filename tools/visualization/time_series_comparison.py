from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN

from objects.training_set import TrainingSet


def dbscan_1d(values: List[int]):
    np_values = np.array(values)
    np_values = np_values.reshape(-1, 1)
    obj = DBSCAN(eps=20, min_samples=int(len(values)*0.05))
    obj.n_features_in_ = 1
    clustering_list = obj.fit(np_values).labels_
    return clustering_list


def count_occurences(values: List) -> dict:
    occurences = {}
    for value in values:
        if value in occurences.keys():
            occurences[value] += 1
        else:
            occurences[value] = 1

    return occurences


def plot_time_series_density(series_data, label: str, set_name: str):
    fig, axs = plt.subplots(2)

    patient_has_values = []
    for patient_data in series_data:
        axs[0].plot(patient_data)

        for i in range(len(patient_data)):
            if not np.isnan(patient_data[i]):
                patient_has_values.append(i)

    counts = [0 for i in range(series_data.shape[1])]
    for val in patient_has_values:
        counts[val] += 1

    # start clustering
    clustering_list = dbscan_1d(counts)
    cluster_sizes_dict = count_occurences(clustering_list)
    cluster_sizes = list(cluster_sizes_dict.items())
    cluster_sizes.sort(key=lambda x: x[1], reverse=True)
    greatest_cluster = cluster_sizes[0]
    cluster_value = {}
    for i in range(len(counts)):
        if clustering_list[i] in cluster_value.keys():
            cluster_value[clustering_list[i]] += counts[i]
        else:
            cluster_value[clustering_list[i]] = counts[i]
    print("Cluster Values:", cluster_value)
    sum_cluster_value = sum(cluster_value.values())
    print("Sum Cluster Value", sum_cluster_value)
    for k in cluster_value.keys():
        print(f"{cluster_value[k]} /= {cluster_sizes_dict[k]}")
        cluster_value[k] /= cluster_sizes_dict[k]

    print("Greatest Cluster:", greatest_cluster)
    print("Cluster sizes:", cluster_sizes)
    #print("Cluster list:", clustering_list)
    print("rel. Cluster Value:", cluster_value)
    rel_cl_values = list(cluster_value.items())
    rel_cl_values.sort(key=lambda x: x[1])
    print("Cluster to discard:", rel_cl_values[0])

    print("Labeling:", [f"Cluster #{i}" for i in clustering_list])
    # end clustering

    axs[0].set_title(f"Time series data ({label}, {set_name})")
    axs[1].set_title("Count of Values at Index (DBSCAN)")

    cnts, values, bars = axs[1].hist(patient_has_values, bins=series_data.shape[1])
    col = ['aqua', 'red', 'gold', 'royalblue', 'darkorange', 'green', 'purple', 'cyan', 'yellow', 'lime']
    for i, (cnt, clue, bar) in enumerate(zip(cnts, values, bars)):
        bar.set_facecolor(col[clustering_list[i] % len(col)])

    axs[1].legend()

    fig.tight_layout()

    plt.show()


def plot_complete_time_series_for_patients(training_set: TrainingSet, limit_to_features: List[str], plot_maximum: int):
    sepsis_df = training_set.get_sepsis_label_df()
    plot_counter = 0
    for patient_id in training_set.data:
        if training_set.check_patient_has_sepsis(sepsis_df, patient_id):
            temp_patient = training_set.get_patient_from_id(patient_id)
            temp_patient.plot_features_time_series(limit_to_features)
            plot_counter += 1
            if plot_counter > plot_maximum:
                break

def plot_reduced_time_series_data(training_set, time_series):
    """ This method is used to plot the reduced timeseries data (only 40 timesteps) for all patients
    that have sepsis. You can select the feature which you wish to compare to the SepsisLabel. """
    sepsis_df = training_set.get_sepsis_label_df()
    temp_hr = None
    for patient_feature_tuple in time_series.transpose():
        if training_set.check_patient_has_sepsis(sepsis_df, patient_feature_tuple[0]):
            if patient_feature_tuple[1] == "HR":  # select desired feature here
                temp_hr = time_series.loc[patient_feature_tuple]
            elif patient_feature_tuple[1] == "SepsisLabel":
                temp_sepsis = time_series.loc[patient_feature_tuple]
                try:
                    if temp_hr.name[0] == temp_sepsis.name[0]:
                        # plot_two_time_series(patient_id=patient_feature_tuple[0],
                        #                      series_one=temp_hr,
                        #                      label_one="HR",
                        #                      series_two=temp_sepsis,
                        #                      label_two="SepsisLabel")
                        plot_time_series_sepsis_background(patient_id=patient_feature_tuple[0],
                                                           series_one=temp_hr,
                                                           label_one="HR",
                                                           series_two=temp_sepsis,
                                                           label_two="SepsisLabel")
                except AttributeError:
                    pass
            else:
                pass
        else:
            pass


def plot_two_time_series(patient_id: str, series_one, label_one: str, series_two, label_two: str):
    fig, axs = plt.subplots(2)
    # vlt auch für mehr als zwei features bauen?

    axs[0].set_title(f"Time series data ({label_one}, {patient_id})")
    axs[0].plot(series_one)
    axs[1].set_title(f"Time series data ({label_two}, {patient_id})")
    axs[1].plot(series_two)

    fig.tight_layout()
    plt.show()

def plot_time_series_sepsis_background(patient_id, series_one, label_one, series_two, label_two):
    fig, axs = plt.subplots(1)
    # vlt auch für mehr als zwei features bauen?

    axs.set_title(f"Time series data ({label_one}, {patient_id})")
    axs.plot(series_one)

    temp_df = series_two.to_frame()
    sepsis_indices = temp_df.loc[(temp_df[series_two.name] >= 1)]             # careful: series_two_name is not "SepsisLabel" but "<patient_id>_SepsisLabel"
    highlight(sepsis_indices.index, axs)

    fig.tight_layout()
    plt.show()

def highlight(indices, ax):
    i = 0
    try:
        while i < len(indices):
            ax.axvspan(indices[i]-0.5, indices[i]+0.5, facecolor='pink', edgecolor='none', alpha=.2)
            i += 1
    except KeyError:            # this happens if indices is empty
        pass
