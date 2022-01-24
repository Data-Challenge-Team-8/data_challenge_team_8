from typing import List
import matplotlib.pyplot as plt
from matplotlib import cm, colors
import numpy as np
import pandas as pd
from sklearn.cluster import DBSCAN


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
