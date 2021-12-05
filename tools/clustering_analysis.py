from typing import List
import os

import matplotlib
import matplotlib.pyplot as plt
import numpy as np
from sklearn.cluster import DBSCAN, KMeans
from sklearn.metrics import silhouette_score

from objects.training_set import TrainingSet
from IO.data_reader import FIGURE_OUTPUT_FOLDER
from tools.pacmap_analysis import plot_pacmap2D, calculate_pacmap


def implement_DBSCAN(training_set, pacmap_data, patient_ids):
    """
    only used for the actual implementation of a dbscan clustering. To remove it from main.py
    """
    # DBSCAN auf z_values_df mit interpolation
    z_value_df = training_set.get_z_value_df(use_interpolation=True, fix_missing_values=True)
    z_value_np = z_value_df.transpose().to_numpy()
    z_value_np.reshape(z_value_np.shape[0], -1)

    avg_silhouettes = []
    eps_range = [0.01, 0.025, 0.05, 0.075, 0.1]
    min_samples = 5                             # we can also test this
    for eps in eps_range:
        db_scan_list, sh_score = calculate_cluster_dbscan(z_value_np, eps=eps, min_samples=min_samples)
        plot_pacmap2D(plot_title=f"DBSCAN with eps: {eps} and min_samp=5", data=z_value_np,
                      coloring=db_scan_list,
                      color_map="cool",
                      save_to_file=True)
        avg_silhouettes.append(sh_score)
    plot_sh_scores(avg_silhouettes, eps_range, title="DBSCAN silhouettes score with min_samples=5")


    # # DBSCAN auf z_values_df ohne interpolation
    # eps = 0.5
    # min_samples = 5
    # z_value_df = training_set.get_z_value_df(use_interpolation=False, fix_missing_values=False)
    # z_value_np = z_value_df.transpose().to_numpy()
    # z_value_np.reshape(z_value_np.shape[0], -1)
    # db_scan_list = calculate_cluster_dbscan(z_value_np, eps=eps, min_samples=min_samples)
    # print("Clusters found:", set(db_scan_list))
    # plot_pacmap2D(f"DBSCAN clusters ({training_set.name}) no interpol. and eps={eps} min_sampl={min_samples}_",
    #               data=pacmap_data,  # Ist pacmap_data hier korrekt?
    #               coloring=db_scan_list,
    #               color_map='tab20c',
    #               save_to_file=True)
    #
    # # DBSCAN auf pacmap-data
    # db_scan_list = calculate_cluster_dbscan(pacmap_data, eps=eps, min_samples=min_samples)
    # print("Clusters found:", set(db_scan_list))
    # plot_pacmap2D(f"DBSCAN clusters ({training_set.name}) based on pacmap_data with interpol. and eps={eps} min_sampl={min_samples}_",
    #               data=pacmap_data,
    #               coloring=db_scan_list,
    #               color_map='tab20c',
    #               save_to_file=True)


def implement_k_means(training_set, pacmap_data, patient_ids):
    """
    only used for the actual implementation of a k-means clustering. To remove it from main.py
    """
    # # k-Means without imputation before Pacmap
    amount_of_clusters = 12
    k_means_list, sh_score = calculate_cluster_kmeans(training_set_to_data(training_set), n_clusters=amount_of_clusters)
    title = f"{amount_of_clusters} k-Means clusters ({training_set.name})"

    plot_clustering_with_silhouette_score_sepsis(title, pacmap_data, sh_score=sh_score, coloring=k_means_list,
                                                 patient_ids=patient_ids, training_set=training_set,
                                                 color_map='tab20c', save_to_file=True)
    # # k-Means without imputation after Pacmap
    k_means_list, sh_score = calculate_cluster_kmeans(pacmap_data, n_clusters=amount_of_clusters)
    title = f"{amount_of_clusters} k-Means clusters ({training_set.name}) after PaCMAP"
    plot_clustering_with_silhouette_score_sepsis(title, pacmap_data, sh_score=sh_score, coloring=k_means_list,
                                                 patient_ids=patient_ids, training_set=training_set,
                                                 color_map='tab20c', save_to_file=True)
    # k-means with imputation after Pacmap
    data_imp, patient_ids = calculate_pacmap(training_set, use_interpolation=True)
    k_means_list, sh_score = calculate_cluster_kmeans(training_set_to_data(training_set, use_interpolation=True), n_clusters=amount_of_clusters)
    title = f"{amount_of_clusters} k-Means ({training_set.name}) (interpolated)"
    plot_clustering_with_silhouette_score_sepsis(title, data_imp, sh_score=sh_score, coloring=k_means_list,
                                                 patient_ids=patient_ids, training_set=training_set,
                                                 color_map='tab20c', save_to_file=True)

    # Implement silhouettes score analysis for k-means clustering
    print("\nSilhouettes Score Analysis: ")
    krange = list(range(2, 11))
    avg_silhouettes = []
    for n in krange:
        k_means_list, sh_score = calculate_cluster_kmeans(pacmap_data, n_clusters=n)
        avg_silhouettes.append(sh_score)
    plot_sh_scores(avg_silhouettes, krange)


def training_set_to_data(training_set: TrainingSet, use_interpolation: bool = False) -> np.ndarray:
    avg_df = training_set.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation)
    avg_np = avg_df.transpose().to_numpy()
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?

    return avg_np


def calculate_cluster_dbscan(avg_np: np.ndarray, eps: float, min_samples: int, use_interpolation: bool = False):
    """
    density Based Spatial Clustering of Applications with Noise. Dateninstanzen der selben 'dichten' Region werden geclustert.
    dichte Region := Radius eps mit Mindestanzahl min_samples

    Based on TrainingSet.get_average_df() and the fix_missing_values flag
    :param avg_np:
    :param eps: needed for DBSCAN clustering regions
    :param min_samples: needed for minimum amount of neighbors in DBSAN clustering regions
    :return: list of responding clusters in the same order as patients list
    """
    # we could use weights per label to give imputed labels less weights?
    clustering_obj = DBSCAN(eps=eps, min_samples=min_samples).fit(avg_np)
    clustering_labels_list = clustering_obj.labels_
    sh_score = calculate_silhouette_score(avg_np, clustering_labels_list)
    return clustering_labels_list, sh_score


def calculate_cluster_kmeans(avg_np: np.ndarray, n_clusters: int, use_interpolation: bool = False):
    """
    k-means clustering: choose amount n_clusters to calculate k centroids for these clusters

    Based on TrainingSet.get_average_df() and the fix_missing_values flag
    :param avg_np:
    :param n_clusters: amount of k clusters
    :return: list of responding clusters in the same order as patients list
    """

    # we could use weights per label to give imputed labels less weights?
    kmeans_obj = KMeans(init="k-means++", n_clusters=n_clusters, n_init=4, random_state=0, max_iter=350,
                        verbose=True).fit(avg_np)
    clustering_labels_list = kmeans_obj.labels_

    sh_score = calculate_silhouette_score(avg_np, clustering_labels_list)

    return clustering_labels_list, sh_score


def calculate_silhouette_score(avg_np: np.ndarray, clustering_list: list, use_interpolation: bool = False) -> float:
    """
    Calculates the silhouette score, a quality measurement of clustering, for a given set of data and its clustering.
    :param avg_np: transposed numpy array of training_set.get_average_df()
    :param clustering_list:
    :return:
    """
    avg_np.reshape(avg_np.shape[0], -1)  # does this have an effect?
    if len(set(clustering_list)) < 2:
        return 0
    else:
        return silhouette_score(avg_np, labels=clustering_list, metric='euclidean', random_state=0)


def plot_clustering_with_silhouette_score(plot_title: str, data: np.ndarray, sh_score: float, coloring: List[float],
                                          color_map: str, save_to_file: bool):
    """
    Plot a clustering with its silhouette score.

    Based on PaCMAP plotting methods
    :param plot_title:
    :param data:
    :param sh_score:
    :param coloring:
    :param color_map:
    :param save_to_file:
    :return:
    """
    fig = plt.figure()
    axs = fig.subplots(2, 2)
    fig.tight_layout(h_pad=2)

    axs[0, 0].set_title(plot_title)

    axs[0, 0].scatter(data[:, 0], data[:, 1], cmap=color_map, c=coloring, s=0.6, label="Patient")

    cb = fig.colorbar(matplotlib.cm.ScalarMappable(cmap=color_map,
                                                   norm=matplotlib.colors.Normalize(vmin=min(coloring), vmax=max(coloring))),
                 ax=axs[0, 0])
    cb.set_label("Clusters")
    cb.set_ticks(list(set(coloring)))

    axs[1, 0].set_title("Silhouette Score")
    axs[1, 0].set_xlim(0, 1.0)
    axs[1, 0].barh("score", sh_score)
    # plt.legend()

    if not save_to_file:
        plt.show()
    else:
        if not os.path.exists(FIGURE_OUTPUT_FOLDER):
            os.mkdir(FIGURE_OUTPUT_FOLDER)

        f = os.path.join(FIGURE_OUTPUT_FOLDER, "pacmap-" + plot_title.replace(" ", "_") + ".png")
        print(f"Saving figure \"{plot_title}\" to file {f}")
        plt.savefig(f)
    plt.close()


def plot_sh_scores(avg_silhouettes, cluster_range, title="Silhouettes Score"):
    plt.figure(dpi=100)
    plt.title = title
    plt.plot(cluster_range, avg_silhouettes)
    plt.xlabel("$k$")
    plt.ylabel("Average Silhouettes Score")
    plt.show()


def plot_clustering_with_silhouette_score_sepsis(plot_title: str, data: np.ndarray, patient_ids: List[str],
                                                 training_set: TrainingSet, sh_score: float, coloring: List[float],
                                                 color_map: str, save_to_file: bool):
    """
    Plot a clustering with its silhouette score and sepsis per cluster

    Based on PaCMAP plotting methods
    :param patient_ids:
    :param training_set:
    :param plot_title:
    :param data:
    :param sh_score: the calculated silhouette score of the entire clustering
    :param coloring: List that maps the index to a cluster, index is synchronized with data and patient_ids
    :param color_map: color map to use for coloring clusters among all plots (except silhouette score)
    :param save_to_file:
    :return:
    """
    fig = plt.figure()
    axs = fig.subplots(2, 2)
    fig.tight_layout(h_pad=2, w_pad=2)

    # Clustering
    axs[0, 0].set_title(plot_title)

    axs[0, 0].scatter(data[:, 0], data[:, 1], cmap=color_map, c=coloring, s=0.6, label="Patient")

    sm = matplotlib.cm.ScalarMappable(cmap=color_map,
                                      norm=matplotlib.colors.Normalize(vmin=min(coloring), vmax=max(coloring)))
    cb = fig.colorbar(sm, ax=axs[0, 0])
    cb.set_label("Clusters")
    cb.set_ticks(list(set(coloring)))

    # Silhouette Score
    axs[1, 0].set_title("silhouette score")
    axs[1, 0].set_xlim(0, 1.0)
    axs[1, 0].barh("score", sh_score, color=plt.get_cmap("RdYlGn").reversed()(sh_score/1))

    # Sepsis per Cluster
    sepsis_list = []
    for patient_id in patient_ids:
        sepsis_list.append(training_set.data[patient_id].sepsis_label.sum() > 0)  # if the patient has sepsis or not

    cluster_sepsis_sum = {}
    cluster_sum = {}
    for i in range(len(sepsis_list)):
        cluster = coloring[i]
        if sepsis_list[i]:  # total sum of sepsis in a cluster
            if cluster not in cluster_sepsis_sum.keys():
                cluster_sepsis_sum[cluster] = 0
            cluster_sepsis_sum[cluster] += 1

        if cluster not in cluster_sum.keys():  # total sum of cases in a cluster
            cluster_sum[cluster] = 0
        cluster_sum[cluster] += 1

    axs[1, 1].set_title("total sepsis cases per cluster")
    axs[1, 1].set_xticks(list(set(coloring)))
    for cluster in cluster_sepsis_sum.keys():

        axs[1, 1].bar(cluster, cluster_sepsis_sum[cluster],
                      color=plt.get_cmap(color_map)(cluster/max(coloring)))

    # rel Sepsis per Cluster
    axs[0, 1].set_title("relative sepsis cases per cluster")
    axs[0, 1].set_xticks(list(set(coloring)))
    axs[0, 1].set_ylim(0, 1.0)
    for cluster in cluster_sepsis_sum.keys():
        axs[0, 1].bar(cluster, cluster_sepsis_sum[cluster]/cluster_sum[cluster],
                      color=plt.get_cmap(color_map)(cluster/max(coloring)))

    # plt.legend()

    if not save_to_file:
        plt.show()
    else:
        if not os.path.exists(FIGURE_OUTPUT_FOLDER):
            os.mkdir(FIGURE_OUTPUT_FOLDER)

        f = os.path.join(FIGURE_OUTPUT_FOLDER, "pacmap-" + plot_title.replace(" ", "_") + ".png")
        print(f"Saving figure \"{plot_title}\" to file {f}")
        plt.savefig(f, bbox_inches="tight")
    plt.close()
