from objects.training_set import TrainingSet
from tools.visualization.interpolation_comparison import plot_most_interesting_interpolation_patients, \
    plot_data_with_and_without_interpolation
from tools.pacmap_analysis import calculate_pacmap
from tools.clustering_analysis import calculate_cluster_dbscan, calculate_cluster_kmeans, calculate_silhouette_score, \
    plot_clustering_with_silhouette_score_sepsis, training_set_to_data


if __name__ == '__main__':
    #mini_set = TrainingSet(TrainingSet.PRESETS["Set A"][:61], name="Mini Set")

    # Load Trainingset (with interpolation and caching)
    set_a = TrainingSet.get_training_set("Set A")
    # plot_most_interesting_interpolation_patients(set_a)

    # Pacmap for Sepsis
    # TODO: Können wir das Pacmap data cachen? Dann gehen die ganzen Clusterings viel schneller.
    data, patient_ids = calculate_pacmap(set_a)

    # Calculate different clusters
    # TODO: Wir brauchen eigentlich standardisierte Daten: z = (wert_i - mean) / std_dev
    # db_scan_list = calculate_cluster_dbscan(set_a, eps=0.05, min_samples=200)
    # plot_pacmap2D(f"PaCMAP colored by DBSCAN clusters ({set_a.name})", data, coloring=db_scan_list, color_map='tab20c', save_to_file=True)

    amount_of_clusters = 12
    k_means_list, sh_score = calculate_cluster_kmeans(training_set_to_data(set_a), n_clusters=amount_of_clusters)
    title = f"{amount_of_clusters} k-Means clusters ({set_a.name})"

    plot_clustering_with_silhouette_score_sepsis(title, data, sh_score=sh_score, coloring=k_means_list,
                                                 patient_ids=patient_ids, training_set=set_a,
                                                 color_map='tab20c', save_to_file=True)

    # k-Means AFTER PaCMAP
    k_means_list, sh_score = calculate_cluster_kmeans(data, n_clusters=amount_of_clusters)
    title = f"{amount_of_clusters} k-Means clusters ({set_a.name}) after PaCMAP"
    plot_clustering_with_silhouette_score_sepsis(title, data, sh_score=sh_score, coloring=k_means_list,
                                                 patient_ids=patient_ids, training_set=set_a,
                                                 color_map='tab20c', save_to_file=True)

    #k-means with imputation
    data_imp, patient_ids = calculate_pacmap(set_a, use_interpolation=True)

    k_means_list, sh_score = calculate_cluster_kmeans(training_set_to_data(set_a, use_interpolation=True), n_clusters=amount_of_clusters)
    title = f"{amount_of_clusters} k-Means ({set_a.name}) (interpolated)"
    plot_clustering_with_silhouette_score_sepsis(title, data_imp, sh_score=sh_score, coloring=k_means_list,
                                                 patient_ids=patient_ids, training_set=set_a,
                                                 color_map='tab20c', save_to_file=True)


