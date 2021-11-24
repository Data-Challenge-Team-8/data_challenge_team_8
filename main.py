from objects.training_set import TrainingSet
from tools.visualization.interpolation_comparison import plot_most_interesting_interpolation_patients, \
    plot_data_with_and_without_interpolation
from tools.pacmap_analysis import calculate_pacmap, plot_pacmap2D_sepsis, plot_pacmap2D
from tools.clustering_analysis import calculate_cluster_dbscan, calculate_cluster_kmeans


if __name__ == '__main__':
    #mini_set = TrainingSet(TrainingSet.PRESETS["Set A"][:61], name="Mini Set")

    # Load Trainingset (with interpolation and caching)
    set_a = TrainingSet.get_training_set("Set A")
    # plot_most_interesting_interpolation_patients(set_a)

    # Pacmap for Sepsis
    # TODO: Können wir das Pacmap data cachen? Dann gehen die ganzen Clusterings viel schneller.
    data, patient_ids = calculate_pacmap(set_a)
    # plot_pacmap2D_sepsis(f"PaCMAP colored by sepsis ({set_a.name})", data, patient_ids, training_set=set_a)

    # Calculate different clusters
    # TODO: Wir brauchen eigentlich standardisierte Daten: z = (wert_i - mean) / std_dev
    # TODO: DBSCAN funktioniert gar nicht??
    # db_scan_list = calculate_cluster_dbscan(set_a, eps=0.05, min_samples=200)
    # plot_pacmap2D(f"PaCMAP colored by DBSCAN clusters ({set_a.name})", data, coloring=db_scan_list, color_map='tab20c', save_to_file=True)
    amount_of_clusters = 12
    k_means_list = calculate_cluster_kmeans(set_a, n_clusters=amount_of_clusters)
    title = "PaCMAP colored by " + str(amount_of_clusters) + " k-Means clusters " + str(set_a.name)
    plot_pacmap2D(title, data, coloring=k_means_list, color_map='tab20c', save_to_file=True)
