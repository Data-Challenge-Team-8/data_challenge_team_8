import os
import pickle

from objects.training_set import TrainingSet
from tools.visualization.interpolation_comparison import plot_most_interesting_interpolation_patients, \
    plot_data_with_and_without_interpolation
from tools.pacmap_analysis import calculate_pacmap, plot_pacmap2D_sepsis, plot_pacmap2D, plot_pacmap3D
from tools.clustering_analysis import calculate_cluster_dbscan, calculate_cluster_kmeans, calculate_silhouette_score, \
    plot_clustering_with_silhouette_score_sepsis, training_set_to_data, plot_multiple_clusters

if __name__ == '__main__':
    # mini_set = TrainingSet(TrainingSet.PRESETS["Set A"][:61], name="Mini Set")

    # Load Trainingset (with interpolation and caching)
    set_a = TrainingSet.get_training_set("Set A")
    # plot_most_interesting_interpolation_patients(set_a)

    # Calculate avg_per_patient for each label

    # Calculate std_dev for each label

    # Save std_dev also to cache

    # Calculate z-values = (avg_per_patient - avg_of_label) / std_dev of label

    # Add z-values_per_patient for each label

    # Pacmap for Sepsis
    # temporarily caching pacmap_data
    file_path = os.path.join(TrainingSet.CACHE_PATH, 'pacmap_temp_save')
    if os.path.exists(file_path):
        print("Loading pacmap_data from cache.")
        pacmap_data = pickle.load(open(file_path, "rb"))
    else:
        pacmap_data, patient_ids = calculate_pacmap(set_a)
        print("Writing pacmap_temp_save to pickle cache!")
        pickle.dump(pacmap_data, open(file_path, "wb"))
    # pacmap_data, patient_ids = calculate_pacmap(set_a)
    # plot_pacmap2D_sepsis(f"PaCMAP colored by sepsis ({set_a.name})", pacmap_data, patient_ids, training_set=set_a)


    print("Starting with DBSCAN:")
    # TODO: We need to find out better settings
    eps = 0.4
    min_samples = 8
    # DBSCAN auf z_values_df ohne interpolation
    z_value_df = set_a.get_z_value_df(use_interpolation=False, fix_missing_values=False)
    # print(z_value_df.head())
    z_value_np = z_value_df.transpose().to_numpy()
    z_value_np.reshape(z_value_np.shape[0], -1)
    db_scan_list = calculate_cluster_dbscan(z_value_np, eps=eps, min_samples=min_samples)
    print("Clusters found:", set(db_scan_list))
    plot_pacmap2D(f"PaCMAP colored by DBSCAN clusters ({set_a.name}) without interpolation", pacmap_data, coloring=db_scan_list,
                  color_map='tab20c', save_to_file=True)

    # DBSCAN auf z_values_df mit interpolation # TODO: Ist das auch wirklich mit Interpolation? Die Datasets sehen gleich aus
    z_value_df = set_a.get_z_value_df(use_interpolation=True, fix_missing_values=True)
    # print(z_value_df.head())
    z_value_np = z_value_df.transpose().to_numpy()
    z_value_np.reshape(z_value_np.shape[0], -1)
    db_scan_list = calculate_cluster_dbscan(z_value_np, eps=eps, min_samples=min_samples)
    print("Clusters found:", set(db_scan_list))
    plot_pacmap2D(f"PaCMAP colored by DBSCAN clusters ({set_a.name}) with interpolation", pacmap_data, coloring=db_scan_list,
                  color_map='tab20c', save_to_file=True)

    # DBSCAN auf pacmap-data
    db_scan_list = calculate_cluster_dbscan(pacmap_data, eps=eps, min_samples=min_samples)
    print("Clusters found:", set(db_scan_list))
    plot_pacmap2D(f"2D PaCMAP colored by DBSCAN clusters ({set_a.name})", pacmap_data, coloring=db_scan_list,
                  color_map='tab20c', save_to_file=True)
    # TODO: 3D funktioniert nicht weil IndexError von data[2]
    # plot_pacmap3D(f"3D PaCMAP colored by DBSCAN clusters ({set_a.name})", pacmap_data, coloring=db_scan_list,
    #               color_map='tab20c', save_to_file=True)



    # # k-Means without imputation before Pacmap                      # TODO: Frage von Jakob: Wie funktioniert das? Was kommt denn dabei raus?
    # amount_of_clusters = 12
    # k_means_list, sh_score = calculate_cluster_kmeans(training_set_to_data(set_a), n_clusters=amount_of_clusters)
    # title = f"{amount_of_clusters} k-Means clusters ({set_a.name})"
    #
    # plot_clustering_with_silhouette_score_sepsis(title, pacmap_data, sh_score=sh_score, coloring=k_means_list,
    #                                              patient_ids=patient_ids, training_set=set_a,
    #                                              color_map='tab20c', save_to_file=True)
    # # k-Means without imputation after Pacmap
    # k_means_list, sh_score = calculate_cluster_kmeans(pacmap_data, n_clusters=amount_of_clusters)
    # title = f"{amount_of_clusters} k-Means clusters ({set_a.name}) after PaCMAP"
    # plot_clustering_with_silhouette_score_sepsis(title, pacmap_data, sh_score=sh_score, coloring=k_means_list,
    #                                              patient_ids=patient_ids, training_set=set_a,
    #                                              color_map='tab20c', save_to_file=True)
    # # k-means with imputation after Pacmap
    # data_imp, patient_ids = calculate_pacmap(set_a, use_interpolation=True)
    # k_means_list, sh_score = calculate_cluster_kmeans(training_set_to_data(set_a, use_interpolation=True), n_clusters=amount_of_clusters)
    # title = f"{amount_of_clusters} k-Means ({set_a.name}) (interpolated)"
    # plot_clustering_with_silhouette_score_sepsis(title, data_imp, sh_score=sh_score, coloring=k_means_list,
    #                                              patient_ids=patient_ids, training_set=set_a,
    #                                              color_map='tab20c', save_to_file=True)

    # # Implement silhouettes score analysis for k-means clustering
    # print("\nSilhouettes Score Analysis: ")
    # krange = list(range(2, 11))
    # avg_silhouettes = []
    # for n in krange:
    #     k_means_list, sh_score = calculate_cluster_kmeans(pacmap_data, n_clusters=n)
    #     avg_silhouettes.append(sh_score)
    # plot_multiple_clusters(avg_silhouettes, krange)
