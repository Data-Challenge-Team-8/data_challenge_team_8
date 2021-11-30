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



    # TODO: We need to find out better settings --> use silhouette score
    print("Starting with DBSCAN:")
    eps = 0.4
    min_samples = 8
    # DBSCAN auf z_values_df ohne interpolation
    z_value_df = set_a.get_z_value_df(use_interpolation=False, fix_missing_values=False)
    z_value_np = z_value_df.transpose().to_numpy()
    z_value_np.reshape(z_value_np.shape[0], -1)
    db_scan_list = calculate_cluster_dbscan(z_value_np, eps=eps, min_samples=min_samples)
    print("Clusters found:", set(db_scan_list))
    plot_pacmap2D(f"DBSCAN clusters ({set_a.name}) no interpol. and eps={eps} min_sampl={min_samples}_",
                  data=pacmap_data,                         # Ist pacmap_data hier korrekt?
                  coloring=db_scan_list,
                  color_map='tab20c',
                  save_to_file=True)

    # DBSCAN auf z_values_df mit interpolation
    z_value_df = set_a.get_z_value_df(use_interpolation=True, fix_missing_values=True)
    z_value_np = z_value_df.transpose().to_numpy()
    z_value_np.reshape(z_value_np.shape[0], -1)
    db_scan_list = calculate_cluster_dbscan(z_value_np, eps=eps, min_samples=min_samples)
    print("Clusters found:", set(db_scan_list))
    plot_pacmap2D(f"DBSCAN clusters ({set_a.name}) with interpol. and eps={eps} min_sampl={min_samples}_",
                  data=pacmap_data,
                  coloring=db_scan_list,
                  color_map='tab20c',
                  save_to_file=True)

    # DBSCAN auf pacmap-data               # macht das Sinn?
    db_scan_list = calculate_cluster_dbscan(pacmap_data, eps=eps, min_samples=min_samples)
    print("Clusters found:", set(db_scan_list))
    plot_pacmap2D(f"DBSCAN clusters ({set_a.name}) based on pacmap_data with interpol. and eps={eps} min_sampl={min_samples}_",
                  data=pacmap_data,
                  coloring=db_scan_list,
                  color_map='tab20c',
                  save_to_file=True)
    print("Finished")
    # TODO: 3D funktioniert nicht weil IndexError von data[2] --> pacmap3D benötigt
    # plot_pacmap3D(f"DBSCAN clusters ({set_a.name}) without interpol. and parameters: eps={eps} min_sampl={min_samples}_",
    #               data=pacmap_data,
    #               coloring=db_scan_list,
    #               color_map='tab20c',
    #               save_to_file=True)


    # # k-Means without imputation before Pacmap
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
