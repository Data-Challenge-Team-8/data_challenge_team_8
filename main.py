import sys
import os
from streamlit import cli as stcli
from objects.training_set import TrainingSet
from classifier.timeseries.time_series_forest import TimeSeriesForest
from tools.clustering_analysis import implement_DBSCAN_on_avg_df, implement_k_means_on_avg_df
from tools.imbalance_methods import get_near_miss_for_training_set
from tools.pacmap_analysis import plot_pacmap2D_sepsis, calculate_pacmap_on_avg_df
from tools.visualization.time_series_comparison import plot_time_series_density, plot_complete_time_series_for_patients, \
    plot_reduced_time_series_data

if __name__ == '__main__':
    # TASK FINAL: WebApp Visualization
    # sys.argv = ["streamlit", "run", os.path.join(".", "web", "app.py")]
    # sys.exit(stcli.main())

    # TASK FINAL: Implementing Balanced Set for Clustering
    set_a = TrainingSet.get_training_set("Set A")

    # PacMap imbalanced set
    # avg_df = set_a.get_average_df(use_interpolation=True, fix_missing_values=True)
    # Plot PacMap of normal (imbalanced) set and of near_miss (balanced) set
    # pacmap_data_normal, sepsis_ids_normal = calculate_pacmap_on_avg_df(avg_df)
    # plot_pacmap2D_sepsis(plot_title="PacMap on Set A, interpolated, Imbalanced",
    #                      data=pacmap_data_normal,
    #                      patient_ids=sepsis_ids_normal,
    #                      training_set=set_a,
    #                      save_to_file=True)

    # PacMap balanced set
    avg_df_near_miss, sepsis_label_near_miss = get_near_miss_for_training_set(training_set=set_a, version=2)
    # pacmap_data_near_miss, sepsis_ids_near_miss = calculate_pacmap_on_avg_df(avg_df_near_miss.transpose())
    # plot_pacmap2D_sepsis(plot_title="PacMap on Set A, interpolated, NearMiss(v2)",
    #                      data=pacmap_data_near_miss,
    #                      patient_ids=sepsis_ids_near_miss,
    #                      training_set=set_a,
    #                      save_to_file=True)

    # Clustering after near miss
    # implement_k_means_on_avg_df(training_set=set_a, avg_df=avg_df_near_miss,
    #                             additional_options_title="interpolated_NearMiss(v2)", save_to_file=True)
    implement_DBSCAN_on_avg_df(training_set=set_a, avg_df=avg_df_near_miss,
                               additional_options_title="interpolated, NearMiss(v2)", save_to_file=True,
                               filter_labels=True)


    # TASK 06: TimeSeries Visualization & Classification
    # get small, random TrainingSet
    # set_a = TrainingSet.get_training_set("rnd Sample A")
    # avg_df = set_a.get_average_df(use_interpolation=True, fix_missing_values=True)
    # sepsis_df = set_a.get_sepsis_label_df()

    # # get timeseries df for selected features
    # limit_to_features = ["HR"]
    # time_series = set_a.get_timeseries_df(use_interpolation=True, fix_missing_values=True,
    #                                       limit_to_features=limit_to_features)
    # # plotting complete timeseries data for a single patient
    # # plot_complete_time_series_for_patients(set_a, limit_to_features, plot_maximum=15)
    #
    # # plotting the time_series data that was reduced to only 39 timesteps
    # # plot_reduced_time_series_data(set_a, time_series)

    # # Implement TimeSeriesForest classification
    # tsf = TimeSeriesForest(data_set=set_a, train_fraction=0.8, feature="HR")
    # print("Starting setup ...")
    # tsf.setup()
    # print("Starting plotting ...")
    # plot_time_series_density(tsf.train_data[0], label="HR", set_name=set_a.name+" (fixed, interpolated)")
    # print("Start training & testing ...")
    # tsf.train()
    # tsf.display_confusion_matrix(plotting=True)
