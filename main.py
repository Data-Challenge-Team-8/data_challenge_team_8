import sys
import pandas as pd

from streamlit import cli as stcli
from objects.training_set import TrainingSet
#from classifier.timeseries.time_series_forest import TimeSeriesForest
#from tools.visualization.time_series_comparison import plot_time_series_density, plot_complete_time_series_for_patients, \
#    plot_reduced_time_series_data
from tools.analyse_tool import CompleteAnalysis

if __name__ == '__main__':
    selected_set = TrainingSet.get_training_set("Set A")
    temp_ca = CompleteAnalysis(selected_label="HR", selected_tool='Set Analysis', selected_set='Set Analysis', training_set=selected_set)
    general_info = {'Hospital System': ['Number of patients', 'Number of septic patients', 'Sepsis prevalence',
                                        'Number of entries', 'Number of NaNs', 'Relative number of NaNs',
                                        'Days recorded', 'Average hospital stay duration'],
                    selected_set.name: [len(selected_set.data.keys()), # TODO
                          temp_ca.get_sepsis_patients(selected_set),
                          temp_ca.get_rel_sepsis_amount(selected_set),
                          temp_ca.get_data_amount(selected_set),
                          temp_ca.get_total_NaN_amount(selected_set),
                          temp_ca.get_rel_NaN_amount(selected_set),
                          'Time Series Length',
                          temp_ca.get_avg_data_duration(selected_set)]
                    }
    df_general_info = pd.DataFrame(general_info)

    # TASK 06: WebApp Visualization of TimeSeries
    sys.argv = ["streamlit", "run", ".\\web\\app.py"]
    sys.exit(stcli.main())




    # TASK 06: TimeSeries Visualization & Classification
    # get small, random TrainingSet
    # set_a = TrainingSet.get_training_set("rnd Sample A")
    # avg_df = set_a.get_average_df(use_interpolation=True, fix_missing_values=True)
    # sepsis_df = set_a.get_sepsis_label_df()
    #
    # # get timeseries df for selected features
    # limit_to_features = ["HR"]
    # time_series = set_a.get_timeseries_df(use_interpolation=True, fix_missing_values=True,
    #                                       limit_to_features=limit_to_features)
    # # plotting complete timeseries data for a single patient
    # # plot_complete_time_series_for_patients(set_a, limit_to_features, plot_maximum=15)
    #
    # # plotting the time_series data that was reduced to only 39 timesteps
    # # plot_reduced_time_series_data(set_a, time_series)
    #
    # # Implement TimeSeriesForest classification
    # tsf = TimeSeriesForest(data_set=set_a, train_fraction=0.8, feature="HR")
    # print("Starting setup ...")
    # tsf.setup()
    # print("Starting plotting ...")
    # plot_time_series_density(tsf.train_data[0], label="HR", set_name=set_a.name+" (fixed, interpolated)")
    # print("Start training & testing ...")
    # tsf.train()
    # tsf.display_confusion_matrix(plotting=True)
