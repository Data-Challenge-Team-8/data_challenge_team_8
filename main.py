import sys
import sys
from streamlit import cli as stcli
# from objects.training_set import TrainingSet
# from classifier.timeseries.time_series_forest import TimeSeriesForest
# from tools.visualization.time_series_comparison import plot_time_series_density, plot_two_time_series, \
#     plot_time_series_sepsis_background

if __name__ == '__main__':
    sys.argv = ["streamlit", "run", ".\\web\\app.py"]
    sys.exit(stcli.main())
    # set_a = TrainingSet.get_training_set("rnd Sample A")
    # set_a = TrainingSet.get_training_set("Set A")
    # avg_df = set_a.get_average_df(use_interpolation=True, fix_missing_values=True)
    # sepsis_df = set_a.get_sepsis_label_df()
    # list_of_features = ["HR", "SepsisLabel"]
    # time_series = set_a.get_timeseries_df(use_interpolation=True, fix_missing_values=True,
    #                                       limit_to_features=list_of_features)
    # set_a_sepsis_df = set_a.get_sepsis_label_df()
    #
    # # plotting time_series for a single patient
    # plot_counter = 0
    # for patient_id in set_a.data:
    #     if set_a.check_patient_has_sepsis(set_a_sepsis_df, patient_id):
    #         temp_patient = set_a.get_patient_form_id(patient_id)
    #         temp_patient.plot_features_time_series(list_of_features)
    #         plot_counter += 1
    #         if plot_counter > 100:
    #             break

    # plotting the time_series data that was reduced to only 39 timesteps
    # temp_hr = None
    # for patient_feature_tuple in time_series.transpose():
    #     if set_a.check_patient_has_sepsis(set_a_sepsis_df, patient_feature_tuple[0]):
    #         if patient_feature_tuple[1] == "HR":                        # select complete column
    #             temp_hr = time_series.loc[patient_feature_tuple]
    #         elif patient_feature_tuple[1] == "SepsisLabel":
    #             temp_sepsis = time_series.loc[patient_feature_tuple]
    #             try:
    #                 if temp_hr.name[0] == temp_sepsis.name[0]:
    #                     # plot_two_time_series(patient_id=patient_feature_tuple[0],
    #                     #                      series_one=temp_hr,
    #                     #                      label_one="HR",
    #                     #                      series_two=temp_sepsis,
    #                     #                      label_two="SepsisLabel")
    #                     plot_time_series_sepsis_background(patient_id=patient_feature_tuple[0],
    #                                                        series_one=temp_hr,
    #                                                        label_one="HR",
    #                                                        series_two=temp_sepsis,
    #                                                        label_two="SepsisLabel")
    #             except AttributeError:
    #                 pass
    #         else:
    #             pass
    #     else:
    #         pass

    # Task 4: Implement different subspace clustering methods
    # scikit_learn module biclustering, alternatives: houghnet, biclustlib
    # print("Doing Biclustering ...")
    # implement_bicluster_spectralcoclustering(set_a, use_interpolation=False)
    # implement_bicluster_spectralbiclustering(set_a, use_interpolation=False)
    # TODO convert result into cluster data for pacmap visualization?
