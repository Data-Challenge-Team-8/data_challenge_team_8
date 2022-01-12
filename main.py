from objects.training_set import TrainingSet
from tools.classifier.decision_tree import implement_decision_tree
from tools.classifier.random_forest import implement_random_forest

if __name__ == '__main__':
    # Task 1: Build a Dashboard for Visualization of general statistics
    # correlation_to_sepsis_df = get_and_plot_sepsis_correlation(set_a)
    # start streamlit app mit web.app.create_app()

    # Task 2: Load Trainingset with interpolation (and caching)
    # set_a_mini = TrainingSet(TrainingSet.PRESETS["Set A"][:61], name="Mini Set")
    set_a = TrainingSet.get_training_set("Set A")
    # plot_most_interesting_interpolation_patients(set_a)

    # Task 3: Implement different clustering methods
    # pacmap_data, patient_ids = set_a.get_pacmap()
    # implement_DBSCAN(set_a, pacmap_data, patient_ids)
    # implement_k_means(set_a, pacmap_data, patient_ids)

    # Task 4: Implement different subspace clustering methods
    # scikit_learn module for biclustering, alternatives: houghnet, biclustlib
    # print("Doing Biclustering ...")
    # implement_bicluster_spectralcoclustering(set_a, use_interpolation=False)
    # implement_bicluster_spectralbiclustering(set_a, use_interpolation=False)
    # TODO convert result into cluster data for pacmap visualization? - Frage Jakob: ist das todo noch aktuell?

    # Task 5: Use Classifier (DecisionTree)
    avg_df = set_a.get_average_df(use_interpolation=True, fix_missing_values=True)
    sepsis_df = set_a.get_sepsis_label_df()
    print("Implementing Decision Tree Clustering.")
    implement_random_forest(avg_df, sepsis_df)
