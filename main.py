import os
import pickle

from objects.training_set import TrainingSet
from tools.subspace_clustering_analysis import implement_bicluster_spectralcoclustering, \
    implement_bicluster_spectralbiclustering
from tools.visualization.interpolation_comparison import plot_most_interesting_interpolation_patients, \
    plot_data_with_and_without_interpolation
from tools.pacmap_analysis import get_pacmap
from tools.clustering_analysis import implement_DBSCAN, implement_k_means

if __name__ == '__main__':
    # Task 1: Build a Dashboard for Visualization of general statistics
    # ...

    # Task 2: Load Trainingset with interpolation (and caching)
    # set_a = TrainingSet(TrainingSet.PRESETS["Set A"][:61], name="Mini Set")
    set_a = TrainingSet.get_training_set("Set A")
    # plot_most_interesting_interpolation_patients(set_a)

    # Task 3: Implement different clustering methods
    pacmap_data, patient_ids = get_pacmap(set_a)
    #implement_DBSCAN(set_a, pacmap_data, patient_ids)
    # implement_k_means(set_a, pacmap_data, patient_ids)

    # Task 4: Implement different subspace clustering methods
    # scikit_learn module biclustering, alternatives: houghnet, biclustlib
    print("Doing Biclustering ...")
    implement_bicluster_spectralcoclustering(set_a, use_interpolation=False)
    # implement_bicluster_spectralbiclustering(set_a, use_interpolation=False)
    # TODO convert result into cluster data for pacmap visualization?
