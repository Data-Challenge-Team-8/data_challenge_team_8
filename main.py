import os.path
import pickle
from IO.data_reader import DataReader
from objects.patient import Patient
from objects.training_set import TrainingSet
from tools.visualization.interpolation_comparison import plot_most_interesting_interpolation_patients, \
    plot_data_with_and_without_interpolation
from tools.pacmap_analysis import calculate_pacmap, plot_pacmap2D_sepsis


if __name__ == '__main__':
    #mini_set = TrainingSet(TrainingSet.PRESETS["Set A"][:61], name="Mini Set")

    set_a = TrainingSet.get_training_set("Set A")

    data, patient_ids = calculate_pacmap(set_a)

    plot_pacmap2D_sepsis(f"PaCMAP colored by sepsis ({set_a.name})", data, patient_ids, training_set=set_a)

    set_A = TrainingSet.get_training_set("Set A")
    plot_most_interesting_interpolation_patients(set_A)

