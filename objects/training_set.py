import hashlib
from typing import Dict, Tuple, List
import os
import pickle

from objects.patient import Patient
from IO.data_reader import DataReader


def construct_cache_file_name(selected_label, selected_tool, selected_set):
    # keys is a list of the inputs selected f.e. ['Max, Min, Average', 'Label', 'Set']
    key_concat = ""                     # not good to use keys.sort() -> changes every time
    key_concat += selected_label
    key_concat += selected_tool
    key_concat += selected_set
    return hashlib.md5(key_concat.encode("utf-8")).hexdigest() + ".pickle"


class TrainingSet:
    CACHE_PATH = os.path.join(".", "cache")

    def __init__(self, patients: Dict[str, Patient], selected_label, selected_tool, selected_set):
        self.set_name = selected_set
        self.data = patients
        self.set_cache_name = construct_cache_file_name(selected_label, selected_tool, selected_set)
        print("New Training Set was created with set_name:", self.set_name)

    def get_subgroup(self, label: str, low_value, high_value, new_set_id: str = None):  # TODO: Not yet reworked
        """
        Split this set into a sub set based on low_value and high_value range
        :param label:
        :param low_value:
        :param high_value:
        :param new_set_id:
        :return:
        """
        if label not in Patient.LABELS:
            raise ValueError(f"The requested label {label} is not part of Patient.LABELS")
        subgroup_dict = {}
        for patient in self.data.values():
            if low_value <= patient.data[label] <= high_value:
                subgroup_dict[patient.ID] = patient

        subgroup_keys = ["subgroup", label, "create_subgroups"]
        if len(subgroup_dict.keys()) != 0:
            if new_set_id is None:
                return TrainingSet(self.set_name + f"-SubGroup_{label}", subgroup_dict, subgroup_keys)
            else:
                return TrainingSet(new_set_id, subgroup_dict, subgroup_keys)
        else:
            return None

    @classmethod        # TODO: loads complete Training Set with DataReader (we can remove that later but it works for now)
    def get_training_set(cls, selected_label, selected_tool, selected_set):
        file_dir_path = ''
        if selected_set == 'Set A':
            file_dir_path = r'./data/training_setA/'
        elif selected_set == 'Set B':
            file_dir_path = r'./data/training_setB/'
        elif selected_set == 'SetA + B':
            file_dir_path = r'./data/'      # TODO: How to select multiple folders for all sets?
        else:
            print("Please enter a valid dataset.", selected_set, "is unknown.")
            return
        new_dict = DataReader().load_new_training_set(file_dir_path)
        new_set = TrainingSet(new_dict, selected_label, selected_tool, selected_set)
        return new_set

    # Not useful:
    def __save_obj_to_cache(self):  # very large file (280mb) and loading results in 'set' not obj
        pickle_data = {self}
        pickle.dump(pickle_data, open(os.path.join(TrainingSet.CACHE_PATH, self.set_cache_name), "wb"))
        print("Training Set", self.set_name, "was cached into", self.set_cache_name)

    @classmethod
    def load_obj_from_cache(cls, file_name: str):  # KeyError: 'plot_label_to_sepsis'
        pickle_data = pickle.load(open(os.path.join(TrainingSet.CACHE_PATH, file_name), "rb"))
        print("type:", type(pickle_data))
        return pickle_data  # seems to return a set if trying to import a pickle obj?

    # alternative way to call analysis from here - not a good idea because circular calls
    def conduct_analysis(self):
        pass
