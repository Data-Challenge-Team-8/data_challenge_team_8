from typing import Dict, Tuple, List
import os
import pickle

from objects.patient import Patient
from IO.data_reader import DataReader
from web.UI_tools.analyse_tool import CompleteAnalysis


class TrainingSet:
    CACHE_PATH = os.path.join(".", "cache")

    def __init__(self, set_name: str, patients: Dict[str, Patient], keys: List[str], conduct_analysis: bool):
        self.set_name = set_name
        self.data = patients
        # variables are declared here and calculated in analysis
        self.test = "Teststring"
        self.__min_for_label: Dict[str, Tuple[str, float]] = {}
        self.__max_for_label: Dict[str, Tuple[str, float]] = {}
        self.__avg_for_label: Dict[str, float] = {}
        self.__NaN_amount_for_label: Dict[str, int] = {}
        self.__non_NaN_amount_for_label: Dict[str, int] = {}
        self.__plot_label_to_sepsis: Dict[str, Tuple[List[float], List[float]]] = {}
        self.__min_data_duration: Tuple[str, int] = None
        self.__max_data_duration: Tuple[str, int] = None
        self.__avg_data_duration: float = None
        self.__sepsis_patients: List[str] = None

        if conduct_analysis:
            self.analysis_dict, analysis_cache_name = CompleteAnalysis.get_analysis_from_cache(keys, self)
            print("New Training Set was created with set_name:", self.set_name, "and analysis_cache_name:",
                  analysis_cache_name)
        else:
            print("New Training Set was created with set_name:", self.set_name, "and no analysis in cache.")

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

    @classmethod        # TODO: loads Training Set with Data Reader (we can remove that later but it works for now)
    def get_training_set(cls, keys: List[str]):
        file_dir_path = ''
        if keys[0] == 'Set A':
            file_dir_path = r'./data/training_setA/'
        elif keys[0] == 'Set B':
            file_dir_path = r'./data/training_setB/'
        elif keys[0] == 'SetA + B':
            file_dir_path = r'./data/'      # TODO: How to select multiple folders for all sets?
        else:
            print("Please enter a valid dataset.", keys[0], "is unknown.")
        new_dict = DataReader().load_new_training_set(file_dir_path)
        new_set = TrainingSet(set_name="new_set", patients=new_dict, keys=keys, conduct_analysis=False)         # important not to start another analysis here (circle)
        return new_set

    # Not useful:
    def __save_obj_to_cache(self):  # very large file (280mb) and loading results in 'set' not obj
        pickle_data = {self}
        pickle.dump(pickle_data, open(os.path.join(TrainingSet.CACHE_PATH, self.analysis.analysis_cache_name), "wb"))

    @classmethod
    def load_obj_from_cache(cls, file_name: str):  # KeyError: 'plot_label_to_sepsis'
        pickle_data = pickle.load(open(os.path.join(TrainingSet.CACHE_PATH, file_name), "rb"))
        print("type:", type(pickle_data))
        return pickle_data  # seems to return a set if trying to import a pickle obj?
