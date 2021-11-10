import json
from typing import Dict, Tuple, List
import os
import hashlib
import pickle
import pandas as pd

from objects.patient import Patient
from objects.training_set import TrainingSet

USE_CACHE = True


def construct_cache_file_name(selected_label, selected_tool, selected_set):
    # keys is a list of the inputs selected f.e. ['Max, Min, Average', 'Label', 'Set']
    key_concat = ""                     # not good to use keys.sort() -> changes every time
    key_concat += selected_label
    key_concat += selected_tool
    key_concat += selected_set
    return hashlib.md5(key_concat.encode("utf-8")).hexdigest() + ".pickle"


class CompleteAnalysis:
    global USE_CACHE
    CACHE_PATH = os.path.join(".", "cache")

    def __init__(self, selected_label, selected_tool, selected_set, training_set):
        self.selected_set = selected_set
        self.selected_tool = selected_tool
        self.selected_label = selected_label
        self.analysis_cache_name = construct_cache_file_name(selected_label, selected_tool, selected_set)
        self.training_set = training_set

        # variables are declared here and calculated in analysis
        self.min_for_label: Dict[str, Tuple[str, float]] = {}
        self.max_for_label: Dict[str, Tuple[str, float]] = {}
        self.avg_for_label: Dict[str, float] = {}
        self.NaN_amount_for_label: Dict[str, int] = {}
        self.non_NaN_amount_for_label: Dict[str, int] = {}
        self.plot_label_to_sepsis: Dict[str, Tuple[List[float], List[float]]] = {}
        self.min_data_duration: Tuple[str, int] = None
        self.max_data_duration: Tuple[str, int] = None
        self.avg_data_duration: float = None
        self.sepsis_patients: List[str] = None

        self.calculate_complete_analysis()

    @classmethod
    def check_analysis_is_cached(cls, file_name: str):
        return os.path.isfile(os.path.join(CompleteAnalysis.CACHE_PATH, file_name))

    @classmethod
    def get_analysis(cls, selected_label, selected_tool, selected_set):
        file_name = construct_cache_file_name(selected_label, selected_tool, selected_set)
        if CompleteAnalysis.check_analysis_is_cached(file_name) and USE_CACHE:
            print("Loading Analysis from cache:", file_name)
            return CompleteAnalysis.load_analysis_from_cache(file_name), file_name
        else:
            print("Starting new Analysis with cache name:", file_name)
            print("Loading complete Training Set for this Analysis.")
            loaded_training_set = TrainingSet.get_training_set(selected_label, selected_tool, selected_set)                 # if analysis not cached TS needs to be loaded
            CompleteAnalysis(selected_label, selected_tool, selected_set, loaded_training_set)     # Construct this new Analysis, directly calculate all and save to cache
            return CompleteAnalysis.load_analysis_from_cache(file_name), file_name      # get this analysis from cache

    @classmethod
    def load_analysis_from_cache(cls, file_name: str):
        pickle_data = pickle.load(open(os.path.join(CompleteAnalysis.CACHE_PATH, file_name), "rb"))
        return pickle_data  # returns a dict

    # TODO: Es ist vermutlich besser diese ganzen Attribute gleich der Analysis zu geben und nicht beim TS
    def calculate_complete_analysis(self):
        self.get_min_for_label(self.selected_label)
        self.get_max_for_label(self.selected_label)
        self.get_avg_for_label(self.selected_label)

        self.get_NaN_amount_for_label(self.selected_label)
        # non missing vals - already calculated in avg_label
        # self.get_non_NaN_amount_for_label(self.selected_label)

        self.get_plot_label_to_sepsis(self.selected_label)

        self.get_min_data_duration()
        self.get_max_data_duration()
        self.get_avg_data_duration()

        self.get_rel_sepsis_amount()

        self.save_analysis_to_cache()  # only save once = saves time

    def get_min_for_label(self, label: str) -> Tuple[str, float]:
        """
        Get the minimal value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.min_for_label.keys() or self.min_for_label[label] is None:
            # was not calculated before, calculating now
            min_patient: str = None
            min_value: float = None
            for patient in self.training_set.data.values():                 # TODO: AttributeError: 'NoneType' object has no attribute 'data'
                v = patient.data[label].min()                               # TODO: Raise KeyError(key) for 02Sat - maybe misspelled??
                if pd.isna(v):
                    continue
                if min_value is None or min_value > v:
                    min_value = v
                    min_patient = patient.ID
            self.min_for_label[label] = (min_patient, min_value)

        return self.min_for_label[self.selected_label]

    def get_max_for_label(self, label: str) -> Tuple[str, float]:
        """
        Get the maximal value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.max_for_label.keys() or self.max_for_label[label] is None:
            # was not calculated before, calculating now
            max_patient: str = None
            max_value: float = None  # 3. Error message "none type" this is the value because of which there is an error
            for patient in self.training_set.data.values():
                v = patient.data[label].max()
                if pd.isna(v):
                    continue
                if max_value is None or max_value < v:
                    max_value = v
                    max_patient = patient.ID
            self.max_for_label[label] = (max_patient, max_value)

        return self.max_for_label[label]

    def get_avg_for_label(self, label: str) -> float:
        """
        Get the average value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.avg_for_label.keys() or self.avg_for_label[label] is None:
            # was not calculated before, calculating now
            s = 0
            count = 0
            for patient in self.training_set.data.values():
                series = patient.data[label].dropna()
                s += series.sum()
                count += len(series)
            self.non_NaN_amount_for_label[label] = count  # caching side result
            if count > 0:
                average = s / count
            else:
                average = None
            self.avg_for_label[label] = average

        return self.avg_for_label[label]

    def get_NaN_amount_for_label(self, label: str) -> int:
        """
        Get the amount of NaN values for the label across all Patient objects in this set
        :param label:
        :param no_cache:
        :return:
        """
        if label not in self.NaN_amount_for_label.keys() or self.NaN_amount_for_label[label] is None:
            count = 0
            for patient in self.training_set.data.values():
                count += patient.data[label].isnull().sum()
            self.NaN_amount_for_label[label] = count

        return self.NaN_amount_for_label[label]

    def get_total_NaN_amount(self) -> int:
        """
        Get the amount of NaN values across all Patient objects in this set
        :return:
        """
        count = 0
        for label in Patient.LABELS:
            count += self.get_NaN_amount_for_label(label)

        return count

    def get_avg_rel_NaN_amount_for_label(self, label: str) -> float:
        """
        Get the average relative amount of NaN values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        r = self.get_NaN_amount_for_label(label) / self.get_data_amount_for_label(label)

        return r

    def get_rel_NaN_amount(self) -> float:  # TODO: This will be helpful for Task 2.2 b)
        """
        Get the relative amount of NaN values across all Patient objects in this set
        :return:
        """
        r = self.get_total_NaN_amount() / self.get_data_amount()

        return r

    def get_data_amount_for_label(self, label: str) -> float:
        """
        Get the amount of values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        return self.get_NaN_amount_for_label(label) / self.get_non_NaN_amount_for_label(label)

    def get_data_amount(self) -> int:
        """
        Get the amount of values across all Patient objects in this set
        :return:
        """
        r = self.get_total_NaN_amount() + self.get_non_NaN_amount()

        return r

    def get_non_NaN_amount_for_label(self, label: str) -> int:  # also gets calculated with avg_label
        """
        Get the amount of non-NaN values for the label across all Patient objects in this set
        :param label:
        :param no_cache:
        :return:
        """
        if label not in self.non_NaN_amount_for_label.keys() or \
                self.non_NaN_amount_for_label[label] is None:
            s = 0
            for patient in self.training_set.data.values():
                s += len(patient.data[label].dropna())
            self.non_NaN_amount_for_label[label] = s

        return self.non_NaN_amount_for_label[label]

    def get_non_NaN_amount(self) -> int:
        """
        Get the amount of non-NaN values across all Patient objects in this set
        :return:
        """
        count = 0
        for label in Patient.LABELS:
            count += self.get_non_NaN_amount_for_label(label)

        return count

    def get_avg_rel_non_NaN_amount_for_label(self, label: str) -> float:
        """
        Get the average relative amount of non-NaN values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        r = 1 - self.get_avg_rel_NaN_amount_for_label(label)

        return r

    def get_rel_non_NaN_amount(self) -> float:
        """
        Get the relative amount of non-NaN values across all Patient objects in this set
        :return:
        """
        r = 1 - self.get_rel_NaN_amount()

        return r

    def get_min_data_duration(self) -> Tuple[str, int]:
        """
        Get the minimal amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.min_data_duration is None:
            # not calculated before, calculating now

            min_value: int = None
            min_patient: str = None
            for patient in self.training_set.data.values():
                if min_value is None or min_value > len(patient.data):
                    min_value = len(patient.data)
                    min_patient = patient.ID

            self.min_data_duration = (min_patient, min_value)

        return self.min_data_duration

    def get_max_data_duration(self) -> Tuple[str, int]:
        """
        Get the minimal amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.max_data_duration is None:
            # not calculated before, calculating now

            max_value: int = None
            max_patient: str = None
            for patient in self.training_set.data.values():
                if max_value is None or max_value < len(patient.data):
                    max_value = len(patient.data)
                    max_patient = patient.ID

            self.max_data_duration = (max_patient, max_value)

        return self.max_data_duration

    def get_avg_data_duration(self) -> float:
        """
        Get the average amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.avg_data_duration is None:
            # not calculated before, calculating now
            avg_sum = 0
            avg_count = 0
            for patient in self.training_set.data.values():
                avg_count += 1
                avg_sum += len(patient.data)

            self.avg_data_duration = avg_sum / avg_count

        return self.avg_data_duration

    def get_sepsis_patients(self) -> List[str]:
        if self.sepsis_patients is None:
            self.sepsis_patients = []
            for patient in self.training_set.data.values():
                if patient.data[
                    "SepsisLabel"].dropna().sum() > 0:  # at least one sepsis - adding patients not time steps
                    self.sepsis_patients.append(patient.ID)

        return self.sepsis_patients

    def get_rel_sepsis_amount(self) -> float:
        """
        Get the relative amount of patients that develop sepsis across all Patient objects in this set
        :return:
        """
        r = len(self.get_sepsis_patients()) / len(self.training_set.data.keys())

        return r

    def get_plot_label_to_sepsis(self, label: str):  # TODO: remove from here to plot....py
        """
        Gets the selected labels for patient with and without sepsis
        :return:
        """
        if label not in self.plot_label_to_sepsis.keys() or not self.plot_label_to_sepsis:
            # was not calculated before, calculating now
            sepsis_pos = []
            sepsis_neg = []
            for patient in self.training_set.data.values():  # Error message: AttributeError: 'NoneType' object has no attribute 'values'
                label_vals = patient.data[label]
                for label_val in label_vals:
                    if pd.notna(label_val):
                        if int(patient.data["SepsisLabel"][1]) == 1:
                            sepsis_pos.append(float(label_val))
                        else:
                            sepsis_neg.append(float(label_val))

            self.plot_label_to_sepsis[label] = (
                sepsis_pos, sepsis_neg)  # pos = plot_data[0] and neg = plot_data[1]

        return self.plot_label_to_sepsis

    def save_analysis_to_cache(self):  # this saves a dict not the obj
        pickle_data = {
            "min_for_label": self.min_for_label,
            "max_for_label": self.max_for_label,
            "avg_for_label": self.avg_for_label,
            "NaN_amount_for_label": self.NaN_amount_for_label,
            "non_NaN_amount_for_label": self.non_NaN_amount_for_label,
            "min_data_duration": self.min_data_duration,
            "max_data_duration": self.max_data_duration,
            "avg_data_duration": self.avg_data_duration,
            "sepsis_patients": self.sepsis_patients,
            "plot_label_to_sepsis": self.plot_label_to_sepsis
        }
        pickle.dump(pickle_data, open(os.path.join(CompleteAnalysis.CACHE_PATH, self.analysis_cache_name), "wb"))
        print("Analysis was cached into file", self.analysis_cache_name)

    def save_analysis_to_JSON(self):
        json_data = {
            "min_for_label": self.min_for_label,
            "max_for_label": self.max_for_label,
            "avg_for_label": self.avg_for_label,
            "NaN_amount_for_label": self.NaN_amount_for_label,
            "non_NaN_amount_for_label": self.non_NaN_amount_for_label,
            "min_data_duration": self.min_data_duration,
            "max_data_duration": self.max_data_duration,
            "avg_data_duration": self.avg_data_duration,
            "sepsis_patients": self.sepsis_patients,
            "plot_label_to_sepsis": self.plot_label_to_sepsis
        }
        json.dump(json_data, open(os.path.join(CompleteAnalysis.CACHE_PATH, self.analysis_cache_name), "wb"))
        print("Analysis was cached into file", self.analysis_cache_name)
