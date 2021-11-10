import pprint
from typing import Dict, Tuple, List
import os
import hashlib
import pickle
import pandas as pd

from objects.patient import Patient

USE_CACHE = True


def construct_cache_file_name(keys):
    # keys is a list of the inputs selected f.e. ['Max, Min, Average', 'Label', 'Set']
    key_concat = ""
    keys.sort()
    for key in keys:
        key_concat += key
    return hashlib.md5(key_concat.encode("utf-8")).hexdigest() + ".pickle"


class CompleteAnalysis:
    global USE_CACHE
    CACHE_PATH = os.path.join(".", "cache")

    def __init__(self, keys: List[str], training_set):
        self.selected_set = keys[2]
        self.selected_tool = keys[1]
        self.selected_label = keys[0]
        self.analysis_cache_name = construct_cache_file_name(keys)
        self.training_set = training_set
        self.calculate_complete_analysis()

    @classmethod
    def check_analysis_is_cached(cls, file_name: str):
        return os.path.isfile(os.path.join(CompleteAnalysis.CACHE_PATH, file_name))

    @classmethod
    def get_analysis_from_cache(cls, keys, training_set):
        file_name = construct_cache_file_name(keys)
        if CompleteAnalysis.check_analysis_is_cached(file_name):
            print("Loading Analysis from cache:", file_name)
            return CompleteAnalysis.load_analysis_from_cache(file_name), file_name
        else:
            print("Starting new Analysis with name:", file_name)
            new_analysis = CompleteAnalysis(keys, training_set)
            new_analysis.save_analysis_to_cache()
            return CompleteAnalysis.load_analysis_from_cache(file_name), file_name

    @classmethod
    def load_analysis_from_cache(cls, file_name: str):  # KeyError: 'plot_label_to_sepsis'
        pickle_data = pickle.load(open(os.path.join(CompleteAnalysis.CACHE_PATH, file_name), "rb"))
        return pickle_data  # seems to return a set if trying to import a pickle obj?

    # TODO: Es ist vermutlich besser diese ganzen Attribute gleich der Analysis zu geben und nicht beim TS
    def calculate_complete_analysis(self):
        # minmaxavg
        min_label = self.calc_min_for_label(self.selected_label)
        max_label = self.calc_max_for_label(self.selected_label)
        avg_label = self.calc_avg_for_label(self.selected_label)
        # missing vals
        nan_amount = self.get_NaN_amount_for_label(self.selected_label)
        # non missing vals - already calculated in avg_label
        # non_nan_amount = self.get_non_NaN_amount_for_label(self.selected_label)
        # plot
        plot = self.get_plot_label_to_sepsis(self.selected_label)
        # data duration
        min_data_duration = self.get_min_data_duration()
        max_data_duration = self.get_max_data_duration()
        avg_data_duration =  self.get_avg_data_duration()
        # sepsis patients
        rel_sepsis_amount = self.get_rel_sepsis_amount()

        self.save_analysis_to_cache()                 # only save once = saves time

    def calc_min_for_label(self, label: str) -> Tuple[str, float]:
        """
        Get the minimal value for the label across all Patient objects in this set
        :param label:
        :return:
        """

        print(self.training_set.__min_for_label.keys())                                     # TODO: Error message: AttributeError: 'TrainingSet' object has no attribute '_CompleteAnalysis__min_for_label'
        if label not in self.training_set.__min_for_label.keys() or self.training_set.__min_for_label[label] is None:
            # was not calculated before, calculating now
            min_patient: str = None
            min_value: float = None
            for patient in self.training_set.data.values():
                v = patient.data[label].min()
                if pd.isna(v):
                    continue
                if min_value is None or min_value > v:
                    min_value = v
                    min_patient = patient.ID
            self.training_set.__min_for_label[label] = (min_patient, min_value)

        return self.training_set.__min_for_label[self.selected_label]

    def calc_max_for_label(self, label: str) -> Tuple[str, float]:
        """
        Get the maximal value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.training_set.__max_for_label.keys() or self.training_set.__max_for_label[label] is None:
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
            self.training_set.__max_for_label[label] = (max_patient, max_value)

        return self.training_set.__max_for_label[label]

    def calc_avg_for_label(self, label: str) -> float:
        """
        Get the average value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.training_set.__avg_for_label.keys() or self.training_set.__avg_for_label[label] is None:
            # was not calculated before, calculating now
            s = 0
            count = 0
            for patient in self.training_set.data.values():
                series = patient.data[label].dropna()
                s += series.sum()
                count += len(series)
            self.training_set.__non_NaN_amount_for_label[label] = count  # caching side result
            if count > 0:
                average = s / count
            else:
                average = None
            self.training_set.__avg_for_label[label] = average

        return self.training_set.__avg_for_label[label]

    def get_NaN_amount_for_label(self, label: str) -> int:
        """
        Get the amount of NaN values for the label across all Patient objects in this set
        :param label:
        :param no_cache:
        :return:
        """
        if label not in self.training_set.__NaN_amount_for_label.keys() or self.training_set.__NaN_amount_for_label[
            label] is None:
            count = 0
            for patient in self.training_set.data.values():
                count += patient.data[label].isnull().sum()
            self.training_set.__NaN_amount_for_label[label] = count

        return self.training_set.__NaN_amount_for_label[label]

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
        if label not in self.training_set.__non_NaN_amount_for_label.keys() or \
                self.training_set.__non_NaN_amount_for_label[label] is None:
            s = 0
            for patient in self.training_set.data.values():
                s += len(patient.data[label].dropna())
            self.training_set.__non_NaN_amount_for_label[label] = s

        return self.training_set.__non_NaN_amount_for_label[label]

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
        if self.training_set.__min_data_duration is None:
            # not calculated before, calculating now

            min_value: int = None
            min_patient: str = None
            for patient in self.training_set.data.values():
                if min_value is None or min_value > len(patient.data):
                    min_value = len(patient.data)
                    min_patient = patient.ID

            self.training_set.__min_data_duration = (min_patient, min_value)

        return self.training_set.__min_data_duration

    def get_max_data_duration(self) -> Tuple[str, int]:
        """
        Get the minimal amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.training_set.__max_data_duration is None:
            # not calculated before, calculating now

            max_value: int = None
            max_patient: str = None
            for patient in self.training_set.data.values():
                if max_value is None or max_value < len(patient.data):
                    max_value = len(patient.data)
                    max_patient = patient.ID

            self.training_set.__max_data_duration = (max_patient, max_value)

        return self.training_set.__max_data_duration

    def get_avg_data_duration(self) -> float:
        """
        Get the average amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.training_set.__avg_data_duration is None:
            # not calculated before, calculating now
            avg_sum = 0
            avg_count = 0
            for patient in self.training_set.data.values():
                avg_count += 1
                avg_sum += len(patient.data)

            self.training_set.__avg_data_duration = avg_sum / avg_count

        return self.training_set.__avg_data_duration

    def sepsis_patients(self) -> List[str]:
        if self.training_set.__sepsis_patients is None:
            # not calculated before, calculating now
            self.training_set.__sepsis_patients = []
            for patient in self.training_set.data.values():
                if patient.data["SepsisLabel"].dropna().sum() > 0:  # at least one sepsis
                    self.training_set.__sepsis_patients.append(patient.ID)

        return self.training_set.__sepsis_patients

    def get_rel_sepsis_amount(self) -> float:
        """
        Get the relative amount of patients that develop sepsis across all Patient objects in this set
        :return:
        """
        r = len(self.training_set.sepsis_patients) / len(self.training_set.data.keys())

        return r

    def get_plot_label_to_sepsis(self, label: str):  # TODO: remove from here to plot....py
        """
        Gets the selected labels for patient with and without sepsis
        :return:
        """
        if label not in self.training_set.__plot_label_to_sepsis.keys() or not self.training_set.__plot_label_to_sepsis:
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

            self.training_set.__plot_label_to_sepsis[label] = (
                sepsis_pos, sepsis_neg)  # pos = plot_data[0] and neg = plot_data[1]

        return self.training_set.__plot_label_to_sepsis[label]

    def save_analysis_to_cache(self):  # this saves a dict not the obj
        pickle_data = {
            "min_for_label": self.training_set.__min_for_label,
            "max_for_label": self.training_set.__max_for_label,
            "avg_for_label": self.training_set.__avg_for_label,
            "NaN_amount_for_label": self.training_set.__NaN_amount_for_label,
            "non_NaN_amount_for_label": self.training_set.__non_NaN_amount_for_label,
            "min_data_duration": self.training_set.__min_data_duration,
            "max_data_duration": self.training_set.__max_data_duration,
            "avg_data_duration": self.training_set.__avg_data_duration,
            "sepsis_patients": self.training_set.__sepsis_patients,
            "plot_label_to_sepsis": self.training_set.__plot_label_to_sepsis
        }
        pickle.dump(pickle_data, open(os.path.join(CompleteAnalysis.CACHE_PATH, self.analysis_cache_name), "wb"))
