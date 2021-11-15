from typing import Dict, Tuple, List
import os
import hashlib
import pickle
import pandas as pd
import jsonpickle
import json
import datetime

from objects.patient import Patient
from objects.training_set import TrainingSet

USE_CACHE = True


def construct_cache_file_name(selected_label, selected_set):                # removed "tool" from here - always complete analysis for all features
    # keys is a list of the inputs selected f.e. ['Label', 'Set']
    key_concat = ""  # not good to use keys.sort() -> changes every time
    key_concat += selected_label
    key_concat += selected_set
    return hashlib.md5(key_concat.encode("utf-8")).hexdigest() + "_OBJ" + ".pickle"         # Now working with OBJ - bad size, but better usefulness of get_analysis


class CompleteAnalysis:
    global USE_CACHE
    CACHE_PATH = os.path.join("../web/UI_tools", "cache")

    def __init__(self, selected_label, selected_tool, selected_set, training_set: TrainingSet):
        self.selected_set = selected_set
        self.selected_tool = selected_tool
        self.selected_label = selected_label
        self.analysis_cache_name = construct_cache_file_name(selected_label, selected_set)

        # variables are declared here and calculated in analysis
        self.min_for_label: Dict[str, Tuple[str, float]] = {}
        self.max_for_label: Dict[str, Tuple[str, float]] = {}
        self.avg_for_label: Dict[str, float] = {}
        self.NaN_amount_for_label: Dict[str, int] = {}
        self.rel_NaN_for_label: float = None
        self.non_NaN_amount_for_label: Dict[str, int] = {}
        self.plot_label_to_sepsis: Dict[str, Tuple[List[float], List[float]]] = {}          # TODO: This is on time frame level not patient level for the histogram!
        self.min_data_duration: Tuple[str, int] = None
        self.max_data_duration: Tuple[str, int] = None
        self.avg_data_duration: float = None
        self.sepsis_patients: List = None
        self.rel_sepsis_amount: float = None

        self.calculate_complete_analysis(training_set)

    @classmethod
    def check_analysis_is_cached(cls, file_name: str):
        return os.path.isfile(os.path.join(CompleteAnalysis.CACHE_PATH, file_name))

    @classmethod
    def get_analysis(cls, selected_label, selected_tool, selected_set):
        file_name = construct_cache_file_name(selected_label, selected_set)
        if CompleteAnalysis.check_analysis_is_cached(file_name) and USE_CACHE:
            print("\nLoading Analysis for", selected_label, selected_set, "from cache:", file_name,
                  " At time: ", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))
            return CompleteAnalysis.load_analysis_from_cache(file_name), file_name
        else:
            print("\nStarting new Analysis for", selected_label, selected_set, "with cache name:", file_name,
                  " At time: ", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))
            loaded_training_set = TrainingSet.get_training_set(name=selected_set)  # if analysis not cached TS needs
            # to be loaded
            CompleteAnalysis(selected_label=selected_label, selected_tool=selected_tool,
                             selected_set=selected_set,
                             training_set=loaded_training_set)  # Construct this new Analysis, directly calculate all and save to cache
            return CompleteAnalysis.load_analysis_from_cache(file_name), file_name  # get this analysis_dict from cache

    @classmethod
    def load_analysis_from_cache(cls, file_name: str):
        pickle_data = pickle.load(open(os.path.join(CompleteAnalysis.CACHE_PATH, file_name), "rb"))
        return pickle_data  # returns a dict

    def calculate_complete_analysis(self, training_set):
        self.get_min_for_label(self.selected_label, training_set)
        self.get_max_for_label(self.selected_label, training_set)
        self.get_avg_for_label(self.selected_label, training_set)

        self.get_rel_NaN_amount_for_label(self.selected_label, training_set)
        # non missing vals - already calculated in avg_label
        # self.get_non_NaN_amount_for_label(self.selected_label)

        self.get_plot_label_to_sepsis(self.selected_label, training_set)

        self.get_min_data_duration(training_set)
        self.get_max_data_duration(training_set)
        self.get_avg_data_duration(training_set)

        self.get_rel_sepsis_amount(training_set)

        self.save_analysis_obj_to_cache()
        # self.save_analysis_to_cache()  # saves selected features to dict
        # self.save_analysis_to_JSON()  # Tool for JSON export

    def get_min_for_label(self, label: str, training_set) -> Tuple[str, float]:
        """
        Get the minimal value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.min_for_label.keys() or self.min_for_label[label] is None:
            min_patient: str = None
            min_value: float = None
            for patient in training_set.data.values():
                v = patient.data[label].min()
                if pd.isna(v):
                    continue
                if min_value is None or min_value > v:
                    min_value = v
                    min_patient = patient.ID
            self.min_for_label[label] = (min_patient, min_value)

        return self.min_for_label[self.selected_label]

    def get_max_for_label(self, label: str, training_set) -> Tuple[str, float]:
        """
        Get the maximal value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.max_for_label.keys() or self.max_for_label[label] is None:
            # was not calculated before, calculating now
            max_patient: str = None
            max_value: float = None  # 3. Error message "none type" this is the value because of which there is an error
            for patient in training_set.data.values():
                v = patient.data[label].max()
                if pd.isna(v):
                    continue
                if max_value is None or max_value < v:
                    max_value = v
                    max_patient = patient.ID
            self.max_for_label[label] = (max_patient, max_value)

        return self.max_for_label[label]

    def get_avg_for_label(self, label: str, training_set) -> float:
        """
        Get the average value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.avg_for_label.keys() or self.avg_for_label[label] is None:
            # was not calculated before, calculating now
            s = 0
            count = 0
            for patient in training_set.data.values():
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

    def get_NaN_amount_for_label(self, label: str, training_set) -> int:
        """
        Get the amount of NaN values for the label across all Patient objects in this set
        :param label:
        :param no_cache:
        :return:
        """
        if label not in self.NaN_amount_for_label.keys() or self.NaN_amount_for_label[label] is None:
            count = 0
            for patient in training_set.data.values():
                count += patient.data[label].isnull().sum()
            self.NaN_amount_for_label[label] = count

        return self.NaN_amount_for_label[label]

    def get_total_NaN_amount(self, training_set) -> int:
        """
        Get the amount of NaN values across all Patient objects in this set
        :return:
        """
        count = 0
        for label in Patient.LABELS:
            count += self.get_NaN_amount_for_label(label, training_set)

        return count

    def get_rel_NaN_amount_for_label(self, label: str, training_set) -> float:          # TODO: This will be helpful for Task 2.2 b)
        """
        Get the average relative amount of NaN values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        self.rel_NaN_for_label = (self.get_NaN_amount_for_label(label, training_set) / self.non_NaN_amount_for_label[label])

        return self.rel_NaN_for_label

    def get_rel_NaN_amount(self, training_set) -> float:
        """
        Get the relative amount of NaN values across all Patient objects in this set
        :return:
        """
        r = self.get_total_NaN_amount(training_set) / self.get_data_amount(training_set)

        return r

    def get_non_NaN_amount_for_label(self, label: str, training_set) -> int:  # also gets calculated with avg_label
        """
        Get the amount of non-NaN values for the label across all Patient objects in this set
        :param label:
        :param no_cache:
        :return:
        """
        if label not in self.non_NaN_amount_for_label.keys() or \
                self.non_NaN_amount_for_label[label] is None:
            s = 0
            for patient in training_set.data.values():
                s += len(patient.data[label].dropna())
            self.non_NaN_amount_for_label[label] = s

        return self.non_NaN_amount_for_label[label]

    def get_non_NaN_amount(self, training_set) -> int:
        """
        Get the amount of non-NaN values across all Patient objects in this set
        :return:
        """
        count = 0
        for label in Patient.LABELS:
            count += self.get_non_NaN_amount_for_label(label, training_set)

        return count

    def get_rel_non_NaN_amount_for_label(self, label: str, training_set) -> float:
        """
        Get the average relative amount of non-NaN values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        r = 1 - self.get_rel_NaN_amount_for_label(label, training_set)

        return r

    def get_rel_non_NaN_amount(self, training_set) -> float:
        """
        Get the relative amount of non-NaN values across all Patient objects in this set
        :return:
        """
        r = 1 - self.get_rel_NaN_amount(training_set)

        return r

    def get_data_amount_for_label(self, label: str, training_set) -> float:
        """
        Get the amount of values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        return self.get_NaN_amount_for_label(label, training_set) / self.get_non_NaN_amount_for_label(label, training_set)

    def get_data_amount(self, training_set) -> int:
        """
        Get the amount of values across all Patient objects in this set
        :return:
        """
        r = self.get_total_NaN_amount(training_set) + self.get_non_NaN_amount()

        return r

    def get_min_data_duration(self, training_set) -> Tuple[str, int]:
        """
        Get the minimal amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.min_data_duration is None:
            # not calculated before, calculating now

            min_value: int = None
            min_patient: str = None
            for patient in training_set.data.values():
                if min_value is None or min_value > len(patient.data):
                    min_value = len(patient.data)
                    min_patient = patient.ID

            self.min_data_duration = (min_patient, min_value)

        return self.min_data_duration

    def get_max_data_duration(self, training_set) -> Tuple[str, int]:
        """
        Get the minimal amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.max_data_duration is None:
            # not calculated before, calculating now

            max_value: int = None
            max_patient: str = None
            for patient in training_set.data.values():
                if max_value is None or max_value < len(patient.data):
                    max_value = len(patient.data)
                    max_patient = patient.ID

            self.max_data_duration = (max_patient, max_value)

        return self.max_data_duration

    def get_avg_data_duration(self, training_set) -> float:
        """
        Get the average amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.avg_data_duration is None:
            # not calculated before, calculating now
            avg_sum = 0
            avg_count = 0
            for patient in training_set.data.values():
                avg_count += 1
                avg_sum += len(patient.data)

            self.avg_data_duration = avg_sum / avg_count

        return self.avg_data_duration

    def get_sepsis_patients(self, training_set) -> List[str]:
        if self.sepsis_patients is None:
            self.sepsis_patients = []
            for patient in training_set.data.values():
                if patient.data["SepsisLabel"].dropna().sum() > 0:  # count patients not time series
                    self.sepsis_patients.append(patient.ID)

        return self.sepsis_patients

    def get_rel_sepsis_amount(self, training_set) -> float:
        """
        Get the relative amount of patients that develop sepsis across all Patient objects in this set
        :return:
        """
        self.rel_sepsis_amount = len(self.get_sepsis_patients(training_set))/len(training_set.data.keys())

        return self.rel_sepsis_amount

    def get_plot_label_to_sepsis(self, label: str, training_set):
        """
        Gets the selected labels for patient with and without sepsis
        :return:
        """
        if label not in self.plot_label_to_sepsis.keys() or not self.plot_label_to_sepsis:
            sepsis_pos = []
            sepsis_neg = []
            for patient in training_set.data.values():
                label_vals = patient.data[label]                    # counts amount of time series not patient level!
                for label_val in label_vals:
                    if pd.notna(label_val):
                        if int(patient.data["SepsisLabel"][1]) == 1:
                            sepsis_pos.append(float(label_val))
                        else:
                            sepsis_neg.append(float(label_val))

            self.plot_label_to_sepsis[label] = (
                sepsis_pos, sepsis_neg)  # pos = plot_data[0] and neg = plot_data[1]

        return self.plot_label_to_sepsis

    def save_analysis_obj_to_cache(self):
        print(CompleteAnalysis.CACHE_PATH, self.analysis_cache_name)
        pickle.dump(self, open(os.path.join(CompleteAnalysis.CACHE_PATH, self.analysis_cache_name), "wb"))
        print("Analysis Object was cached into file", self.analysis_cache_name,
              " At time: ", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))

    def save_analysis_to_cache(self):
        pickle_data = {
            "min_for_label": self.min_for_label,
            "max_for_label": self.max_for_label,
            "avg_for_label": self.avg_for_label,
            "rel_NaN_for_label": self.rel_NaN_for_label,
            "non_NaN_amount_for_label": self.non_NaN_amount_for_label,
            "min_data_duration": self.min_data_duration,
            "max_data_duration": self.max_data_duration,
            "avg_data_duration": self.avg_data_duration,
            "rel_sepsis_amount": self.rel_sepsis_amount,
            "plot_label_to_sepsis": self.plot_label_to_sepsis
        }
        pickle.dump(pickle_data, open(os.path.join(CompleteAnalysis.CACHE_PATH, self.analysis_cache_name), "wb"))
        print("Analysis was cached into file", self.analysis_cache_name,
              " At Time: ", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))

    def save_analysis_to_JSON(self):
        json_file_name = self.analysis_cache_name + "_2_JSON.json"
        json_data = {
            "min_for_label": self.min_for_label,
            "max_for_label": self.max_for_label,
            "avg_for_label": self.avg_for_label,
            "rel_NaN_for_label": self.rel_NaN_for_label,
            "non_NaN_amount_for_label": self.non_NaN_amount_for_label,
            "min_data_duration": self.min_data_duration,
            "max_data_duration": self.max_data_duration,
            "avg_data_duration": self.avg_data_duration,
            "rel_sepsis_amount": self.rel_sepsis_amount,
            "plot_label_to_sepsis": self.plot_label_to_sepsis
        }
        frozen = jsonpickle.encode(json_data)
        json.dump(frozen,
                  open(os.path.join(CompleteAnalysis.CACHE_PATH, json_file_name), "w"))  # needed to change wb to w
        print("Analysis was cached into JSON", json_file_name,
              " At Time: ", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))
