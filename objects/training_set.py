from typing import Dict, Tuple, List
import os
import hashlib
import pickle

import pandas as pd

from objects.patient import Patient


class TrainingSet:

    CACHE_PATH = os.path.join(".", "cache")

    def __init__(self, set_id: str, patients: Dict[str, Patient], use_cache: bool = True):
        self.__set_id = set_id
        self.__data = patients

        key_concat = ""
        keys = list(patients.keys())
        keys.sort()
        for key in keys:
            key_concat += key
        self.__cache_file_name = hashlib.md5(key_concat.encode("utf-8")).hexdigest() + ".pickle"

        if os.path.isfile(os.path.join(TrainingSet.CACHE_PATH, self.__cache_file_name)) and use_cache:  # cache exists?
            print(f"Found cache! TrainingSet {self.__set_id} uses", self.__cache_file_name)
            self.__load_from_cache()
            return
        else:
            print("Found no cache!", self.__cache_file_name)

        # caching variables
        self.__min_for_label: Dict[str, Tuple[str, float]] = {}
        self.__max_for_label: Dict[str, Tuple[str, float]] = {}
        self.__avg_for_label: Dict[str, float] = {}
        self.__NaN_amount_for_label: Dict[str, int] = {}
        self.__non_NaN_amount_for_label: Dict[str, int] = {}
        self.__min_data_duration: Tuple[str, int] = None
        self.__max_data_duration: Tuple[str, int] = None
        self.__avg_data_duration: float = None
        self.__sepsis_patients: List[str] = None

    def __len__(self) -> int:
        """
        Returns amount of patients in this set
        :return:
        """
        return len(self.data.keys())

    @property
    def ID(self) -> str:
        return self.__set_id

    @property
    def data(self) -> Dict[str, Patient]:
        return self.__data

    def get_subgroup(self, label: str, low_value, high_value):
        """
        Split this set into a sub set based on low_value and high_value range
        :param label:
        :param low_value:
        :param high_value:
        :return:
        """
        raise NotImplementedError

    @property
    def sepsis_patients(self) -> List[str]:
        if self.__sepsis_patients is None:
            # not calculated before, calculating now
            self.__sepsis_patients = []
            for patient in self.data.values():
                if patient.data["SepsisLabel"].dropna().sum() > 0:  # at least one sepsis
                    self.__sepsis_patients.append(patient.ID)

            self.__save_to_cache()

        return self.__sepsis_patients

    def get_min_for_label(self, label: str) -> Tuple[str, float]:
        """
        Get the minimal value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.__min_for_label.keys() or self.__min_for_label[label] is None:
            # was not calculated before, calculating now

            min_patient: str = None
            min_value: float = None

            for patient in self.data.values():
                v = patient.data[label].min()
                if pd.isna(v):
                    continue
                if min_value is None or min_value > v:
                    min_value = v
                    min_patient = patient.ID

            self.__min_for_label[label] = (min_patient, min_value)
            self.__save_to_cache()

        return self.__min_for_label[label]

    def get_max_for_label(self, label: str) -> Tuple[str, float]:
        """
        Get the maximal value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.__max_for_label.keys() or self.__max_for_label[label] is None:
            # was not calculated before, calculating now

            max_patient: str = None
            max_value: float = None

            for patient in self.data.values():
                v = patient.data[label].max()
                if pd.isna(v):
                    continue
                if max_value is None or max_value < v:
                    max_value = v
                    max_patient = patient.ID

            self.__max_for_label[label] = (max_patient, max_value)
            self.__save_to_cache()

        return self.__max_for_label[label]

    def get_avg_for_label(self, label: str) -> float:
        """
        Get the average value for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.__avg_for_label.keys() or self.__avg_for_label[label] is None:
            # was not calculated before, calculating now

            s = 0
            count = 0

            for patient in self.data.values():
                series = patient.data[label].dropna()
                s += series.sum()
                count += len(series)

            self.__non_NaN_amount_for_label[label] = count  # caching side result
            if count > 0:
                average = s / count
            else:
                average = None

            self.__avg_for_label[label] = average
            self.__save_to_cache()

        return self.__avg_for_label[label]

    def get_NaN_amount_for_label(self, label: str, no_cache: bool = False) -> int:
        """
        Get the amount of NaN values for the label across all Patient objects in this set
        :param label:
        :param no_cache:
        :return:
        """
        if label not in self.__NaN_amount_for_label.keys() or self.__NaN_amount_for_label[label] is None:
            # was not calculated before, calculating now

            count = 0
            for patient in self.data.values():
                count += patient.data[label].isnull().sum()

            self.__NaN_amount_for_label[label] = count
            if not no_cache:
                self.__save_to_cache()

        return self.__NaN_amount_for_label[label]

    def get_NaN_amount(self, no_cache: bool = False) -> int:
        """
        Get the amount of NaN values across all Patient objects in this set
        :return:
        """
        count = 0
        for label in Patient.LABELS:
            count += self.get_NaN_amount_for_label(label, no_cache=True)

        if not no_cache:
            self.__save_to_cache()
        return count

    def get_avg_rel_NaN_amount_for_label(self, label: str, no_cache: bool = False) -> float:
        """
        Get the average relative amount of NaN values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        r = self.get_NaN_amount_for_label(label, no_cache=True) / self.get_data_amount_for_label(label, no_cache=True)
        if not no_cache:
            self.__save_to_cache()
        return r

    def get_rel_NaN_amount(self, no_cache: bool = False) -> float:
        """
        Get the relative amount of NaN values across all Patient objects in this set
        :return:
        """
        r = self.get_NaN_amount(no_cache=True) / self.get_data_amount(no_cache=True)
        if not no_cache:
            self.__save_to_cache()
        return r

    def get_data_amount_for_label(self, label: str, no_cache: bool = False) -> float:
        """
        Get the amount of values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        return self.get_NaN_amount_for_label(label, no_cache=True) + \
               self.get_non_NaN_amount_for_label(label, no_cache=True)

    def get_data_amount(self, no_cache: bool = False) -> int:
        """
        Get the amount of values across all Patient objects in this set
        :return:
        """
        r = self.get_NaN_amount(no_cache=True) + self.get_non_NaN_amount(no_cache=True)
        if not no_cache:
            self.__save_to_cache()
        return r

    def get_non_NaN_amount_for_label(self, label: str, no_cache: bool = False) -> int:
        """
        Get the amount of non-NaN values for the label across all Patient objects in this set
        :param label:
        :param no_cache:
        :return:
        """
        if label not in self.__non_NaN_amount_for_label.keys() or self.__non_NaN_amount_for_label[label] is None:
            # was not calculated before, calculating now

            s = 0
            for patient in self.data.values():
                s += len(patient.data[label].dropna())

            self.__non_NaN_amount_for_label[label] = s
            if not no_cache:
                self.__save_to_cache()

        return self.__non_NaN_amount_for_label[label]

    def get_non_NaN_amount(self, no_cache: bool = False) -> int:
        """
        Get the amount of non-NaN values across all Patient objects in this set
        :return:
        """
        count = 0
        for label in Patient.LABELS:
            count += self.get_non_NaN_amount_for_label(label, no_cache=True)

        if not no_cache:
            self.__save_to_cache()
        return count

    def get_avg_rel_non_NaN_amount_for_label(self, label: str, no_cache: bool = False) -> float:
        """
        Get the average relative amount of non-NaN values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        r = 1 - self.get_avg_rel_NaN_amount_for_label(label, no_cache=True)
        if not no_cache:
            self.__save_to_cache()
        return r

    def get_rel_non_NaN_amount(self, no_cache: bool = False) -> float:
        """
        Get the relative amount of non-NaN values across all Patient objects in this set
        :return:
        """
        r = 1 - self.get_rel_NaN_amount(no_cache=True)
        if not no_cache:
            self.__save_to_cache()
        return r

    def get_min_data_duration(self, no_cache: bool = False) -> Tuple[str, int]:
        """
        Get the minimal amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.__min_data_duration is None:
            # not calculated before, calculating now

            min_value: int = None
            min_patient: str = None
            for patient in self.data.values():
                if min_value is None or min_value > len(patient.data):
                    min_value = len(patient.data)
                    min_patient = patient.ID

            self.__min_data_duration = (min_patient, min_value)
            if not no_cache:
                self.__save_to_cache()

        return self.__min_data_duration

    def get_max_data_duration(self, no_cache: bool = False) -> Tuple[str, int]:
        """
        Get the minimal amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.__max_data_duration is None:
            # not calculated before, calculating now

            max_value: int = None
            max_patient: str = None
            for patient in self.data.values():
                if max_value is None or max_value < len(patient.data):
                    max_value = len(patient.data)
                    max_patient = patient.ID

            self.__max_data_duration = (max_patient, max_value)
            if not no_cache:
                self.__save_to_cache()

        return self.__max_data_duration

    def get_avg_data_duration(self, no_cache: bool = False) -> float:
        """
        Get the average amount or duration of data across all Patient objects in this set

        Note: one data point is the result of one hour of measurement in real time
        :return:
        """
        if self.__avg_data_duration is None:
            # not calculated before, calculating now

            avg_sum = 0
            avg_count = 0
            for patient in self.data.values():
                avg_count += 1
                avg_sum += len(patient.data)

            self.__avg_data_duration = avg_sum / avg_count
            if not no_cache:
                self.__save_to_cache()

        return self.__avg_data_duration

    def get_sepsis_amount(self) -> int:
        """
        Get the amount of patients that develop sepsis across all Patient objects in this set
        :return:
        """
        return len(self.sepsis_patients)

    def get_rel_sepsis_amount(self, no_cache: bool = False) -> float:
        """
        Get the relative amount of patients that develop sepsis across all Patient objects in this set
        :return:
        """
        r = self.get_sepsis_amount() / len(self)
        if not no_cache:
            self.__save_to_cache()
        return r

    def __load_from_cache(self):
        pickle_data = pickle.load(open(os.path.join(TrainingSet.CACHE_PATH, self.__cache_file_name), "rb"))

        self.__min_for_label = pickle_data["min_for_label"]
        self.__max_for_label = pickle_data["max_for_label"]
        self.__avg_for_label = pickle_data["avg_for_label"]
        self.__NaN_amount_for_label = pickle_data["NaN_amount_for_label"]
        self.__non_NaN_amount_for_label = pickle_data["non_NaN_amount_for_label"]
        self.__min_data_duration = pickle_data["min_data_duration"]
        self.__max_data_duration = pickle_data["max_data_duration"]
        self.__avg_data_duration = pickle_data["avg_data_duration"]
        self.__sepsis_patients = pickle_data["sepsis_patients"]

    def __save_to_cache(self):
        pickle_data = {
            "min_for_label": self.__min_for_label,
            "max_for_label": self.__max_for_label,
            "avg_for_label": self.__avg_for_label,
            "NaN_amount_for_label": self.__NaN_amount_for_label,
            "non_NaN_amount_for_label": self.__non_NaN_amount_for_label,
            "min_data_duration": self.__min_data_duration,
            "max_data_duration": self.__max_data_duration,
            "avg_data_duration": self.__avg_data_duration,
            "sepsis_patients": self.__sepsis_patients
        }
        pickle.dump(pickle_data, open(os.path.join(TrainingSet.CACHE_PATH, self.__cache_file_name), "wb"))
