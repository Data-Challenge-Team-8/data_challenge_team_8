from typing import Dict, Tuple, List

import numpy as np
import pandas as pd

from objects.patient import Patient


class TrainingSet:

    def __init__(self, set_id: str, patients: Dict[str, Patient]):
        self.__set_id = set_id
        self.__data = patients

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

    @property
    def sepsis_patients(self) -> List[str]:
        if self.__sepsis_patients is None:
            # not calculated before, calculating now
            self.__sepsis_patients = []
            for patient in self.data.values():
                if patient.data["SepsisLabel"].dropna().sum() > 0:  # at least one sepsis
                    self.__sepsis_patients.append(patient.ID)

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

        return self.__avg_for_label[label]

    def get_NaN_amount_for_label(self, label: str) -> int:
        """
        Get the amount of NaN values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.__NaN_amount_for_label.keys() or self.__NaN_amount_for_label[label] is None:
            # was not calculated before, calculating now

            count = 0
            for patient in self.data.values():
                count += patient.data[label].isnull().sum()

            self.__NaN_amount_for_label[label] = count

        return self.__NaN_amount_for_label[label]

    def get_NaN_amount(self) -> int:
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

        return self.get_NaN_amount_for_label(label) / self.get_data_amount_for_label(label)

    def get_rel_NaN_amount(self) -> float:
        """
        Get the relative amount of NaN values across all Patient objects in this set
        :return:
        """

        return self.get_NaN_amount() / self.get_data_amount()

    def get_data_amount_for_label(self, label: str) -> float:
        """
        Get the amount of values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        return self.get_NaN_amount_for_label(label) + self.get_non_NaN_amount_for_label(label)

    def get_data_amount(self) -> int:
        """
        Get the amount of values across all Patient objects in this set
        :return:
        """
        return self.get_NaN_amount() + self.get_non_NaN_amount()

    def get_non_NaN_amount_for_label(self, label: str) -> int:
        """
        Get the amount of non-NaN values for the label across all Patient objects in this set
        :param label:
        :return:
        """
        if label not in self.__non_NaN_amount_for_label.keys() or self.__non_NaN_amount_for_label[label] is None:
            # was not calculated before, calculating now

            s = 0
            for patient in self.data.values():
                s += len(patient.data[label].dropna())

            self.__non_NaN_amount_for_label[label] = s

        return self.__non_NaN_amount_for_label[label]

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
        return 1 - self.get_avg_rel_NaN_amount_for_label(label)

    def get_rel_non_NaN_amount(self) -> float:
        """
        Get the relative amount of non-NaN values across all Patient objects in this set
        :return:
        """
        return 1 - self.get_rel_NaN_amount()

    def get_min_data_duration(self) -> Tuple[str, int]:
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

        return self.__min_data_duration

    def get_max_data_duration(self) -> Tuple[str, int]:
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

        return self.__max_data_duration

    def get_avg_data_duration(self) -> float:
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

        return self.__avg_data_duration

    def get_sepsis_amount(self) -> int:
        """
        Get the amount of patients that develop sepsis across all Patient objects in this set
        :return:
        """
        return len(self.sepsis_patients)

    def get_rel_sepsis_amount(self) -> float:
        """
        Get the relative amount of patients that develop sepsis across all Patient objects in this set
        :return:
        """
        return self.get_sepsis_amount() / len(self)
