import hashlib
from typing import Dict, Tuple, List
import os
import pickle
import datetime
import pandas as pd
import numpy as np

import numpy as np
import pandas as pd

import numpy as np
import pandas as pd

from objects.patient import Patient
from IO.data_reader import DataReader


class TrainingSet:
    CACHE_PATH = os.path.join(".", "cache")
    CACHE_FILE_PREFIX = "trainingset_data-"

    PRESETS = {
        "Set A": [file_name.split(".")[0] for file_name in os.listdir(os.path.join(".", "data", "training_setA"))],
        "Set B": [file_name.split(".")[0] for file_name in os.listdir(os.path.join(".", "data", "training_setB"))],
        "Set A + B": [file_name.split(".")[0] for file_name in os.listdir(os.path.join(".", "data", "training_setA"))]
                     + [file_name.split(".")[0] for file_name in os.listdir(os.path.join(".", "data", "training_setB"))]
    }

    __instances = {}

    def __init__(self, patients: List[str], name: str):
        TrainingSet.__instances[name] = self

        self.name = name
        self.data = {key: None for key in patients}
        self.cache_name = self.__construct_cache_file_name()

        self.active_labels = []

        self.__dirty: bool = False

        self.average_df_fixed: pd.DataFrame = None
        self.labels_average = {}
        self.labels_std_dev = {}
        self.labels_rel_NaN = {}

        self.__load_data_from_cache()
        self.__save_data_to_cache()

    def get_active_labels(self):  # get labels from first entry(patient) in data_dict, we could implement label filtering here
        self.active_labels = list(self.data.values())[0].data.columns.values
        return self.active_labels

    def calc_stats_for_labels(self):
        self.get_active_labels()
        labels_average_dict = dict.fromkeys(self.active_labels)
        labels_std_dev_dict = dict.fromkeys(self.active_labels)
        labels_rel_NaN_dict = dict.fromkeys(self.active_labels)
        for label in self.active_labels[0:1]:                               # TODO: Only first selected label
            print("Analysing Label: ", label)
            averages_list = []
            std_dev_list = []
            rel_nan_list = []
            for patient in self.data.values():
                averages_list.append(patient.get_average(label))
                std_dev_list.append(patient.get_standard_deviation(label))
                rel_nan_list.append(patient.get_NaN(label))
            print("List of averages per patient", averages_list[:10])       # All elements in averages list are definitely float (and not numpy.float)
            labels_average_dict[label] = np.nansum(averages_list) / len(averages_list)          # TODO: 1) Error with sum(list) method ???
            labels_std_dev_dict[label] = np.nansum(std_dev_list) / len(std_dev_list)
            labels_rel_NaN_dict[label] = sum(rel_nan_list) / len(rel_nan_list)
        self.labels_average.update(labels_average_dict)                     # TODO: Is update a good solution?
        self.labels_std_dev.update(labels_std_dev_dict)
        self.labels_rel_NaN.update(labels_rel_NaN_dict)
        return self.labels_average, self.labels_std_dev, self.labels_rel_NaN

    def get_dataframe_averages(self):                                       # TODO: Test this implementation. Is patient_id missing?
        data_rows = []
        for patient in self.data.values():
            data_rows.append(list(patient.labels_average.values()))
        df = pd.DataFrame(data_rows, columns=self.active_labels)
        return df

    def __len__(self):
        return len(self.data.keys())

    def __construct_cache_file_name(self):
        key_concat = ""
        for patient_id in self.data.keys():
            key_concat += patient_id

        return hashlib.md5(key_concat.encode('utf-8')).hexdigest() + ".pickle"

    def __load_data_from_cache(self, force_no_cache: bool = False):
        file_path = os.path.join(TrainingSet.CACHE_PATH, TrainingSet.CACHE_FILE_PREFIX + self.cache_name)
        if not force_no_cache and os.path.exists(file_path) and os.path.isfile(file_path):
            print(f"Loading TrainingSet {self.name} data from pickle cache")
            start_time = datetime.datetime.now()
            d = pickle.load(open(file_path, "rb"))
            self.data = d["data"]
            self.average_df_fixed = d["avg_df_fixed"]
            end_time = datetime.datetime.now()
            print("Took", end_time - start_time, "to load from pickle!")
        else:
            print(f"Loading TrainingSet {self.name} data from DataReader")
            start_time = datetime.datetime.now()
            for key in self.data.keys():
                if self.data[key] is None:  # Patient not loaded yet
                    self.data[key] = DataReader.get_instance().get_patient(key)
            end_time = datetime.datetime.now()
            print("Took", end_time - start_time, "to load from DataReader!")

    def __save_data_to_cache(self):
        file_path = os.path.join(TrainingSet.CACHE_PATH, TrainingSet.CACHE_FILE_PREFIX + self.cache_name)
        if not os.path.exists(file_path) or self.__dirty:
            print("Writing TrainingSet", self.name, "data to pickle cache!")
            self.__dirty = False
            pickle.dump({"data": self.data, "avg_df_fixed": self.average_df_fixed}, open(file_path, "wb"))

    @classmethod
    def get_training_set(cls, name: str, patients: List[str] = None):
        """
        Fetches an instance of the training set specified by set_name.

        If the set_name is not already used or a preset an Exception is thrown.
        :param patients: list of patient ids to be included into the TrainingSet should it not exist
        :param name:
        :return:
        """
        if name in TrainingSet.__instances.keys():
            return TrainingSet.__instances[name]
        elif name in TrainingSet.PRESETS.keys():
            return TrainingSet(TrainingSet.PRESETS[name], name)
        else:
            if patients is None:
                raise ValueError("Unknown Training Set name")
            else:
                return TrainingSet(patients=patients, name=name)

    def get_average_df(self, use_interpolation: bool = False, fix_missing_values: bool = False):
        """
        Calculate a concatenated dataframe of averages for each label/feature of each patient.

        Important: Drops the sepsis label!
        Note: PacMAP expects a (number of samples, dimension) format. This is (dimension, number of samples).
        So use transpose()!
        :param use_interpolation: if interpolation is used by the patient before calculating averages
        :param fix_missing_values: Attempts to remove NaN values from the dataframe by either imputation or
        row removal. Method decision is based on Patient.NAN_DISMISSAL_THRESHOLD.
        :return:
        """
        if self.average_df_fixed is not None and fix_missing_values:
            return self.average_df_fixed

        avg_dict = {}
        for patient_id in self.data.keys():
            avg_dict[patient_id] = self.data[patient_id].get_average_df(use_interpolation)

        avg_df = pd.DataFrame(avg_dict)
        avg_df.drop("SepsisLabel", inplace=True)

        if not fix_missing_values:
            return avg_df
        else:

            label_avgs = self.get_label_averages(use_interpolation)

            for label in avg_df.index:
                rel_missing = avg_df.loc[label].isna().sum() / len(avg_df.loc[label])
                print()

                if rel_missing >= Patient.NAN_DISMISSAL_THRESHOLD/1.5:  # kick the row because too many missing values
                    print(f"TrainingSet.get_average_df kicked \"{label}\" out because too many missing values "
                          f"({rel_missing} > {Patient.NAN_DISMISSAL_THRESHOLD/2})")
                    avg_df.drop(label, inplace=True)

                else:  # try filling with mean imputation
                    for patient_id in avg_dict.keys():
                        if avg_df.isna()[patient_id][label]:
                            avg_df[patient_id][label] = label_avgs[label]
            self.average_df_fixed = avg_df
            self.__dirty = True
            self.__save_data_to_cache()
            return self.average_df_fixed

    def get_label_averages(self, use_interpolation: bool = False) -> pd.Series:
        """
        Calculate the average of a label across the whole set

        :param use_interpolation:
        :return:
        """
        label_sums = {key: 0 for key in Patient.LABELS}
        label_count = {key: 0 for key in Patient.LABELS}
        for patient_id in self.data.keys():
            for label in Patient.LABELS:
                if not use_interpolation:
                    label_sums[label] += self.data[patient_id].data[label].dropna().sum()
                    label_count[label] += len(self.data[patient_id].data[label].dropna())
                else:
                    label_sums[label] += self.data[patient_id].get_interp_data()[label].dropna().sum()
                    label_count[label] += len(self.data[patient_id].get_interp_data()[label].dropna())

        avgs = {}
        for label in label_sums.keys():
            if label_count[label] == 0:
                avgs[label] = None
            else:
                avgs[label] = label_sums[label] / label_count[label]

        return pd.Series(avgs)

    def get_subgroup(self, label: str, low_value, high_value, new_set_id: str = None):
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
            if low_value <= patient.data[label].min() and patient.data[label].max() <= high_value:
                subgroup_dict[patient.ID] = patient

        if len(subgroup_dict.keys()) != 0:
            if new_set_id is None:
                return TrainingSet.get_training_set(
                    name=self.name + f"-SubGroup_{label}_low{low_value}_high{high_value}",
                    patients=list(subgroup_dict.keys()))
            else:
                return TrainingSet.get_training_set(name=new_set_id, patients=list(subgroup_dict.keys()))
        else:
            return None
