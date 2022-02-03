import hashlib
from typing import Dict, Tuple, List
import os
import pickle
import datetime
import random

import numpy as np
import pandas as pd

from objects.patient import Patient
from IO.data_reader import DataReader
from tools.pacmap_analysis import calculate_pacmap


def get_set_a() -> List[str]:
    return [file_name.split(".")[0] for file_name in os.listdir(os.path.join(".", "data", "training_setA"))]


def get_set_b() -> List[str]:
    return [file_name.split(".")[0] for file_name in os.listdir(os.path.join(".", "data", "training_setB"))]


def get_rnd_sample_a() -> List[str]:
    random.seed(1337)
    sample = []
    set_a = get_set_a()
    for i in range(200):
        sample.append(set_a[random.randint(0, len(set_a))])
    return sample


class TrainingSet:
    CACHE_PATH = os.path.join(".", "cache")
    CACHE_FILE_PREFIX = "trainingset_data"
    CACHE_FILE_BASIC_POSTFIX = "basic"
    CACHE_FILE_AVG_POSTFIX = "averages"
    CACHE_FILE_TS_POSTFIX = "timeseries"
    CACHE_FILE_PACMAP_POSTFIX = "pacmap"

    PRESETS = {
        "Set A": get_set_a,
        "Set B": get_set_b,
        "Set A + B": lambda: get_set_a() + get_set_b(),
        "rnd Sample A": get_rnd_sample_a,
    }

    __instances = {}

    def __init__(self, patients: List[str], name: str):
        TrainingSet.__instances[name] = self

        self.name = name
        self.data: Dict[str, Patient] = {key: None for key in patients}
        self.cache_name = self.__construct_cache_file_name()

        # TODO: Besprechen brauchen wir die beiden auskommentieren avg cases doch auch?
        # self.avg_df_no_fixed: pd.DataFrame = None
        self.average_df_fixed_no_interpol: pd.DataFrame = None
        self.average_df_fixed_interpol: pd.DataFrame = None

        self.z_value_df_no_interpol: pd.DataFrame = None
        # self.z_value_df_fixed_no_interpol: pd.DataFrame = None       # Und das?
        self.z_value_df: pd.DataFrame = None

        self.__pacmap_2d_no_interpol = None
        self.__pacmap_3d_no_interpol = None
        self.__pacmap_2d_interpol = None
        self.__pacmap_3d_interpol = None

        self.__timeseries_interpol_no_fix = None
        self.__timeseries_interpol_fix = None
        self.__timeseries_no_interpol = None
        self.__timeseries_no_interpol_fix = None

        if not self.__load_data_from_cache():  # if we couldn't load basics from cache, create one for next time
            self.__save_data_to_cache()

    def __len__(self):
        return len(self.data.keys())

    def __construct_cache_file_name(self) -> str:
        key_concat = ""
        for patient_id in self.data.keys():
            key_concat += patient_id

        return hashlib.md5(key_concat.encode('utf-8')).hexdigest()

    def __load_data_from_cache(self, force_no_cache: bool = False):
        # basic/patient data
        loaded_basics_from_cache = None
        file_path = self.get_cache_file_path(TrainingSet.CACHE_FILE_BASIC_POSTFIX)
        if not force_no_cache and os.path.isfile(file_path):
            print(f"Loading TrainingSet {self.name} patient data from pickle cache")
            print(f"Cache file path is: {file_path}")
            loaded_basics_from_cache = True
            start_time = datetime.datetime.now()
            self.data = pickle.load(open(file_path, "rb"))
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
            loaded_basics_from_cache = False

        # avg_df data
        file_path = self.get_cache_file_path(TrainingSet.CACHE_FILE_AVG_POSTFIX)
        if not force_no_cache and os.path.isfile(file_path):
            print(f"Loading TrainingSet {self.name} average data from pickle cache")
            print(f"Cache file path is: {file_path}")
            start_time = datetime.datetime.now()
            d = pickle.load(open(file_path, 'rb'))
            self.average_df_fixed_no_interpol = d["fixed_no_interpolation"]
            self.average_df_fixed_interpol = d["fixed_interpolation"]
            end_time = datetime.datetime.now()
            print("Took", end_time - start_time, "to load from pickle!")
        else:
            print("Found no pickle cache for average data!")

        # pacmap data
        file_path = self.get_cache_file_path(TrainingSet.CACHE_FILE_PACMAP_POSTFIX)
        if not force_no_cache and os.path.isfile(file_path):
            print(f"Loading TrainingSet {self.name} PaCMAP data from pickle cache")
            print(f"Cache file path is: {file_path}")
            start_time = datetime.datetime.now()
            d = pickle.load(open(file_path, 'rb'))
            self.__pacmap_2d_no_interpol = d["no_interpolation"]["2d"]
            self.__pacmap_3d_no_interpol = d["no_interpolation"]["3d"]
            self.__pacmap_2d_interpol = d["interpolation"]["2d"]
            self.__pacmap_3d_interpol = d["interpolation"]["3d"]
            end_time = datetime.datetime.now()
            print("Took", end_time - start_time, "to load from pickle!")

        # timeseries data
        file_path = self.get_cache_file_path(TrainingSet.CACHE_FILE_TS_POSTFIX)
        if not force_no_cache and os.path.isfile(file_path):
            print(f"Loading TrainingSet {self.name} interp timeseries data from pickle cache")
            print(f"Cache file path is: {file_path}")
            start_time = datetime.datetime.now()
            d = pickle.load(open(file_path, 'rb'))
            self.__timeseries_no_interpol = d["no_interpolation"]["no_fix"]
            self.__timeseries_no_interpol_fix = d["no_interpolation"]["fix"]
            self.__timeseries_interpol_no_fix = d["interpolation"]["no_fix"]
            self.__timeseries_interpol_fix = d["interpolation"]["fix"]
            end_time = datetime.datetime.now()
            print("Took", end_time - start_time, "to load from pickle!")

    def __save_data_to_cache(self):
        self.__save_basic_data_to_cache()
        if self.average_df_fixed_interpol is not None or self.average_df_fixed_no_interpol is not None:
            self.__save_average_data_to_cache()
        if self.__pacmap_2d_interpol is not None or self.__pacmap_2d_no_interpol is not None or \
                self.__pacmap_3d_no_interpol is not None or self.__pacmap_3d_interpol is not None:
            self.__save_pacmap_data_to_cache()
        if self.__timeseries_interpol_no_fix is not None or self.__timeseries_no_interpol_fix is not None or \
                self.__timeseries_interpol_fix is not None or self.__timeseries_interpol_no_fix is not None:
            self.__save_timeseries_data_to_cache()

    def __save_basic_data_to_cache(self):
        # basic/patient data
        file_path = self.get_cache_file_path(TrainingSet.CACHE_FILE_BASIC_POSTFIX)
        print("Writing TrainingSet", self.name, "patient data to pickle cache!")
        pickle.dump(self.data, open(file_path, "wb"))

    def __save_average_data_to_cache(self):
        # average data
        file_path = self.get_cache_file_path(TrainingSet.CACHE_FILE_AVG_POSTFIX)
        print("Writing TrainingSet", self.name, "average data to pickle cache!")
        pickle.dump({"fixed_no_interpolation": self.average_df_fixed_no_interpol,
                     "fixed_interpolation": self.average_df_fixed_interpol},
                    open(file_path, "wb"))

    def __save_pacmap_data_to_cache(self):
        # pacmap data
        file_path = self.get_cache_file_path(TrainingSet.CACHE_FILE_PACMAP_POSTFIX)
        print(f"Writing TrainingSet {self.name} PaCMAP data to pickle cache!")
        pickle.dump({"no_interpolation": {"2d": self.__pacmap_2d_no_interpol, "3d": self.__pacmap_3d_no_interpol},
                     "interpolation": {"2d": self.__pacmap_2d_interpol, "3d": self.__pacmap_3d_interpol}},
                    open(file_path, 'wb'))

    def __save_timeseries_data_to_cache(self):
        file_path = self.get_cache_file_path(TrainingSet.CACHE_FILE_TS_POSTFIX)
        print(f"Writing TrainingSet {self.name} Timeseries data to pickle cache!")
        pickle.dump({"no_interpolation": {"no_fix": self.__timeseries_no_interpol,
                                          "fix": self.__timeseries_no_interpol_fix},
                     "interpolation": {"no_fix": self.__timeseries_interpol_no_fix,
                                       "fix": self.__timeseries_interpol_fix}},
                    open(file_path, 'wb'))

    def get_cache_file_path(self, postfix: str):
        """
        Get the full file path to a cache file with the given file postfix.
        :param postfix:
        :return:
        """
        return os.path.join(TrainingSet.CACHE_PATH, TrainingSet.CACHE_FILE_PREFIX+f"-{self.cache_name}-{postfix}.pickle")

    def get_feature(self, feature: str) -> pd.DataFrame:
        """
        Query for a specific feature by name (must be present in Patient) and return a DataFrame with only that feature.
        :param feature:
        :return:
        """
        data = {}
        if feature == "SepsisLabel":
            feature = "sepsis_label"
        for p in self.data.keys():
            data[p] = getattr(self.data[p], feature)

        s = pd.DataFrame(data)
        if feature == "sepsis_label":
            return s.max()
        return s

    def get_patient_from_id(self, patient_id) -> Patient:
        return self.data[patient_id]

    def get_pacmap(self, dimension: int = 2, use_interpolation: bool = False):
        """
        Return the requested PaCMAP data for this training set. Uses precalculated results if available,
        otherwise calculates them.
        :param dimension:
        :param use_interpolation:
        :return:
        """
        pacmap_data = None
        patient_ids = None
        if not use_interpolation:
            if dimension == 2:
                if self.__pacmap_2d_no_interpol is not None:
                    return self.__pacmap_2d_no_interpol, self.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation) \
                .columns.tolist()
                else:
                    self.__pacmap_2d_no_interpol, patient_ids = calculate_pacmap(self, dimension=dimension,
                                                                                 use_interpolation=use_interpolation)
                    self.__save_pacmap_data_to_cache()
                    return self.__pacmap_2d_no_interpol, patient_ids
            elif dimension == 3:
                if self.__pacmap_3d_no_interpol is not None:
                    return self.__pacmap_3d_no_interpol, self.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation) \
                .columns.tolist()
                else:
                    self.__pacmap_3d_no_interpol, patient_ids = calculate_pacmap(self, dimension=dimension,
                                                                                 use_interpolation=use_interpolation)
                    self.__save_pacmap_data_to_cache()
                    return self.__pacmap_3d_no_interpol, patient_ids

        else:  # used interpolation
            if dimension == 2:
                if self.__pacmap_2d_interpol is not None:
                    return self.__pacmap_2d_interpol, self.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation) \
                .columns.tolist()
                else:
                    print(f"Calculating PaCMAP {dimension}D data for TrainingSet {self.name} with"
                          f"{'out' if not use_interpolation else ''} interpolation")
                    self.__pacmap_2d_interpol, patient_ids = calculate_pacmap(self, dimension=dimension,
                                                                              use_interpolation=use_interpolation)
                    self.__save_pacmap_data_to_cache()
                    return self.__pacmap_2d_interpol, patient_ids

            elif dimension == 3:
                if self.__pacmap_3d_interpol is not None:
                    return self.__pacmap_3d_interpol, self.get_average_df(fix_missing_values=True, use_interpolation=use_interpolation) \
                .columns.tolist()
                else:
                    self.__pacmap_3d_interpol, patient_ids = calculate_pacmap(self, dimension=dimension,
                                                                              use_interpolation=use_interpolation)
                    self.__save_pacmap_data_to_cache()
                    return self.__pacmap_3d_interpol, patient_ids

        return pacmap_data, patient_ids

    @classmethod
    def get_training_set(cls, name: str, patients: List[str] = None):
        """
        Fetches an instance of the training set specified by set_name.

        If the set_name is not already used or a preset an Exception is thrown.
        :param patients: list of patient ids to be included into the TrainingSet should it not exist
        :param name:
        :return:
        """
        print(f"Fetching TrainingSet \"{name}\" ...")
        if name in TrainingSet.__instances.keys():
            return TrainingSet.__instances[name]
        elif name in TrainingSet.PRESETS.keys():
            return TrainingSet(TrainingSet.PRESETS[name](), name)
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
        if self.average_df_fixed_no_interpol is not None and fix_missing_values and not use_interpolation:
            return self.average_df_fixed_no_interpol
        elif self.average_df_fixed_interpol is not None and fix_missing_values and use_interpolation:
            return self.average_df_fixed_interpol

        avg_dict = {}
        for patient_id in self.data.keys():
            avg_dict[patient_id] = self.data[patient_id].get_average_df(use_interpolation)

        avg_df = pd.DataFrame(avg_dict)
        avg_df.drop("SepsisLabel", inplace=True)

        if not fix_missing_values:
            return avg_df
        else:
            label_avgs = self.get_label_averages(use_interpolation=use_interpolation)

            for label in avg_df.index:
                rel_missing = avg_df.loc[label].isna().sum() / len(avg_df.loc[label])
                print()

                if rel_missing >= Patient.NAN_DISMISSAL_THRESHOLD / 1.5:  # kick the row because too many missing values
                    print(f"{self.name}.get_average_df kicked \"{label}\" out because too many missing values "
                          f"({rel_missing} > {Patient.NAN_DISMISSAL_THRESHOLD / 1.5})")
                    avg_df.drop(label, inplace=True)

                else:  # try filling with mean imputation
                    for patient_id in avg_dict.keys():
                        if avg_df.isna()[patient_id][label]:
                            avg_df[patient_id][label] = label_avgs[label]
            if not use_interpolation:
                self.average_df_fixed_no_interpol = avg_df
            else:
                self.average_df_fixed_interpol = avg_df
            self.__save_data_to_cache()
            return avg_df

    def get_timeseries_df(self, use_interpolation: bool = False, fix_missing_values: bool = False,
                          limit_to_features: List[str] = None) -> pd.DataFrame:
        """
        Return a DataFrame with all the timeseries data per patient.

        The list limit_to_features is used to filter the dataframe for only specific features (e.g. only "HR").
        :param use_interpolation:
        :param fix_missing_values:
        :param limit_to_features:
        :return:
        """
        if use_interpolation:
            if self.__timeseries_interpol_no_fix is not None:
                return self.__timeseries_interpol_no_fix
        print(f"Creating timeseries DataFrame for TrainingSet {self.name} "+"with interpolation" if use_interpolation else "")
        time_series_dict = dict()

        start_time = datetime.datetime.now()
        print("Creating time_series_dict ...")
        max_series_len = -1
        for patient_id in self.data.keys():
            if use_interpolation:
                time_series_dict[patient_id] = pd.DataFrame(self.data[patient_id]
                                                            .get_interp_data(limit_to_features=limit_to_features))
            else:
                time_series_dict[patient_id] = pd.DataFrame(self.data[patient_id].data)
            max_series_len = len(time_series_dict[patient_id]) if len(
                time_series_dict[patient_id]) > max_series_len else max_series_len
        end_time = datetime.datetime.now()
        print("Finished creating raw time_series_dict in", end_time-start_time)

        max_timeseries_length = 40
        if fix_missing_values:
            max_series_len = max_timeseries_length
            for patient_id in time_series_dict.keys():
                time_series_dict[patient_id] = \
                    time_series_dict[patient_id].loc[1:max_timeseries_length]


        # pad timeseries lengths to biggest denominator
        start_time = datetime.datetime.now()
        print("Padding timeseries to biggest denominator ...")
        values = []
        for patient_id in time_series_dict.keys():
            for label in list(time_series_dict[list(time_series_dict.keys())[0]].columns):
                curr_len = max_series_len - len(time_series_dict[patient_id][label])
                padded_time_series = time_series_dict[patient_id][label] \
                    .append(pd.Series([np.NAN for i in range(curr_len)]), ignore_index=True)
                if use_interpolation:
                    padded_time_series.interpolate(method="linear", inplace=True)
                    padded_time_series.bfill(inplace=True)
                    padded_time_series.fillna(method='ffill', inplace=True)
                values.append(padded_time_series)
        end_time = datetime.datetime.now()
        print("Finished padding timeseries to biggest denominator in", end_time-start_time)

        print("Creating timeseries index ...")
        multi_index = pd.MultiIndex.from_product([time_series_dict.keys(),
                                                  list(time_series_dict[list(time_series_dict.keys())[0]].columns)])
        ts_df = pd.DataFrame(values, index=multi_index).transpose()

        print("Dropping unwanted features from dataframe ...")
        if limit_to_features is not None:  # might already be covered done by get_interp_data()
            for patient_id, label in multi_index:
                if label not in limit_to_features:
                    ts_df.drop((patient_id, label), axis=1, inplace=True)

        ts_df.drop(0, inplace=True)  # drop very first column (it keeps containing NaNs after interpolation)
        ts_df = ts_df.transpose()
        print("Dropping patients with all-NaN data:", ts_df[ts_df.isna().sum(axis=1) != 0].index)
        ts_df.drop(ts_df[ts_df.isna().sum(axis=1) != 0].index, inplace=True)

        if use_interpolation and not fix_missing_values:
            if self.__timeseries_interpol_no_fix is None:
                self.__timeseries_interpol_no_fix = dict()
            self.__timeseries_interpol_no_fix[tuple(limit_to_features)] = ts_df
        if use_interpolation and fix_missing_values:
            if self.__timeseries_interpol_fix is None:
                self.__timeseries_interpol_fix = dict()
            self.__timeseries_interpol_fix[tuple(limit_to_features)] = ts_df
        if not use_interpolation and not fix_missing_values:
            if self.__timeseries_no_interpol is None:
                self.__timeseries_no_interpol = dict()
            self.__timeseries_no_interpol[tuple(limit_to_features)] = ts_df
        if not use_interpolation and fix_missing_values:
            if self.__timeseries_no_interpol_fix is None:
                self.__timeseries_no_interpol_fix = dict()
            self.__timeseries_no_interpol_fix[tuple(limit_to_features)] = ts_df
        self.__save_data_to_cache()

        return ts_df

    def get_sepsis_label_df(self) -> pd.DataFrame:
        sepsis_column_dict = {}
        sepsis_value = {}
        for patient_id in self.data.keys():
            patient = self.data[patient_id]
            if patient.data["SepsisLabel"].sum() > 0:  # alternativ max()
                sepsis_value["SepsisLabel"] = 1
                sepsis_column_dict[patient_id] = pd.Series(sepsis_value) # needs to be a series to be in the same format like avg_df
            else:
                sepsis_value["SepsisLabel"] = 0
                sepsis_column_dict[patient_id] = pd.Series(sepsis_value)
        sepsis_df = pd.DataFrame(data=sepsis_column_dict, dtype='int32')
        return sepsis_df.transpose()          # Patients are rows, columns = SepsisLabel (no transpose needed)  - stimmt das??

    def check_patient_has_sepsis(self, sepsis_df, patient_id) -> bool:
        if sepsis_df.loc[patient_id, "SepsisLabel"] >= 1:
            return True
        else:
            return False

    # Vorschlag von Jakob: wird benötigt wenn man einen Cluster gezielt untersuchen will, leider habe ich es nicht ganz hinbekommen
    # def get_patients_for_clusters(self, clustering_list):
    #     clusters_with_patients: dict = {}
    #     for cluster in set(clustering_list):
    #         temp_index: List = []
    #         temp_patients: List = []
    #         for index, position in enumerate(clustering_list):
    #             if position == cluster:
    #                 temp_index.append(index)
    #         temp_patients = self.data.iloc(temp_index)              # Problem: iloc geht nur für df, wir haben hier leider dict, wie macht man das?
    #         clusters_with_patients[cluster] = temp_patients
    #     return clusters_with_patients

    def get_z_value_df(self, use_interpolation: bool = False, fix_missing_values: bool = False):
        """
        Used for DBScan calculation
        :param use_interpolation:
        :param fix_missing_values:
        :return:
        """
        avg_df = self.get_average_df(use_interpolation = use_interpolation, fix_missing_values = fix_missing_values)

        if self.z_value_df_no_interpol is not None and fix_missing_values and not use_interpolation:
            return self.z_value_df_no_interpol
        elif self.z_value_df is not None and fix_missing_values and use_interpolation:
            return self.z_value_df

        temp_z_val_df = pd.DataFrame()
        for col in avg_df.columns:
            # TODO: Frage an Philipp: Wie kann man das hier effizienter machen? Und bei no_fixed muss man noch NaN vom avg_df abfangen?
            # if avg_df[col] == "NaN":
            #     avg_df[col] = 0
            temp_z_val_df['z_' + col] = (avg_df[col] - avg_df[col].mean()) / avg_df[col].std()

        # TODO: Frage an Philipp: Was ist mit dem Fall fix_missing_values=True, use_interpolation=False ? (siehe TODO im Konstruktor, brauchen wir noch weitere avg cases?)
        if use_interpolation and fix_missing_values:
            self.z_value_df = temp_z_val_df
            return self.z_value_df
        else:
            self.z_value_df_no_interpol = temp_z_val_df
            return self.z_value_df_no_interpol

    def get_label_averages(self, label: str = None, use_interpolation: bool = False) -> pd.Series:
        """
        Calculate the average of a label across the whole set

        :param label: optional, if None all averages are calculated
        :param use_interpolation:
        :return: pd.Series of the averages calculated
        """
        label_sums = {key: 0 for key in Patient.LABELS}
        label_count = {key: 0 for key in Patient.LABELS}
        for patient_id in self.data.keys():
            if label is None:  # calculating for all labels
                for l in Patient.LABELS:
                    if not use_interpolation:
                        label_sums[l] += self.data[patient_id].data[l].dropna().sum()
                        label_count[l] += len(self.data[patient_id].data[l].dropna())
                    else:
                        label_sums[l] += self.data[patient_id].get_interp_data()[l].dropna().sum()
                        label_count[l] += len(self.data[patient_id].get_interp_data()[l].dropna())
            else:  # calculating only for given label
                if not use_interpolation:
                    label_sums[l] += self.data[patient_id].data[l].dropna().sum()
                    label_count[l] += len(self.data[patient_id].data[l].dropna())
                else:
                    label_sums[l] += self.data[patient_id].get_interp_data()[l].dropna().sum()
                    label_count[l] += len(self.data[patient_id].get_interp_data()[l].dropna())

        avgs = {}
        for l in label_sums.keys():
            if label_count[l] == 0:
                avgs[l] = None
            else:
                avgs[l] = label_sums[l] / label_count[l]

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
