from typing import Dict
import numpy as np
import pandas as pd
import datetime
import os
import csv

from IO.data_writer import DataWriter
from objects.patient import Patient


# TODO: Naming too similar with complete_analysis - but this is needed for analysing more focussed with time_series_data
class AnalyseTool:
    """
    This Class contains methods to analyse the patient data.
    It provides tools for single patient analysis and for analysis over all patients.
    """

    def __init__(self, training_data: Dict[str, Patient], cal_write_data: bool):
        self.__training_data = training_data
        self.__time_series_data = None

        # if we want to save the calculated data to a pickle.
        if cal_write_data:
            # init the data writer class
            self.data_writer = DataWriter()
            min_all_labels = {key: None for key in Patient.LABELS}
            max_all_labels = {key: None for key in Patient.LABELS}
            avg_all_labels = {key: None for key in Patient.LABELS}
            missing_all_labels = {key: None for key in Patient.LABELS}
            for label in Patient.LABELS:
                min_all_labels[label] = self.min_all(label)
                max_all_labels[label] = self.max_all(label)
                avg_all_labels[label] = self.avg_all(label)
                missing_all_labels[label] = self.missing_values_all(label)
            # calling the methods to write the data to a pickle
            self.data_writer.write_min_max_avg(min_all_labels, max_all_labels, avg_all_labels)
            self.data_writer.write_missing_val(missing_all_labels)


    def do_whole_training_set_analysis(self, print_to_stdout: bool = False, export_to_csv: bool = False):
        """
        Makes a basic statistical analysis over the whole training set.
        Includes: minimum, maximum, average of every label. Global NaN count, global average, minimum and maximum
         of time series length and global total amount of measurement points.

        Prints it out to stdout if print_to_stdout is set.
        Exports the result to PSV

        Note: Might take a while.
        :param print_to_stdout:
        :param export_to_csv:
        :return:
        """
        print("Calculating whole training set analysis ...")
        avg_data_points = self.avg_timeseries_all()
        min_data_points = self.min_timeseries_all()
        max_data_points = self.max_timeseries_all()
        data_points_overall_count = self.count_timeseries_all()

        if print_to_stdout:
            print("Average amount of time series in all data: ", avg_data_points)
            print("Minimum amount of time series in all data: ", min_data_points)
            print("Maximum amount of time series in all data: ", max_data_points)
            print("Amount of time series in all data: ", data_points_overall_count)

        min_all_labels = {key: None for key in Patient.LABELS}
        max_all_labels = {key: None for key in Patient.LABELS}
        avg_all_labels = {key: None for key in Patient.LABELS}
        nans_all_labels = {key: None for key in Patient.LABELS}
        rel_nans_all_labels = {key: None for key in Patient.LABELS}
        for label in Patient.LABELS:
            r = self.min_all(label)
            if r is not None:
                min_all_labels[label] = r[1]
            else:
                min_all_labels[label] = r
            r = self.max_all(label)
            if r is not None:
                max_all_labels[label] = r[1]
            else:
                max_all_labels[label] = r

            avg_all_labels[label] = self.avg_all(label)
            nans_all_labels[label] = self.missing_values_all(label)
            rel_nans_all_labels[label] = self.relative_missing_values_all(label)

            if print_to_stdout:
                print()
                print("Minimum Value in all " + label + " columns: ", min_all_labels[label])
                print("Maximum Value in all " + label + " columns: ", max_all_labels[label])
                print("Average Value in all " + label + " columns: ", avg_all_labels[label])
                print("Missing Values in all " + label + " columns: ", nans_all_labels[label])

        print("Calculated whole training set analysis!")
        if export_to_csv:
            file_path = "analyze_tool_whole_training_set_analysis-" + str(datetime.datetime.now()).replace(" ",
                                                                                                           "_").replace(":", "-") + ".psv"
            print("Exporting results to:", os.path.join(".", file_path))
            with open(file_path, 'w') as file:
                w = csv.writer(file, delimiter="|")
                w.writerow(["label", "avg_data_points_overall", "min_data_points_overall", "max_data_points_overall",
                            "data_points_count_overall", "min", "max", "avg", "NaNs", "rel. NaNs"])
                for label in Patient.LABELS:
                    w.writerow([label, avg_data_points, min_data_points, max_data_points, data_points_overall_count,
                                min_all_labels[label], max_all_labels[label], avg_all_labels[label],
                                nans_all_labels[label], rel_nans_all_labels[label]])

    def do_basic_set_analysis(self, print_to_stdout: bool = False, export_to_csv: bool = False):
        """
        Calculate superficial statistical properties of the data set.

        Includes: Percentage of Patients with Sepsis, Chance of Sepsis per Patient, count and chance of NaN values,
        count of values overall
        :return:
        """
        sepsis_patients = []
        for patient_ID in self.__training_data.keys():
            if self.__training_data[patient_ID].data["SepsisLabel"].dropna().sum() > 0:  # at least one sepsis
                sepsis_patients.append(patient_ID)
        sepsis_chance = len(sepsis_patients) / len(self.__training_data.keys())

        row_count_all = 0
        entry_count_nans = 0
        entry_count_non_nans = 0
        for patient_ID in self.__training_data.keys():
            row_count_all += len(self.__training_data[patient_ID].data)
            for label in Patient.LABELS:
                entry_count_nans += self.__training_data[patient_ID].data[label].isna().sum()
                entry_count_non_nans += self.__training_data[patient_ID].data[label].dropna().count()
        entry_nan_chance = entry_count_nans / (entry_count_nans+entry_count_non_nans)

        if print_to_stdout:
            print("Patients with a Sepsis (Count):", len(sepsis_patients))
            print("Sepsis Chance per Patient:", sepsis_chance)
            print("Patients with a Sepsis:", sepsis_patients)
            print("Number of Rows (Total):", row_count_all)
            print("Number of NaNs (Total):", entry_count_nans)
            print("Number of non-NaNs (Total):", entry_count_non_nans)
            print("Chance of NaN per entry:", entry_nan_chance)

        if export_to_csv:
            file_path = "basic_analysis-" + str(datetime.datetime.now()).replace(" ", "_") + ".psv"
            print("Exporting results to:", os.path.join(".", file_path))
            with open(file_path, 'w') as file:
                w = csv.writer(file, delimiter="|")

                w.writerow(["patient_count", "sepsis_count", "sepsis_chance", "row_count_all", "entry_count_nans",
                            "entry_count_non_nans", "entry_nan_chance"])
                w.writerow([len(self.__training_data), len(sepsis_patients), sepsis_chance, row_count_all,
                            entry_count_nans, entry_count_non_nans, entry_nan_chance])

    @property
    def time_series_data(self):
        if self.__time_series_data is None:
            self.__time_series_data = []
            for patientID in self.__training_data:
                single_patient_data = len(getattr(self.__training_data[patientID], 'age'))
                self.__time_series_data.append([patientID, single_patient_data])

        return self.__time_series_data

    @staticmethod
    def __average(data) -> float:
        avg = sum(data) / len(data)
        return avg

    def relative_missing_values_all(self, label):
        """
        Calculates the average fraction of NaNs present per patient
        :param label:
        :return:
        """
        results = []
        for patient_id in self.__training_data.keys():
            results.append(self.relative_missing_values_single(label, patient_id))

        return self.__average(results)

    def average_standard_deviation_all(self, label):
        """
        Calculates the average standard deviation of a label over every timeseries
        :param label:
        :return:
        """
        s = 0
        n = 0
        for patient_ID in self.__training_data.keys():
            s += self.standard_deviation_single(label, patient_ID)
            n += 1

        return s/n

    def min_all(self, label) -> [str, float]:
        """
        Finds the min value over a list with each value of every timeseries
        :param label:
        :return: [PatientID, min_value]
        """
        min_patient = min([[patient[0], x]
                           for patient in self.__training_data.items()
                           for x in patient[1].data[label]
                           if pd.notna(x)],
                          key=lambda val: val[1],
                          default=False)
        if min_patient:
            min_val = min_patient
        else:
            min_val = None
        return min_val

    def max_all(self, label) -> [str, float]:
        """ finds the max value over a list with each value of every timeseries """
        max_patient = max([[patient[0], x]
                           for patient in self.__training_data.items()
                           for x in patient[1].data[label]
                           if pd.notna(x)],
                          key=lambda val: val[1],
                          default=False)
        if max_patient:
            max_val = max_patient
        else:
            max_val = None
        return max_val

    def avg_all(self, label) -> float:
        """ finds the average value over a list with each value of every timeseries """
        all_col = [x
                   for patient in self.__training_data.items()
                   for x in patient[1].data[label]
                   if pd.notna(x)]
        if len(all_col) == 0:
            avg_val = None
        else:
            avg_val = self.__average(all_col)
        return avg_val

    def subset_all(self, label, lowerbound, upperbound) -> []:
        """
        returns a subset of patients, witch contain a value between the lower and upper bound
        in one of there timeseries.
        """
        out = dict()
        for patientID in self.__training_data:
            single_patient_data = self.__training_data[patientID].data[label]
            for feature in single_patient_data:
                if pd.notna(feature):
                    if lowerbound <= feature <= upperbound:
                        out[patientID] = self.__training_data[patientID]
                        break
        return out

    def count_timeseries_all(self) -> int:
        """ counts all timeseries in the dataset (of each patient) """
        return sum([val[1] for val in self.time_series_data if pd.notna(val[1])])

    def count_timeseries_single(self, patientID) -> int:
        """ counts all timeseries of one patient """
        return len(getattr(self.__training_data[patientID], 'age'))

    def avg_timeseries_all(self) -> float:
        """ returns the average amount of timeseries one patient has """
        return self.__average([val[1] for val in self.time_series_data])

    def min_timeseries_all(self) -> float:
        """ returns the min amount of timeseries one patient has """
        return min([val[1] for val in self.time_series_data if pd.notna(val[1])])

    def max_timeseries_all(self) -> float:
        """ returns the max amount of timeseries one patient has """
        return max([val[1] for val in self.time_series_data if pd.notna(val[1])])

    def missing_values_all(self, label) -> int:
        """ returns the amount of missing values over a list with each value of every timeseries """
        nan_count = 0
        for patient_ID in self.__training_data:
            nan_count += self.__training_data[patient_ID].data[label].isna().sum()
        return nan_count

    def missing_values_all_avg(self, label) -> float:
        """ returns the average amount of missing values over a patients values """
        nan_count = 0
        c = 0
        for patient_ID in self.__training_data:
            nan_count += self.__training_data[patient_ID].data[label].isna().sum()
            c += 1
        return nan_count / c


    # TODO: nachfolgend bekomme ich KeyError 0.0 bei den x for x in ...
    # Ich weiss nicht was falsch ist. Aber so kann man nicht auf den dataframe zugreifen.
    def min_single(self, label, patient_ID) -> float:
        """ returns the min value in one patient data """
        patient_data = [x for x in self.__training_data[patient_ID].data[label] if pd.notna(x)]
        return min(patient_data, default="-- all values are nan")

    def max_single(self, label, patient_ID) -> float:
        """ returns the max value in one patient data """
        patient_data = [x for x in self.__training_data[patient_ID].data[label] if pd.notna(x)]
        return max(patient_data)

    def avg_single(self, label, patient_ID) -> float:
        """ returns the average value in one patient data """
        return self.__average([x for x in self.__training_data[patient_ID].data[label] if pd.notna(x)])

    def missing_values_single(self, label, patient_ID) -> int:
        """ returns the amount of missing value in one patient data """
        nan_count = self.__training_data[patient_ID].data[label].isna().sum()
        return nan_count

    def relative_missing_values_single(self, label, patient_ID) -> float:
        """
        Calculates the fraction of NaNs present for the given patient
        :param label:
        :param patient_ID
        :return:
        """
        patient = self.__training_data[patient_ID]
        nans = self.__training_data[patient_ID].data[label].isnull().sum()
        count = len(self.__training_data[patient_ID].data[label])

        return nans / count

    def standard_deviation_single(self, label, patient_ID) -> float:
        """
        Calculates the standard deviation of a single label for one patient
        :param label:
        :param patient_ID:
        :return:
        """
        a = self.__training_data[patient_ID].data[label].dropna().to_numpy()
        return np.std(a)

