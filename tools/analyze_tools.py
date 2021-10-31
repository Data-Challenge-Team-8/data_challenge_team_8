from operator import attrgetter

import pandas as pd
import pickle


class AnalyzeTool:
    """
    This Class contains methods to analyse the patient data.
    It provides tools for single patient analysis and for analysis over all patients.

    Important: the label is the feature key provided by the patient attribute
    """

    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "age", "gender", "unit1", "unit2", "hosp_adm_time", "ICULOS",
              "sepsis_label"]

    def __init__(self, training_data, save_data=True, create_copy_all=True):
        self.__training_data = training_data
        self.__save_data = save_data
        self.__create_copy_all = create_copy_all
        self.__time_series_data = self.time_series_data

    def print_data_set_analysis(self):
        """
        Prints a report over the whole dataset into stdout.

        Note: Might take a while.
        :return:
        """
        print("Average amount of timeseries in all data: ", self.avg_timeseries_all())
        print("Minimum amount of timeseries in all data: ", self.min_timeseries_all())
        print("Maximum amount of timeseries in all data: ", self.max_timeseries_all())
        print("Amount of timeseries in all data: ", self.count_timeseries_all())
        for label in self.LABELS:
            print()
            print(label)
            print("Minimum Value in all " + label + " colums: ", self.min_all(label))
            print("Maximum Value in all " + label + " colums: ", self.max_all(label))
            print("Average Value in all " + label + " colums: ", self.avg_all(label))
            print("Missing Values in all " + label + " colums: ", self.missing_values_all(label))

    @property
    def time_series_data(self) -> pd.DataFrame:
        out = []
        for patientID in self.__training_data:
            single_patient_data = len(getattr(self.__training_data[patientID], 'age'))
            out.append([patientID, single_patient_data])
        return out

    @staticmethod
    def __average(data) -> float:
        avg = sum(data) / len(data)
        return avg

    def min_all(self, label) -> [str, float]:
        """ finds the min value over a list with each value of every timeseries """
        min_patient = min([[patient[0], x]
                           for patient in self.__training_data.items()
                           for x in getattr(patient[1], label)
                           if pd.notna(x)],
                          key=lambda val: val[1],
                          default=False)
        if min_patient:
            min_val = min_patient
        else:
            print("-- All data was nan")
            min_val = "-- All data was nan"
        return min_val

    def max_all(self, label) -> [str, float]:
        """ finds the max value over a list with each value of every timeseries """
        max_patient = max([[patient[0], x]
                           for patient in self.__training_data.items()
                           for x in getattr(patient[1], label)
                           if pd.notna(x)],
                          key=lambda val: val[1],
                          default=False)
        if max_patient:
            max_val = max_patient
        else:
            print("-- All data was nan")
            max_val = "-- All data was nan"
        return max_val

    def avg_all(self, label):
        """ finds the average value over a list with each value of every timeseries """
        all_col = [x
                   for patient in self.__training_data.items()
                   for x in getattr(patient[1], label)
                   if pd.notna(x)]
        if len(all_col) == 0:
            avg_val = "-- all data is nan"
        else:
            avg_val = self.__average(all_col)
        return avg_val

    def subset_all(self, label, lowerbound, upperbound):
        """
        returns a subset of patients, witch contain a value between the lower and upper bound
        in one of there timeseries.
        """
        out = dict()
        for patientID in self.__training_data:
            single_patient_data = getattr(self.__training_data[patientID], label)
            for feature in single_patient_data:
                if pd.notna(feature):
                    if lowerbound <= feature <= upperbound:
                        out[patientID] = self.__training_data[patientID]
                        break
        return out

    def count_timeseries_all(self):
        """ counts all timeseries in the dataset (of each patient) """
        return sum([val[1] for val in self.__time_series_data if pd.notna(val[1])])

    def count_timeseries_single(self, patientID):
        """ counts all timeseries of one patient """
        return len(getattr(self.__training_data[patientID], 'age'))

    def avg_timeseries_all(self) -> float:
        """ returns the average amount of timeseries one patient has """
        return self.__average([val[1] for val in self.__time_series_data])

    def min_timeseries_all(self):
        """ returns the min amount of timeseries one patient has """
        return min([val[1] for val in self.__time_series_data if pd.notna(val[1])])

    def max_timeseries_all(self):
        """ returns the max amount of timeseries one patient has """
        return max([val[1] for val in self.__time_series_data if pd.notna(val[1])])

    def missing_values_all(self, label):
        """ returns the amount of missing values over a list with each value of every timeseries """
        nan_count = 0
        for patientID in self.__training_data:
            nan_count += getattr(self.__training_data[patientID], label).isna().sum()
        return nan_count

    def missing_values_all_avg(self, label):
        """ returns the average amount of missing values over a patients values """
        nan_count = 0
        c = 0
        for patientID in self.__training_data:
            nan_count += getattr(self.__training_data[patientID], label).isna().sum()
            c += 1
        return nan_count/c

    def min_single(self, label, patient_id):
        """ returns the min value in one patient data """
        patient_data = [x for x in getattr(self.__training_data[patient_id], label) if pd.notna(x)]
        return min(patient_data, default="-- all values are nan")

    def max_single(self, label, patient_id):
        """ returns the max value in one patient data """
        patient_data = [x for x in getattr(self.__training_data[patient_id], label) if pd.notna(x)]
        return max(patient_data)

    def avg_single(self, label, patient_id):
        """ returns the average value in one patient data """
        return self.__average([x for x in getattr(self.__training_data[patient_id], label) if pd.notna(x)])

    def missing_values_single(self, label, patient_id):
        """ returns the amount of missing value in one patient data """
        nan_count = getattr(self.__training_data[patient_id], label).isna().sum()
        return nan_count

    def save_data(self, data, fname):
        """ saves data to a pickel """
        pickle.dump(data, open(fname + ".p", "wb"))
