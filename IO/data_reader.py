import os
from typing import Dict
import pandas as pd

from objects.patient import Patient, NotUniqueIDError


class UnknownDataset(Exception):
    pass


class DataReader:
    def __init__(self) -> None:
        self.file_dir_path_setA = r'./data/training_setA/'
        self.file_dir_path_setB = r'./data/training_setB/'

        self.__training_setA: Dict[str, Patient] = None
        self.__training_setB: Dict[str, Patient] = None
        self.__training_set_combined: Dict[str, Patient] = None

    def get_patient(self, patient_id: str) -> Patient:
        """
        Retrieve a single patient specified by the patient_ID (see file name) from either set.
        :param patient_id:
        :return:
        """
        if self.__training_setA is not None and patient_id in self.__training_setA.keys():
            # training set A is already loaded and has patient_ID
            return self.__training_setA[patient_id]
        elif self.__peek_patient_set(self.file_dir_path_setA, patient_id):  # patient is in set A but not loaded
            patient = self.__read_patient_data(self.file_dir_path_setA, patient_id + ".psv")

            if self.__training_setA is None:
                self.__training_setA = dict()
            self.__training_setA[patient_id] = patient

            return patient

        elif self.__training_setB is not None and patient_id in self.__training_setB.keys():
            # training set B is already loaded and has patient_ID
            return self.__training_setB[patient_id]
        elif self.__peek_patient_set(self.file_dir_path_setB, patient_id):  # patient is in set B but not loaded
            patient = self.__read_patient_data(self.file_dir_path_setA, patient_id + ".psv")

            if self.__training_setA is None:
                self.__training_setA = dict()
            self.__training_setA[patient_id] = patient

            return patient

        else:
            raise ValueError("Unknown Patient ID")

    def get_patient_setA(self, patient_id: str) -> Patient:
        """
        Retrieve a single patient specified by the patient_ID (see file name) from the set A

        Note: Assumes all file endings to be .psv (Pipe Separated Value)
        :param patient_id:
        :return:
        """
        if self.__training_setA is not None and patient_id in self.__training_setA.keys():
            # training set A is already loaded and has patient_ID
            return self.__training_setA[patient_id]

        else:  # not loaded, loading only the patient
            patient = self.__read_patient_data(self.file_dir_path_setA, patient_id+".psv")

            if self.__training_setA is None:
                self.__training_setA = dict()
            self.__training_setA[patient_id] = patient

            return patient

    def get_patient_setB(self, patient_id: str) -> Patient:
        """
        Retrieve a single patient specified by the patient_ID (see file name) from the set B

        Note: Assumes all file endings to be .psv (Pipe Separated Value)
        :param patient_id:
        :return:
        """
        if self.__training_setB is not None and patient_id in self.__training_setB.keys():
            # training set B is already loaded and has patient_ID
            return self.__training_setB[patient_id]

        else:
            patient = self.__read_patient_data(self.file_dir_path_setB, patient_id + ".psv")

            if self.__training_setB is None:
                self.__training_setB = dict()
            self.__training_setB[patient_id] = patient

            return patient


    def load_training_set(self, path) -> Dict[str, Patient]:

        if self.__training_setA is None or len(self.__training_setA.keys()) != len(os.listdir(self.file_dir_path_setA)):
            # either never loaded or not completely loaded -> need to read first
            self.__training_setA = self.__read_entire_training_set(self.file_dir_path_setA)

        return self.__training_setA

    @property
    def training_setA(self) -> Dict[str, Patient]:
        """
        Access the whole data set A
        :return:
        """
        if self.__training_setA is None or len(self.__training_setA.keys()) != len(os.listdir(self.file_dir_path_setA)):
            # either never loaded or not completely loaded -> need to read first
            self.__training_setA = self.__read_entire_training_set(self.file_dir_path_setA)

        return self.__training_setA

    @property
    def training_setB(self) -> Dict[str, Patient]:
        """
        Access the whole data set B
        :return: 
        """
        if self.__training_setB is None or len(self.__training_setA.keys()) != len(os.listdir(self.file_dir_path_setB)):
            # either never loaded or not completely loaded -> need to read first
            self.__training_setB = self.__read_entire_training_set(self.file_dir_path_setB)

        return self.__training_setB

    @property
    def combined_training_set(self) -> Dict[str, Patient]:
        """
        A combined version of both available training sets.
        :return:
        """
        combined_set = {}
        for patient in self.training_setA.values():
            combined_set[patient.ID] = patient

        for patient in self.training_setB.values():
            if patient.ID in combined_set.keys():
                raise ValueError("Combining the training sets impossible. Patients with same ID found!")
            combined_set[patient.ID] = patient
        return combined_set

    def __peek_patient_set(self, data_set_path: str, patient_id: str) -> bool:
        """
        Peek into the data set folder and check if the patient ID is present (but don't load it)

        Note: Assumes file extension to be .psv
        :param data_set_path:
        :param patient_id:
        :return:
        """
        filename = patient_id+".psv"
        if filename in os.listdir(data_set_path):
            return True
        else:
            return False

    def __read_patient_data(self, data_set_path: str, patient_filename: str) -> Patient:
        """
        Read a single patient file in the given data_set

        :param data_set_path: path to the data set folder
        :param patient_filename: file name of the patient file
        :return:
        """
        df_patient_data = pd.read_csv(os.path.join(data_set_path, patient_filename), sep='|')
        patient = Patient(os.path.splitext(patient_filename)[0], df_patient_data)

        return patient

    def __read_entire_training_set(self, data_set_path: str) -> Dict[str, Patient]:
        """
        Read all files within the given path and convert to a Patient object for each file.

        This may take some minutes.
        :param data_set_path: path to the data set (e.g. "./data/training_setA")
        :return: dictionary with patient_ID as key and Patient object as value
        """
        training_set_dict = dict()

        for filename in os.listdir(data_set_path):
            try:
                patient = self.__read_patient_data(data_set_path, filename)
            except NotUniqueIDError as e:  # caused ID uniqueness error, meaning we've already loaded this one!
                continue

            training_set_dict[os.path.splitext(filename)[0]] = patient
        return training_set_dict

    def load_new_training_set(self, file_dir_path):
        training_set_dict = dict()
        try:
            for filename in os.listdir(file_dir_path):
                try:
                    patient = self.__read_patient_data(file_dir_path, filename)
                except NotUniqueIDError as e:  # caused ID uniqueness error, meaning we've already loaded this one!
                    continue
                training_set_dict[os.path.splitext(filename)[0]] = patient
        except FileNotFoundError as e:
            print("Error:", e, "Please enter a valid dataset folder.")

        return training_set_dict
