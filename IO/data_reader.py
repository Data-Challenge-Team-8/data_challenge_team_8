from typing import Dict
import os

import pandas as pd

from objects.patient import Patient, NotUniqueIDError


class DataReader:

    def __init__(self) -> None:
        self.file_dir_path_setA = r'./data/training_setA/'
        self.file_dir_path_setB = r'./data/training_setB/'

        self.__training_setA: Dict[str, Patient] = None
        self.__training_setB: Dict[str, Patient] = None

    def get_patient_setA(self, patient_id: str) -> Patient:
        """
        Retrieve a single patient specified by the patient_id (see file name) from the set A

        Note: Assumes all file endings to be .psv (Pipe Separated Value)
        :param patient_id:
        :return:
        """
        if self.__training_setA is not None and patient_id in self.__training_setA.keys():
            # training set A is already loaded and has patient_id
            return self.__training_setA[patient_id]

        else:  # not loaded, loading only the patient
            patient = self.__read_patient_data(self.file_dir_path_setA, patient_id+".psv")

            if self.__training_setA is None:
                self.__training_setA = dict()
            self.__training_setA[patient_id] = patient

            return patient

    def get_patient_setB(self, patient_id: str) -> Patient:
        """
        Retrieve a single patient specified by the patient_id (see file name) from the set B

        Note: Assumes all file endings to be .psv (Pipe Separated Value)
        :param patient_id:
        :return:
        """
        if self.__training_setB is not None and patient_id in self.__training_setB.keys():
            # training set B is already loaded and has patient_id
            return self.__training_setB[patient_id]

        else:
            patient = self.__read_patient_data(self.file_dir_path_setB, patient_id + ".psv")

            if self.__training_setB is None:
                self.__training_setB = dict()
            self.__training_setB[patient_id] = patient

            return patient



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

    def __read_entire_training_set(self, path: str) -> Dict[str, Patient]:
        """
        Read all files within the given path and convert to a Patient object for each file.

        This may take some minutes.
        :param path: path to the data set (e.g. "./data/training_setA")
        :return: dictionary with patient_id as key and Patient object as value
        """
        training_set = dict()

        for filename in os.listdir(path):
            try:
                patient = self.__read_patient_data(path, filename)
            except NotUniqueIDError as e:  # caused ID uniqueness error, meaning we've already loaded this one!
                continue

            training_set[os.path.splitext(filename)[0]] = patient
        return training_set
