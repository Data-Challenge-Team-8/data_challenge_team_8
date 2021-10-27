import pandas as pd
import os
import sys

from DataLoader.Patient import Patient


class DataRead:

    def __init__(self) -> None:
        self.file_dir_path_setA = r'./data/training_setA/'
        self.file_dir_path_setB = r'./data/training_setB/'
        self.trainingSetA = []
        self.trainingSetB = []
        self.readTrainingSet()

    def readTrainingSet(self) -> pd.DataFrame:
        for filename in os.listdir(self.file_dir_path_setA):
            filePath = self.file_dir_path_setA + filename
            df_setA = pd.read_csv(filePath, sep='|')
            patient = Patient(df_setA['HR'].tolist(),
                              df_setA['O2Sat'].tolist(),
                              df_setA['Temp'].tolist(),
                              df_setA['SBP'].tolist(),
                              df_setA['MAP'].tolist(),
                              df_setA['DBP'].tolist(),
                              df_setA['Resp'].tolist(),
                              df_setA['EtCO2'].tolist(),
                              df_setA['BaseExcess'].tolist(),
                              df_setA['HCO3'].tolist(),
                              df_setA['FiO2'].tolist(),
                              df_setA['pH'].tolist(),
                              df_setA['PaCO2'].tolist(),
                              df_setA['SaO2'].tolist(),
                              df_setA['AST'].tolist(),
                              df_setA['BUN'].tolist(),
                              df_setA['Alkalinephos'].tolist(),
                              df_setA['Calcium'].tolist(),
                              df_setA['Chloride'].tolist(),
                              df_setA['Creatinine'].tolist(),
                              df_setA['Bilirubin_direct'].tolist(),
                              df_setA['Glucose'].tolist(),
                              df_setA['Lactate'].tolist(),
                              df_setA['Magnesium'].tolist(),
                              df_setA['Phosphate'].tolist(),
                              df_setA['Potassium'].tolist(),
                              df_setA['Bilirubin_total'].tolist(),
                              df_setA['TroponinI'].tolist(),
                              df_setA['Hct'].tolist(),
                              df_setA['Hgb'].tolist(),
                              df_setA['PTT'].tolist(),
                              df_setA['WBC'].tolist(),
                              df_setA['Fibrinogen'].tolist(),
                              df_setA['Platelets'].tolist(),
                              df_setA['Age'].tolist(),
                              df_setA['Gender'].tolist(),
                              df_setA['Unit1'].tolist(),
                              df_setA['Unit2'].tolist(),
                              df_setA['HospAdmTime'].tolist(),
                              df_setA['ICULOS'].tolist(),
                              df_setA['SepsisLabel'].tolist())
            self.trainingSetA.append(patient)
        return self.trainingSetA
