import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

from IO.data_reader import DataReader
from objects.training_set import TrainingSet
from web.UI_tools.min_max_avg import MinMaxAvg
from web.UI_tools.missing_values import MissingValues


class ExploratoryDataAnalysis:
    DESCRIPTION = 'Die explorative Datenanalyse (EDA) oder explorative Statistik ist ein ' \
                  'Teilgebiet der Statistik. Sie untersucht und begutachtet Daten,' \
                  ' von denen nur ein geringes Wissen über deren Zusammenhänge vorliegt.'

    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
              "SepsisLabel"]

    def __init__(self, display: bool = True):

        self.__training_set = {}

        if display:
            self.create_description()
            col1, col2, col3 = st.columns((1, 2, 1))
            tool_selected, set_selected, label_selected = self.create_selector_tools(col1)
            if tool_selected == 'Min, Max, Average':
                MinMaxAvg(tool_selected, label_selected, set_selected, col3, col2)
            if tool_selected == 'Missing Values':
                MissingValues(tool_selected, label_selected, set_selected, col3, col2)
            if tool_selected == 'Subgroups Groups':
                pass

    def create_description(self):
        st.write(self.DESCRIPTION)

    def create_selector_tools(self, col1):
        selected_tool = col1.selectbox(
            'Choose a tool:',
            (
                'Min, Max, Average',
                'Missing Values',
                'Subgroups Groups'
            )
        )

        selected_label = col1.selectbox(
            'Choose a label:',
            (
                self.LABELS
            )
        )

        selected_set = col1.radio(
            "Choose your data",
            (
                "Set A",
                "Set B",
                "Set A + B"
            )
        )
        return selected_tool, selected_set, selected_label