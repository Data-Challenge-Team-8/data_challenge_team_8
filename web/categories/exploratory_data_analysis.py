import matplotlib.pyplot as plt
import streamlit as st
import numpy as np
import pandas as pd

from IO.data_reader import DataReader
from objects.training_set import TrainingSet


class ExploratoryDataAnalysis:
    DESCRIPTION = 'Die explorative Datenanalyse (EDA) oder explorative Statistik ist ein ' \
                  'Teilgebiet der Statistik. Sie untersucht und begutachtet Daten,' \
                  ' von denen nur ein geringes Wissen über deren Zusammenhänge vorliegt.'

    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "age", "gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
              "SepsisLabel"]

    def __init__(self):

        self.create_description()

        self.__training_set = {}

        tool_selected, set_selected, label_selected = self.create_selector_tools()
        if tool_selected == 'Min, Max, Average':
            self.create_min_max_avg(tool_selected, set_selected, label_selected)
        if tool_selected == 'Missing Values':
            self.create_missing_vals(tool_selected, label_selected, set_selected)
        if tool_selected == 'Subgroups Groups':
            pass

    def create_description(self):
        st.write(self.DESCRIPTION)

    def create_selector_tools(self):
        col1, col2, col3 = st.columns(3)
        selected_tool = col1.selectbox(
            'Choose a tool:',
            (
                'Min, Max, Average',
                'Missing Values',
                'Subgroups Groups'
            )
        )

        selected_label = col2.selectbox(
            'Choose a label:',
            (
                self.LABELS
            )
        )

        selected_set = col3.radio(
            "Choose your data",
            (
                "Set A",
                "Set B",
                "Set A + B"
            )
        )
        return selected_tool, selected_set, selected_label

    def create_missing_vals(self, selected_tool, selected_label, selected_set):
        if TrainingSet(
                "exploratory_data_analysis_missing_values",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
        ).is_cached():
            missing_vals_rel = TrainingSet(
                "exploratory_data_analysis_missing_values",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
            ).get_avg_rel_NaN_amount_for_label(selected_label)
        else:
            dr = DataReader()
            if selected_set == "Set A":
                self.__training_set = dr.training_setA
            elif selected_set == "Set B":
                self.__training_set = dr.training_setB
            else:
                self.__training_set = dr.combined_training_set

            missing_vals_rel = TrainingSet(
                "exploratory_data_analysis_missing_values",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
            ).get_avg_rel_NaN_amount_for_label(selected_label)

        fig, ax = plt.subplots()
        mylabels = ["Amount of NaN Values", "Amount of defined data"]
        ax.pie([missing_vals_rel, 1 - missing_vals_rel], labels=mylabels)
        st.pyplot(fig)

    def create_min_max_avg(self, selected_tool, selected_set, selected_label):
        if TrainingSet(
                "exploratory_data_analysis_min_max_avg",
                self.__training_set,
                [selected_tool,
                 selected_label,
                 selected_set]
        ).is_cached():

            max_label = TrainingSet(
                "exploratory_data_analysis_min_max_avg",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
            ).get_max_for_label(selected_label)

            min_label = TrainingSet(
                "exploratory_data_analysis_min_max_avg",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
            ).get_min_for_label(selected_label)

            avg_label = TrainingSet(
                "exploratory_data_analysis_min_max_avg",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
            ).get_avg_for_label(selected_label)
        else:
            dr = DataReader()
            if selected_set == "Set A":
                self.__training_set = dr.training_setA
            elif selected_set == "Set B":
                self.__training_set = dr.training_setB
            else:
                self.__training_set = dr.combined_training_set

            max_label = TrainingSet(
                "exploratory_data_analysis_min_max_avg",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
            ).get_max_for_label(selected_label)

            min_label = TrainingSet(
                "exploratory_data_analysis_min_max_avg",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
            ).get_min_for_label(selected_label)

            avg_label = TrainingSet(
                "exploratory_data_analysis_min_max_avg",
                self.__training_set,
                [selected_tool, selected_label, selected_set]
            ).get_avg_for_label(selected_label)

        col1, col2, col3 = st.columns(3)
        col1.metric("Max of " + selected_label, max_label[1])
        col2.metric("Min of " + selected_label, min_label[1])
        col3.metric("Average of " + selected_label, round(float(avg_label), 2))
