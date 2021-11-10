from typing import Dict, Tuple, List
import os
import streamlit as st
from matplotlib import pyplot as plt
import hashlib
import pickle
import pandas as pd

from objects.training_set import TrainingSet
from web.UI_tools.analyse_tool import CompleteAnalysis

def create_description():
    info_p1 = "The focus of Explorative Data Analysis is the structuration of data of which " \
              "there is little knowledge. These tools enable you to filter each dataset for " \
              "the previously displayed features."
    st.markdown(info_p1)

    # "min_for_label": self.training_set.__min_for_label,
    # "max_for_label": self.training_set.__max_for_label,
    # "avg_for_label": self.training_set.__avg_for_label,
    # "NaN_amount_for_label": self.training_set.__NaN_amount_for_label,
    # "non_NaN_amount_for_label": self.training_set.__non_NaN_amount_for_label,
    # "min_data_duration": self.training_set.__min_data_duration,
    # "max_data_duration": self.training_set.__max_data_duration,
    # "avg_data_duration": self.training_set.__avg_data_duration,
    # "sepsis_patients": self.training_set.__sepsis_patients,
    # "plot_label_to_sepsis": self.training_set.__plot_label_to_sepsis


def plot_selected_analysis(analysis_dict, keys, col1, col2, col3):
    print("Plotting Selected Analysis:")
    if keys[1] == 'Min, Max, Average':
        max_label = analysis_dict['max_for_label']
        min_label = analysis_dict['min_for_label']
        avg_label = analysis_dict['avg_for_label']
        fig, ax1 = plt.subplots()
        ax1.bar(['max', 'min', 'average'], height=[float(max_label[1]), float(min_label[1]), avg_label], color="g")
        ax1.set_title('Min, Max and average of ' + keys[2])
        col2.pyplot(fig)
        col3.metric("Max of " + keys[2], max_label[1])
        col3.metric("Min of " + keys[2], min_label[1])
        col3.metric("Average of " + keys[2], round(avg_label, 2))
    elif keys[1] == 'Missing Values':
        missing_vals_rel = analysis_dict['NaN_amount_for_label']
        fig, ax = plt.subplots()
        ax.pie([missing_vals_rel, 1 - missing_vals_rel], explode=[0.2, 0], colors=['r', 'g'])
        col2.pyplot(fig)
        col3.metric("Missing (red)", str(round((missing_vals_rel * 100))) + "%")
        col3.metric("Not Missing (green)", str(round(((1 - missing_vals_rel) * 100))) + "%")
    else:
        st.empty()


class ExploratoryDataAnalysis:
    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
              "SepsisLabel"]

    def __init__(self, option: str):
        st.markdown("<h2 style='text-align: left; color: black;'>Exploratory Data Analysis</h2>",
                    unsafe_allow_html=True)
        create_description()

        col1, col2, col3 = st.columns((1, 2, 1))
        selected_label, selected_tool, selected_set = self.create_selector_tools(col1)
        keys = [selected_label, selected_tool, selected_set]
        self.current_training_set = TrainingSet.get_training_set(keys)

        print(self.current_training_set.test)
        print(self.current_training_set.__min_for_label)              # TODO: Doesnt find the attributes - probably should move them here anyways!

        # loads analysis from cache or creates new one
        analysis_dict = CompleteAnalysis.get_analysis_from_cache(keys, self.current_training_set)

        print("Analysis Object:", analysis_dict)
        print("Type:", type(analysis_dict))
        print("Elements:", analysis_dict[0], analysis_dict[1])
        plot_selected_analysis(analysis_dict, keys, col1, col2, col3)

    def create_selector_tools(self, col1):
        selected_set = col1.radio("Choose your data", ("Set A", "Set B", "Set A + B"))
        selected_tool = col1.selectbox('Choose a tool:', (
            'Min, Max, Average', 'Missing Values', 'Subgroups'))  # potentially also subgroups groups
        selected_label = col1.selectbox('Choose a label:', self.LABELS)

        return selected_set, selected_tool, selected_label
