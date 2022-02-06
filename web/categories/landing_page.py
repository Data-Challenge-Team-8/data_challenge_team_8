import streamlit as st
import datetime
import pandas as pd
import numpy as np

from objects.patient import Patient
from objects.training_set import TrainingSet
from tools.analyse_tool import CompleteAnalysis as ca, CompleteAnalysis
from PIL import Image
from decimal import Decimal

def warning():
    color1='#E75919'
    color2='#EE895C'
    color3='#FFFFFF'
    text ='Before starting the analysis, we strongly recommend to load the desired dataset in advance. You can do this in the "Data Loader" tab.'
    st.markdown(
        f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:24px;border-radius:2%;">{text}</p>',
        unsafe_allow_html=True)

def display_feature_graphic(selected_column):
    feature_graphic = Image.open(r'./data/feature_graphic.jpg')
    selected_column.image(feature_graphic, caption='Descriptions for each feature from the underlying PhysioNet paper', width=800)


def display_table(selected_set_name: str, selected_column):
    info_p3 = 'The following table offers an overview of general descriptive statistics about the datasets:'
    selected_column.markdown(info_p3)
    temp_ca, cache_file_name = CompleteAnalysis.get_analysis(selected_label="fake_label", selected_tool='fake_tool',
                                                             selected_set=selected_set_name)

    general_info = {'Hospital System': ['Number of patients', 'Number of septic patients', 'Sepsis prevalence in %',
                                        'Number of entries', 'Number of NaNs', 'NaN prevalence in %',
                                        'Total hours recorded', 'Average hospital stay duration (hours)'],
                    selected_set_name: [(int(temp_ca.total_patients)),
                                        int(temp_ca.sepsis_patients_count),
                                        round((temp_ca.rel_sepsis_amount * 100), 2),
                                        int(temp_ca.data_amount),
                                        int(temp_ca.total_nan),
                                        round(temp_ca.rel_nan_total * 100, 2),
                                        int(temp_ca.total_time_measured),
                                        round(temp_ca.avg_data_duration_total, 2)]
                    }

    df_general_info = pd.DataFrame(general_info)

    # Lösung1
    df_general_info[selected_set_name] = df_general_info[selected_set_name].astype(str)
    df_general_info[selected_set_name] = df_general_info[selected_set_name].str.replace('.0', ' ', regex=False)

    df_general_info = df_general_info.style.format(na_rep='MISSING')
    selected_column.dataframe(df_general_info)

    # Lösung 2(funktioniert so nicht aber wäre schöner mit Trennzeichen implementierbar)
    # df_general_info[selected_set_name]=df_general_info[selected_set_name].astype(float).round(2)
    # df_general_info[selected_set_name]= [round(x) if x == round(x) else "{0:.2f}".format(x) for x in df_general_info[selected_set_name]]
    # col2.dataframe(df_general_info)

    # col2.table(general_info)           # sieht doof aus


def write_info_text(selected_column):  # maybe: if you have any questions or contact: our emails?
    info_p1 = "This dashboard was developed by students to display the data from the **PhysioNet Challenge 2019**. " \
              "The goal of this challenge was the early prediction of sepsis based on clinical data. " \
              "Data set A and B contain real patient data from two respective hospitals. " \
              "We hope the implementation of this dashboard will be helpful to discover " \
              "different patterns within the datasets."
    selected_column.markdown(info_p1)
    info_p2 = "For further information visit: https://physionet.org/content/challenge-2019/1.0.0/."
    selected_column.markdown(info_p2)


def write_info_text_2(selected_column):
    info_p4 = "Below the features collected in the dataset and their respective descriptions are displayed:"
    selected_column.markdown(info_p4)


def start_loading(selected_set_list, selected_label_list, selected_column):
    total_start_time = datetime.datetime.now()
    print("Loading started at time:", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))
    for unique_set in selected_set_list:
        if 'Load all labels' in selected_label_list:
            selected_label_list = selected_label_list.remove('Load all labels')
        for label in selected_label_list:
            start_time = datetime.datetime.now()
            ca.get_analysis(selected_label=label, selected_set=unique_set, selected_tool='none')
            difference_time = datetime.datetime.now() - start_time
            print("Loading of", unique_set, label, "took: ", str(difference_time).replace(" ", "_").replace(":", "-"))
    print("\nLoading finished at time:", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))
    total_difference_time = datetime.datetime.now() - total_start_time
    print("Complete loading took: ", str(total_difference_time).replace(" ", "_").replace(":", "-"))


class LandingPage:
    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
              "SepsisLabel"]

    def __init__(self):
        st.markdown("<h2 style='text-align: left; color: black;'>Project Description</h2>", unsafe_allow_html=True)
        col1, col2 = st.columns((2, 0.5))           # hiermit kann man "ränder" erstellen und test in columns machen
        write_info_text(col1)
        selected_set_name = self.create_selector(col2)
        display_table(selected_set_name, col2)

        warning()
        write_info_text_2(st)
        display_feature_graphic(st)

    def display_load_data_upfront(self, selected_column):
        multiselect_label_list = ['0_Load all labels (long waiting time!)']
        for label in Patient.LABELS:
            multiselect_label_list.append(label)
        # multiselect_label_list.sort()
        st.markdown("<h2 style='text-align: left; color: black;'>Recommended to load all data upfront:</h2>",
                    unsafe_allow_html=True)
        st.write("It can be useful to load the analysis data into a cache before first using this dashboard."
                 " Loading of a complete dataset alone can take up to 45 minutes (>1 minute per label)."
                 " Approximately 5MB are needed to safe the analysis of one label"
                 " (240MB for the complete dataset).")
        selected_set_list = st.multiselect(
            'Choose which set to load before moving to analysis. This can save loading time',
            TrainingSet.PRESETS, [])
        selected_label_list = st.multiselect(
            'Choose which labels to load before moving to analysis. This can save loading time',
            multiselect_label_list, [])
        warning()
        selected_set_name = self.create_selector(col1)
        display_table(selected_set_name, col1) #TODO: Reihenfolge anpassen

        if st.button('Load Data'):
            if selected_set_list and selected_label_list:
                start_loading(selected_set_list, selected_label_list, selected_column)

    def create_selector(self, selected_column):
        # selected_label = col1.selectbox('Choose a label:', self.LABELS)
        selected_set = selected_column.selectbox('Choose a Set for analysis:', ("Set A", "Set B", "Set A + B"))
        # selected_sepsis = col1.selectbox('Choose if sepsis positive or negative:',
        # ("positive + negative", "positive", "negative"))
        # selected_tool = selected_sepsis
        return selected_set
