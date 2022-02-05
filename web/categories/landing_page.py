import streamlit as st
import datetime
import pandas as pd
import numpy as np

from objects.training_set import TrainingSet
from tools.analyse_tool import CompleteAnalysis as ca, CompleteAnalysis
from PIL import Image
from decimal import Decimal


def display_feature_graphic(col1):
    feature_graphic = Image.open(r'./data/feature_graphic.jpg')
    st.image(feature_graphic, caption='Descriptions for each feature from the underlying PhysioNet paper')


def display_table(selected_set_name: str, col2):
    info_p3 = 'The following table offers an overview of general descriptive statistics about the datasets:'
    st.markdown(info_p3)
    temp_ca, cache_file_name = CompleteAnalysis.get_analysis(selected_label="fake_label", selected_tool='fake_tool',
                                                             selected_set=selected_set_name)

    general_info = {'Hospital System': ['Number of patients', 'Number of septic patients', 'Sepsis prevalence in %',
                                        'Number of entries', 'Number of NaNs', 'NaN prevalence in %',
                                        'Total hours recorded', 'Average hospital stay duration (hours)'],
                     selected_set_name: [(int(temp_ca.total_patients)),
                                         int(temp_ca.sepsis_patients_count),
                                         round((temp_ca.rel_sepsis_amount*100),2),
                                         int(temp_ca.data_amount),
                                         int(temp_ca.total_nan),
                                         round((temp_ca.rel_nan_total)*100,2),
                                         int(temp_ca.total_time_measured),
                                         round(temp_ca.avg_data_duration_total,2)]
                     }

    df_general_info = pd.DataFrame(general_info)

    #Lösung1
    df_general_info[selected_set_name] = df_general_info[selected_set_name].astype(str)
    df_general_info[selected_set_name] = df_general_info[selected_set_name].str.replace('.0', ' ', regex = False)

    df_general_info = df_general_info.style.format(na_rep='MISSING')
    col2.dataframe(df_general_info)

    #Lösung 2(funktioniert so nicht aber wäre schöner mit Trennzeichen implementierbar)
    #df_general_info[selected_set_name]=df_general_info[selected_set_name].astype(float).round(2)
    #df_general_info[selected_set_name]= [round(x) if x == round(x) else "{0:.2f}".format(x) for x in df_general_info[selected_set_name]]
    #col2.dataframe(df_general_info)

    #col2.table(general_info)           # sieht doof aus


def write_info_text(col1):  # maybe: if you have any questions or contact: our emails?
    info_p1 = "This dashboard was developed by students to display the data from the **PhysioNet Challenge 2019**. " \
              "The goal of this challenge was the early prediction of sepsis based on clinical data. " \
              "Data set A and B contain real patient data from two respective hospitals. " \
              "We hope the implementation of this dashboard will be helpful to discover " \
              "different patterns within the datasets."
    col1.markdown(info_p1)
    info_p2 = "For further information visit: https://physionet.org/content/challenge-2019/1.0.0/."
    col1.markdown(info_p2)


def write_info_text_2(col1):
    info_p4 = "Below the features collected in the dataset and their respective descriptions are displayed:"

def start_loading(selected_set_list, selected_label_list, col1):
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
        col1, col2, col3, col4 = st.columns((0.3, 2, 1.8, 0.3))           # hiermit kann man "ränder" erstellen und test in columns machen
        col2.markdown("<h2 style='text-align: left; color: black;'>Project Description</h2>", unsafe_allow_html=True)
        write_info_text(col2)
        selected_set_name = self.create_selector(col3)
        display_table(selected_set_name, col3)

        write_info_text_2(col2)                 # todo: vlt sollten die ganzen nachfolgenden Elemente auf der Seite auch innerhalb von columns angeordnet sein. Sieht besser aus
        display_feature_graphic(col2)
        self.display_load_data_upfront(col2)        # todo: vlt sollte dieses "Cache laden" ganz oben sein, und auch anders heißen?

    def display_load_data_upfront(self,col2):
        multiselect_label_list = ['0_Load all labels (long waiting time!)']
        for label in self.LABELS:
            multiselect_label_list.append(label)
        # multiselect_label_list.sort()
        st.markdown("<h2 style='text-align: left; color: black;'>Recommended to load the data upfront:</h2>",
                    unsafe_allow_html=True)
        st.write("It can be useful to load the analysis data into a cache before first using this dashboard."
                 " Loading of a complete dataset alone can take up to 45 minutes (>1 minute per label)."
                 " Approximately 5MB are needed to safe the analysis of one label"
                 " (240MB for the complete dataset).")
        selected_set_list = st.multiselect(
            'Choose which set to load before moving to analysis. This can save loading time',
            ['Set A', 'Set B', 'Set A + B'], [])
        selected_label_list = st.multiselect(
            'Choose which labels to load before moving to analysis. This can save loading time',
            multiselect_label_list, [])

        if st.button('Load Data'):
            if selected_set_list and selected_label_list:
                start_loading(selected_set_list, selected_label_list)

    def create_selector(self, col2):
        # selected_label = col1.selectbox('Choose a label:', self.LABELS)
        selected_set = col2.selectbox('Choose a Set for analysis:', ("Set A", "Set B", "Set A + B"))
        # selected_sepsis = col1.selectbox('Choose if sepsis positive or negative:',
        # ("positive + negative", "positive", "negative"))
        # selected_tool = selected_sepsis
        return selected_set
