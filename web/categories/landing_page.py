import streamlit as st
import datetime
from PIL import Image

from tools.analyse_tool import CompleteAnalysis


def display_feature_graphic():
    feature_graphic = Image.open(r'./data/feature_graphic.jpg')
    st.image(feature_graphic, caption='Descriptions for each feature from the underlying PhysioNet paper')


def display_table():  # TODO: Aufgabe Aline: calculate our own general info per dataset (amount of patients etc) - not just import of an image
    feature_graphic = Image.open(r'./data/descriptive_table.jpg')
    st.image(feature_graphic, caption='Descriptions for each feature from the underlying PhysioNet paper')


def write_info_text():  # maybe: if you have any questions or contact: our emails?
    st.markdown("<h2 style='text-align: left; color: black;'>Project Description</h2>", unsafe_allow_html=True)
    info_p1 = "This dashboard was developed by students to display the data from the **PhysioNet Challenge 2019**. " \
              "The goal of this challenge was the early prediction of sepsis based on clinical data. " \
              "Data set A and B contain real patient data from two respective hospitals. " \
              "We hope the implementation of this dashboard will be helpful to discover " \
              "different patterns within the datasets."
    st.markdown(info_p1)
    info_p2 = "For further information visit: https://physionet.org/content/challenge-2019/1.0.0/."
    st.markdown(info_p2)
    info_p3 = 'The following table offers an overview of general descriptive statistics about the datasets:'
    st.markdown(info_p3)
    display_table()
    info_p4 = "Below the features collected in the dataset and their respective descriptions are displayed:"
    st.markdown(info_p4)
    display_feature_graphic()


def start_loading(selected_set_list, selected_label_list):
    total_start_time = datetime.datetime.now()
    print("Loading started at time:", str(datetime.datetime.now()).replace(" ", "_").replace(":", "-"))
    for unique_set in selected_set_list:
        if 'Load all labels' in selected_label_list:
            selected_label_list = selected_label_list.remove('Load all labels')
        for label in selected_label_list:
            start_time = datetime.datetime.now()
            CompleteAnalysis.get_analysis(selected_label=label, selected_set=unique_set, selected_tool='none')
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
        self.load_data_upfront()
        write_info_text()

    def load_data_upfront(self):
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
