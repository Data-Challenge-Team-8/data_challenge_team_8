import streamlit as st

from web.categories.landing_page import start_loading


class DataLoader:
    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
              "SepsisLabel"]



    def __init__(self):
        col1, col2, col3, col4 = st.columns((0.3, 2, 1.8, 0.3))
        self.display_load_data_upfront(col2)


    def display_load_data_upfront(self, col2):
        # multiselect_label_list = ['0_Load all labels (long waiting time!)']
        multiselect_label_list = []
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
                st.write("The selected set was successfully loaded.")
