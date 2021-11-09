import streamlit as st

from web.UI_tools.min_max_avg import MinMaxAvg
from web.UI_tools.missing_values import MissingValues


def create_description():
    info_p1 = "The focus of Explorative Data Analysis is the structuration of data of which " \
              "there is little knowledge. These tools enable you to filter each dataset for " \
              "the previously displayed features."
    st.markdown(info_p1)


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
        tool_selected, set_selected, label_selected = self.create_selector_tools(col1)
        if tool_selected == 'Min, Max, Average':
            MinMaxAvg(tool_selected, label_selected, set_selected, col3, col2)
        if tool_selected == 'Missing Values':
            MissingValues(tool_selected, label_selected, set_selected, col3, col2)
        if tool_selected == 'Subgroups Groups':
            pass

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
