import streamlit as st
from matplotlib import pyplot as plt

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


def plot_selected_analysis(analysis_dict, selected_label, selected_tool, selected_set, col1, col2, col3):
    print("Plotting Selected Analysis:")
    if selected_tool == 'Min, Max, Average':
        max_dict = analysis_dict['max_for_label']
        min_dict = analysis_dict['min_for_label']
        avg_dict = analysis_dict['avg_for_label']
        max_value = max_dict[selected_label][1]
        min_value = min_dict[selected_label][1]
        avg_value = avg_dict[selected_label]
        fig, ax1 = plt.subplots()
        ax1.bar(['max', 'min', 'average'], height=[float(max_value), float(min_value), avg_value], color="g")
        ax1.set_title('Min, Max and average of ' + selected_set)
        col2.pyplot(fig)
        col3.metric("Max of " + selected_set, max_value)
        col3.metric("Min of " + selected_set, min_value)
        col3.metric("Average of " + selected_set, round(avg_value, 2))
    elif selected_tool == 'Missing Values':
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
        # loads analysis from cache or creates new one
        analysis_dict, file_name = CompleteAnalysis.get_analysis(selected_label, selected_tool, selected_set)           # not good to use keys[] because list changes

        print("Type:", type(analysis_dict))
        print("Len: ", len(analysis_dict))

        plot_selected_analysis(analysis_dict, selected_label, selected_tool, selected_set, col1, col2, col3)

    def create_selector_tools(self, col1):
        selected_set = col1.radio("Choose your data", ("Set A", "Set B", "Set A + B"))
        selected_tool = col1.selectbox('Choose a tool:', (
            'Min, Max, Average', 'Missing Values', 'Subgroups'))  # potentially also subgroups groups
        selected_label = col1.selectbox('Choose a label:', self.LABELS)

        return selected_set, selected_tool, selected_label
