import streamlit as st
from matplotlib import pyplot as plt

from web.UI_tools.analyse_tool import CompleteAnalysis


def create_description():
    info_p1 = "The goal of Explorative Data Analysis is to gain an overview of data for which " \
              "there is little previous knowledge. These tool enables you to filter through the labels " \
              "in each dataset." \
              " If the dataset has not been loaded previously, the background analysis might take up to 3 minutes."
    st.markdown(info_p1)


def plot_selected_analysis(analysis_obj, selected_label, selected_tool, selected_set, col2, col3):
    if selected_tool == 'Min, Max, Average':
        min_value = analysis_obj.min_for_label[selected_label][1]
        max_value = analysis_obj.max_for_label[selected_label][1]
        avg_value = analysis_obj.avg_for_label[selected_label]
        fig, ax1 = plt.subplots()
        ax1.bar(['max', 'min', 'average'], height=[float(max_value), float(min_value), avg_value], color="g")
        ax1.set_title('Min, Max and average of ' + selected_set)
        col2.pyplot(fig)
        col3.metric("Max of " + selected_set, max_value)
        col3.metric("Min of " + selected_set, min_value)
        col3.metric("Average of " + selected_set, round(avg_value, 2))
    elif selected_tool == 'Missing Values':
        missing_vals_rel = analysis_obj.rel_NaN_for_label
        fig, ax = plt.subplots()
        ax.pie([missing_vals_rel, 1 - missing_vals_rel], explode=[0.2, 0], colors=['r', 'g'])
        col2.pyplot(fig)
        col3.metric("Missing (red)", str(round((missing_vals_rel * 100))) + "%")
        col3.metric("Not Missing (green)", str(round(((1 - missing_vals_rel) * 100))) + "%")
    else:
        st.write("Feature not implemented yet.")


class ExploratoryDataAnalysis:
    LABELS = ["HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
              "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
              "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
              "PTT", "WBC", "Fibrinogen", "Platelets", "Age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
              "SepsisLabel"]

    def __init__(self):
        st.markdown("<h2 style='text-align: left; color: black;'>Exploratory Data Analysis</h2>",
                    unsafe_allow_html=True)
        create_description()

        col1, col2, col3 = st.columns((1, 2, 1))
        selected_label, selected_tool, selected_set = self.create_selector_tools(col1)
        # loads analysis from cache or creates new one
        analysis_obj, file_name = CompleteAnalysis.get_analysis(selected_label=selected_label,
                                                                 selected_tool=selected_tool,
                                                                 selected_set=selected_set)           # not good to use keys[] because list changes
        plot_selected_analysis(analysis_obj, selected_label, selected_tool, selected_set, col2, col3)

    def create_selector_tools(self, col1):
        selected_set = col1.radio("Choose your data", ("Set A", "Set B", "Set A + B"))
        selected_tool = col1.selectbox('Choose a tool:', (
            'Min, Max, Average', 'Missing Values', 'Subgroups'))  # potentially also subgroups groups
        selected_label = col1.selectbox('Choose a label:', self.LABELS)

        return selected_label, selected_tool, selected_set
