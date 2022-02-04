import streamlit as st
from matplotlib import pyplot as plt

from objects.training_set import TrainingSet
from tools.analyse_tool import CompleteAnalysis
from objects.patient import Patient


class ExploratoryDataAnalysis:

    TOOL_SELECTION = {"avg": "Min, Max, Average, Variance",
                      "missing": "Missing Values",
                      #"subgroup": "Subgroups",
                      }

    def __init__(self):
        st.markdown("## Exploratory Data Analysis")
        self.__create_description()

        col1, col2, col3 = st.columns((1, 2, 1))
        selected_label, selected_tool, selected_set = self.create_selector_tools(col1)
        # loads analysis from cache or creates new one
        analysis_obj, file_name = CompleteAnalysis.get_analysis(selected_label=selected_label,
                                                                selected_tool=selected_tool,
                                                                selected_set=selected_set)
        self.plot_selected_analysis(analysis_obj, selected_label, selected_tool, selected_set, col2, col3)

    def create_selector_tools(self, col1):
        selected_set = col1.radio("Choose your data", TrainingSet.PRESETS)
        selected_tool = col1.selectbox('Choose a tool:', tuple(ExploratoryDataAnalysis.TOOL_SELECTION.values()))
        # potentially also subgroups groups
        selected_label = col1.selectbox('Choose a label:', Patient.LABELS)

        return selected_label, selected_tool, selected_set

    @staticmethod
    def __create_description():
        info_p1 = "The goal of Explorative Data Analysis is to gain an overview of data for which " \
                  "there is little previous knowledge. These tool enables you to filter through the labels " \
                  "in each dataset." \
                  " If the dataset has not been loaded previously, the background analysis might take up to 3 minutes."
        st.markdown(info_p1)

    @staticmethod
    def plot_selected_analysis(analysis_obj: CompleteAnalysis, selected_label, selected_tool, selected_set, col2, col3):
        if selected_tool == ExploratoryDataAnalysis.TOOL_SELECTION["avg"]:
            min_value = analysis_obj.min_for_label[selected_label][1]
            max_value = analysis_obj.max_for_label[selected_label][1]
            avg_value = analysis_obj.avg_for_label[selected_label]
            variance = analysis_obj.variance_for_label[selected_label]
            fig, ax1 = plt.subplots()
            ax1.bar(['max', 'min', 'average', 'variance'],
                    height=[float(max_value), float(min_value), avg_value, variance], color="g")
            ax1.set_title(f'Min, max, average and variance of {selected_set}, {selected_label}')
            col2.pyplot(fig)
            col3.metric("Max of " + selected_set, max_value)
            col3.metric("Min of " + selected_set, min_value)
            col3.metric("Average of " + selected_set, round(avg_value, 2))
            col3.metric(f"Variance of {selected_set}", round(variance, 2))
        elif selected_tool == ExploratoryDataAnalysis.TOOL_SELECTION["missing"]:
            missing_vals_rel = analysis_obj.rel_NaN_for_label
            fig, ax = plt.subplots()
            ax.pie([missing_vals_rel, 1 - missing_vals_rel], explode=[0.2, 0], colors=['r', 'g'])
            col2.pyplot(fig, transparent=True)  # transparent looks much better imo
            col3.metric("Missing (red)", str(round((missing_vals_rel * 100))) + "%")
            col3.metric("Not Missing (green)", str(round(((1 - missing_vals_rel) * 100))) + "%")
        else:
            st.write("Feature not implemented yet.")
