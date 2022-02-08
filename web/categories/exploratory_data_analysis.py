import streamlit as st
from matplotlib import pyplot as plt
import pandas as pd

from objects.training_set import TrainingSet
from tools.analyse_tool import CompleteAnalysis
from objects.patient import Patient

def warning():
    color1='#E75919'
    color2='#EE895C'
    color3='#FFFFFF'
    text ='Before starting the analysis, we strongly recommend to load the desired dataset in advance. You can do this in the "Data Loader" tab.'
    st.markdown(
        f'<p style="text-align:center;background-image: linear-gradient(to right,{color1}, {color2});color:{color3};font-size:24px;border-radius:2%;">{text}</p>',
        unsafe_allow_html=True)

class ExploratoryDataAnalysis:
    TOOL_SELECTION = {"avg": "Min, Max, Average, Variance",
                      "missing": "Missing Values",
                      "all_distribution": "Distribution",
                      "avg_distribution": "Distribution (Avg)",
                      "dev_distribution": "Distribution (deviation)",
                      # "subgroup": "Subgroups",
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
        warning()

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

    def plot_selected_analysis(self, analysis_obj: CompleteAnalysis, selected_label, selected_tool, selected_set, col2,
                               col3):
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
        elif selected_tool == ExploratoryDataAnalysis.TOOL_SELECTION["avg_distribution"]:
            fig = self.__plot_distribution(analysis_obj, selected_label, selected_set, method="avg")
            col2.pyplot(fig)
        elif selected_tool == ExploratoryDataAnalysis.TOOL_SELECTION["all_distribution"]:
            fig = self.__plot_distribution(analysis_obj, selected_label, selected_set, method="all")
            col2.pyplot(fig)
        elif selected_tool == ExploratoryDataAnalysis.TOOL_SELECTION["dev_distribution"]:
            fig = self.__plot_distribution(analysis_obj, selected_label, selected_set, method="dev")
            col2.pyplot(fig)
        else:
            st.write("Feature not implemented yet.")

    def __plot_distribution(self, analysis_obj: CompleteAnalysis, selected_label: str, selected_set: str,
                            method: str = "avg") -> plt.Figure:
        fig, ax = plt.subplots()

        data_series = pd.Series()
        if method == "avg":
            data_series = TrainingSet.get_training_set(selected_set).get_average_df().loc[selected_label]
        elif method == "all":
            for patient in TrainingSet.get_training_set(selected_set).data.values():
                data_series = data_series.append(patient.data[selected_label], ignore_index=True)
        elif method == "dev":
            for patient in TrainingSet.get_training_set(selected_set).data.values():
                data_series = data_series.append(pd.Series(patient.data[selected_label].var() ** .5,
                                                 index=[patient.ID]))
        min_value = min(data_series)
        max_value = max(data_series)
        avg_value = data_series.mean()
        variance = data_series.var()
        bins = int(max_value - min_value * 0.8)
        if method == "dev" or method == "avg":  # requires data_series to be "per-patient"
            sepsis_df = TrainingSet.get_training_set(selected_set).get_sepsis_label_df().astype(bool)
            sepsis_df.insert(1, selected_label, data_series, allow_duplicates=True)

            sepsis_avg = sepsis_df[sepsis_df.loc[:, "SepsisLabel"]].loc[:, selected_label].mean()
            sepsis_var = sepsis_df[sepsis_df.loc[:, "SepsisLabel"]].loc[:, selected_label].var()

        ax.hist(data_series.tolist(), bins=bins, label=selected_label)
        if method == "dev" or method == "avg":
            ax.hist(sepsis_df[sepsis_df.loc[:, "SepsisLabel"]].loc[:, selected_label], bins=bins, label="Sepsis")
            ax.vlines([sepsis_avg + sepsis_var**.5, sepsis_avg-sepsis_var**.5], ymin=ax.get_ylim()[0],
                      ymax=ax.get_ylim()[1],
                      label=f"std.deviation (Sepsis)", color="purple", alpha=0.5, linestyles='dashdot')
            ax.vlines([sepsis_avg], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label=f"average (Sepsis)",
                      color='cyan', alpha=0.8, linestyles='dashdot')

        ax.vlines([avg_value + variance ** .5, avg_value - variance ** .5], ymin=ax.get_ylim()[0],
                  ymax=ax.get_ylim()[1],
                  label=f"std. deviation ({selected_label})", color='r', alpha=0.5, linestyles="dashed")
        ax.vlines([avg_value], ymin=ax.get_ylim()[0], ymax=ax.get_ylim()[1], label=f"average ({selected_label})",
                  color='g', alpha=0.9, linestyles='dashed')
        if method == "avg":
            ax.set_title(f"Distribution of Average {selected_label} across {selected_set}")
        elif method == "all":
            ax.set_title(f"Distribution of {selected_label} values across {selected_set}")
        elif method == "dev":
            ax.set_title(f"Distribution of {selected_label} deviations across {selected_set}")
        ax.set_xlabel("value")
        ax.set_ylabel("count")
        ax.legend()

        return fig
