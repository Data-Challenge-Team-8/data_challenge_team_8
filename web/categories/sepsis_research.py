import os.path

import numpy
import pickle
import streamlit as st
import statistics
from matplotlib import pyplot as plt
import seaborn as sb
from PIL import Image

from tools.analyse_tool import CompleteAnalysis
from objects.training_set import TrainingSet
from objects.patient import Patient


def create_description():
    info_p1 = "This Sepsis Research Analysis focuses on displaying the relation of selected features and " \
              "the occurrence of sepsis. A histogram is used to visualize the collected data."
    st.markdown(info_p1)


SEPSIS_TOOL_CHOICE = {
    "both": "positive + negative",
    "sepsis": "positive",
    "no_sepsis": "negative",
}


def plot_sepsis_analysis(analysis_obj, col2, selected_label, selected_tool):
    min_val = analysis_obj.min_for_label[selected_label][1]
    max_val = analysis_obj.max_for_label[selected_label][1]
    plot_data = analysis_obj.plot_label_to_sepsis[selected_label]
    # getting the min max average to scale the plot proportional
    bins = numpy.linspace(float(min_val), float(max_val),
                          100 if selected_label != 'Gender' else 2)  # removed [1] from min_val
    # Actually Plotting the Histogram
    fig, ax1 = plt.subplots()
    fig.title = "Histogram"  # doesnt work
    if selected_tool == SEPSIS_TOOL_CHOICE["both"]:
        ax1.hist([plot_data[0], plot_data[1]], density=True, color=['r', 'g'], bins=bins, alpha=0.6,
                 label=["Sepsis", "No Sepsis"])
    elif selected_tool == SEPSIS_TOOL_CHOICE["sepsis"]:
        ax1.hist(plot_data[0], bins=bins, alpha=0.6, color="r", label="Sepsis")
    elif selected_tool == SEPSIS_TOOL_CHOICE["no_sepsis"]:
        ax1.hist(plot_data[1], bins=bins, alpha=0.6, color="g", label="No Sepsis")
    ax1.legend()
    col2.pyplot(fig)
    headline = "Further Statistics for the label " + selected_label + ": "
    st.subheader(headline)
    # Displaying further Statistics
    sepsis_mean = round(statistics.mean(plot_data[0]), 5)
    sepsis_median = round(statistics.median(plot_data[0]), 5)
    sepsis_var = round(statistics.variance(plot_data[0]), 5)
    no_sepsis_mean = round(statistics.mean(plot_data[1]), 5)
    no_sepsis_median = round(statistics.median(plot_data[1]), 5)
    no_sepsis_var = round(statistics.variance(plot_data[1]), 5)
    diff_mean = round(sepsis_mean - no_sepsis_mean, 5)
    diff_median = round(sepsis_median - no_sepsis_median, 5)
    diff_var = round(sepsis_var - no_sepsis_var, 5)
    col0, col1, col2, col3 = st.columns(4)
    col0.markdown("**Sepsis**")
    col1.metric("Average", sepsis_mean, diff_mean)
    col2.metric("Median", sepsis_median, diff_median)
    col3.metric("Variance", sepsis_var, diff_var)
    col0, col1, col2, col3 = st.columns(4)
    col0.markdown("**No Sepsis**")
    col1.metric("", no_sepsis_mean)
    col2.metric("", no_sepsis_median)
    col3.metric("", no_sepsis_var)


class SepsisResearch:

    CACHE_CORRELATION_POSTFIX = "frontend-correlation"

    def __init__(self):
        st.markdown("<h2 style='text-align: left; color: black;'>Histogram for Sepsis Research</h2>",
                    unsafe_allow_html=True)
        create_description()
        col1, col2 = st.columns((1, 2))
        selected_label, selected_set, selected_tool = self.create_selectors(col1)
        analysis_obj, file_name = CompleteAnalysis.get_analysis(selected_label=selected_label,
                                                                selected_tool=selected_tool,
                                                                selected_set=selected_set)
        plot_sepsis_analysis(analysis_obj, col2, selected_label, selected_tool)

        st.markdown(
            "<h2 style='text-align: left; color: black;'>Correlation of relevant Features</h2>",
            unsafe_allow_html=True)
        col1, col2 = st.columns((1, 2))
        selected_label, selected_set, use_fix_missing_values, use_interpolation\
            = self.__create_correlation_selectors(col1)
        self.__plot_correlations(set=TrainingSet.get_training_set(selected_set), label=selected_label,
                                 fix_missing_values=use_fix_missing_values, use_interpolation=use_interpolation,
                                 col=col2)

    def create_selectors(self, col1):
        selected_label = col1.selectbox('Choose a label:', Patient.LABELS)
        selected_set = col1.selectbox('Choose a Set:', TrainingSet.PRESETS.keys())
        selected_sepsis = col1.selectbox('Choose if sepsis positive or negative:',
                                         tuple(SEPSIS_TOOL_CHOICE.values()))
        selected_tool = selected_sepsis
        return selected_label, selected_set, selected_tool

    def __create_correlation_selectors(self, col):
        sepsislabel_indices = [i for i in range(len(Patient.LABELS)) if Patient.LABELS[i] == "SepsisLabel"]
        sepsislabel_index = sepsislabel_indices[0] if len(sepsislabel_indices) > 0 else None
        sample_a_indices = [i for i in range(len(TrainingSet.PRESETS.keys())) if list(TrainingSet.PRESETS.keys())[i] == "rnd Sample A"]
        sample_a_index = sample_a_indices[0] if len(sepsislabel_indices) > 0 else None
        selected_label = col.selectbox("Choose a Label", Patient.LABELS, index=sepsislabel_index, key="corrLabel")          # might be useful to only offer labels that are actually in the selected_set
        selected_set = col.selectbox("Choose a Set:", TrainingSet.PRESETS.keys(), key="corrSet", index=sample_a_index)
        use_fix_missing_values = col.checkbox("Use \"fix missing\"", key="corrFix", value=True)
        use_interpolation = col.checkbox("Use interpolation", key="corrInterpolate", value=True)

        return selected_label, selected_set, use_fix_missing_values, use_interpolation

    def __plot_correlations(self, set: TrainingSet, col, label: str, fix_missing_values: bool, use_interpolation: bool):
        avg_df = set.get_average_df(fix_missing_values=fix_missing_values,
                                    use_interpolation=use_interpolation)
        file_path = set.get_cache_file_path(SepsisResearch.CACHE_CORRELATION_POSTFIX)
        curr_version = "fixed" if fix_missing_values else "no_fixed" + \
                                                          "interpolated" if use_interpolation else "no_interp"
        sorted_corr_df = None
        if not os.path.isfile(file_path):
            print("Frontend SepsisResearch found no cache!")
            d = None
        else:
            print("Frontend SepsisResearch is using cache file:", file_path)
            d = pickle.load(open(file_path, 'rb'))
            if f"sorted_corr_df_{curr_version}" in d.keys():
                sorted_corr_df = d[f'sorted_corr_df_{curr_version}']
            avg_df_corr_without_nan = d['avg_df_corr']
            feature_names = avg_df_corr_without_nan.columns

        if sorted_corr_df is None:  # cache not found or empty
            print("Cache was not found or empty! Calculating ...")
            sepsis_df = set.get_sepsis_label_df()  # no transpose needed
            transposed_df = avg_df.transpose()
            added_sepsis_df = transposed_df
            added_sepsis_df["SepsisLabel"] = sepsis_df.iloc[0:].values
            added_sepsis_df = added_sepsis_df.fillna(0)  # fix NaN problem

            avg_df_corr = added_sepsis_df.corr()
            feature_names = avg_df_corr.columns
            avg_df_corr_without_nan = avg_df_corr.fillna(0)  # if features have no values they are none
            corr_df = avg_df_corr_without_nan[label]
            sorted_corr_df = corr_df.sort_values(ascending=False)
            print("Frontend SepsisResearch writes to cache file:", file_path)
            if d is None:
                pickle.dump({f'sorted_corr_df_{curr_version}': sorted_corr_df, "avg_df_corr": avg_df_corr_without_nan},
                            open(file_path, 'wb'))
            else:
                d[f'sorted_corr_df_{curr_version}'] = sorted_corr_df
                pickle.dump(d, open(file_path, 'wb'))


        # Bar plot of correlation to label
        try:
            fig, ax1 = plt.subplots(1)
            plot_corr_df = sorted_corr_df.drop(label)
            ax1.bar(plot_corr_df.index, plot_corr_df)# = plot_corr_df.plot.bar(x='Features')
            ax1.set_xlabel("Features")
            f = 'fixed values,' if fix_missing_values else ''
            i = 'quadratic interpolation' if use_interpolation else ''
            ax1.set_title(f"Correlation to {label}, {f} {i}")
            props = {"rotation": 90}
            plt.setp(ax1.get_xticklabels(), **props)

            # heat map of feature x feature
            #ax2 = sb.heatmap(data=avg_df_corr_without_nan.to_numpy(), vmin=-1, vmax=1, linewidths=0.5,
            #                 cmap='bwr', yticklabels=feature_names)
            #ax2.set_title(f"Correlations in {set.name}, {f}"
            #              f"{i}")

            # pair plot of greatest features to label
            #important_features = sorted_corr_df.index[:3].tolist()
            #important_features.extend(sorted_corr_df.index[-3:].tolist())
            #selected_labels_df = avg_df.transpose().filter(important_features, axis=1)
            #avg_df_small = selected_labels_df.iloc[:100]  # scatter plot nur 100 patients
            #sb.set_style('darkgrid')
            #pairplot = sb.pairplot(avg_df_small)
            #col.pyplot(pairplot)

            col.pyplot(fig)

        except KeyError:
            info_key_error = "The selected label was not found in the selected dataset. It was probably removed" \
                             " within the imputation. Please select a different label or a different dataset."
            st.markdown(info_key_error)

