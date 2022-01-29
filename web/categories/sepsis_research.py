import numpy
import streamlit as st
import statistics
from matplotlib import pyplot as plt
from PIL import Image

from tools.analyse_tool import CompleteAnalysis


def create_description():
    info_p1 = "This Sepsis Research Analysis focuses on displaying the relation of selected features and " \
              "the occurrence of sepsis. A histogram is used to visualize the collected data."
    st.markdown(info_p1)


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
    if selected_tool == "positive + negative":
        ax1.hist([plot_data[0], plot_data[1]], density=True, color=['r', 'g'], bins=bins, alpha=0.6)
    elif selected_tool == "positive":
        ax1.hist(plot_data[0], bins=bins, alpha=0.6, color="r")
    elif selected_tool == "negative":
        ax1.hist(plot_data[1], bins=bins, alpha=0.6, color="g")
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


# TODO: Aufgabe Aline: This should be a true calculation of correlations not just the .png - (probably with a cache otherwise it takes to long
def plot_correlations():
    feature_graphic = Image.open(r'./data/sepsis_correlation_fixed_values.png')
    st.image(feature_graphic, caption='Correlation of relevant Features to the SepsisLabel')

class SepsisResearch:
    LABELS = ["HR", "Resp", "Temp", "pH", "Age", "Gender", "ICOLUS"]  # Do we need further, more important Labels?

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

        st.markdown("<h2 style='text-align: left; color: black;'>Correlation of relevant Features to the SepsisLabel</h2>",
                    unsafe_allow_html=True)
        plot_correlations()


    def create_selectors(self, col1):
        selected_label = col1.selectbox('Choose a label:', self.LABELS)
        selected_set = col1.selectbox('Choose a Set:', ("Set A", "Set B", "Set A + B"))
        selected_sepsis = col1.selectbox('Choose if sepsis positive or negative:',
                                         ("positive + negative", "positive", "negative"))
        selected_tool = selected_sepsis
        return selected_label, selected_set, selected_tool


