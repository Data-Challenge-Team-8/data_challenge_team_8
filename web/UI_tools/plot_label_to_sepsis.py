import numpy
import streamlit as st
import statistics
from matplotlib import pyplot as plt

from IO.data_reader import DataReader
from objects.training_set import TrainingSet


class PlotLabelToSepsis:
    """
    Loads the data of a selected set and plots a histogram depending on selectbox

    """
    LABELS = ["Temp", "HR", "pH", "Age", "Gender"]

    def __init__(self, option):
        self.__option = option
        self.__training_set = None
        col1, col2 = st.columns((1, 2))
        selected_label, selected_set, selected_sepsis = self.create_selectors(col1)
        self.create_plot(col2, selected_label, selected_set, selected_sepsis)

    def create_selectors(self, col1):
        selected_label = col1.selectbox('Choose a label:', self.LABELS)
        selected_set = col1.selectbox('Choose a Set:', ("Set A", "Set B", "Set A + B"))
        selected_sepsis = col1.selectbox('Choose if sepsis positive or negative:',
                                         ("positive + negative", "positive", "negative"))
        return selected_label, selected_set, selected_sepsis

    def create_plot(self, col2, selected_label, selected_set, selected_sepsis):
        if not TrainingSet(
                "mathematical_statistics",
                self.__training_set,
                [self.__option, selected_label, selected_set]
        ).is_cached():
            dr = DataReader()
            if selected_set == "Set A":  # user can select his set
                self.__training_set = dr.training_setA
            elif selected_set == "Set B":
                self.__training_set = dr.training_setB
            else:
                self.__training_set = dr.combined_training_set

        analyse_set = TrainingSet(
            "mathematical_statistics",  # id
            self.__training_set,  # data set
            [self.__option, selected_label, selected_set]  # keys of selected options
        )
        plot_data = analyse_set.get_plot_label_to_sepsis(selected_label)

        # getting the min max average to scale the plot proportional
        min_val = analyse_set.get_min_for_label(selected_label)
        max_val = analyse_set.get_max_for_label(selected_label)
        bins = numpy.linspace(float(min_val[1]), float(max_val[1]), 100 if selected_label != 'Gender' else 2)

        # Actually Plotting the Histogram
        fig, ax1 = plt.subplots()
        fig.title = "Histogram"  # doesnt work
        if selected_sepsis == "positive + negative":  # user can select the sepsis state
            ax1.hist([plot_data[0], plot_data[1]], density=True, color=['r', 'g'], bins=bins, alpha=0.6)
        elif selected_sepsis == "positive":
            ax1.hist(plot_data[0], bins=bins, alpha=0.6, color="r")
        elif selected_sepsis == "negative":
            ax1.hist(plot_data[1], bins=bins, alpha=0.6, color="g")
        col2.pyplot(fig)

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
