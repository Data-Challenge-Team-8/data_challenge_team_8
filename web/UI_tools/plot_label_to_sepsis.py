import numpy
import streamlit as st
from matplotlib import pyplot as plt

from IO.data_reader import DataReader
from objects.training_set import TrainingSet


class PlotLabelToSepsis:
    LABELS = ["Temp", "HR", "pH", "Age", "Gender"]

    def __init__(self, option, display: bool = True):
        self.__option = option
        self.__training_set = None
        col1, col2 = st.columns((1, 2))
        selected_label, selected_set, selected_sepsis = self.create_selectors(col1)
        self.create_plot(col2, selected_label, selected_set, selected_sepsis)

    def create_selectors(self, col1):
        selected_label = col1.selectbox(
            'Choose a label:',
            (
                self.LABELS
            )
        )

        selected_set = col1.selectbox(
            'Choose a Set:',
            (
                "Set A",
                "Set B",
                "Set A + B"
            )
        )

        selected_sepsis = col1.selectbox(
            'Choose if sepsis positive or negative:',
            (
                "positive + negative",
                "positive",
                "negative"
            )
        )
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

        if selected_sepsis == "positive + negative":  # user can select the sepsis state
            fig, ax1 = plt.subplots()
            ax1.hist([plot_data[0], plot_data[1]], density=True, color=['r', 'g'], bins=bins, alpha=0.6)

            col2.pyplot(fig)

        elif selected_sepsis == "positive":
            fig, ax1 = plt.subplots()
            ax1.hist(plot_data[0], bins=bins, alpha=0.6, color="r")
            col2.pyplot(fig)

        elif selected_sepsis == "negative":
            fig, ax1 = plt.subplots()
            ax1.hist(plot_data[1], bins=bins, alpha=0.6, color="g")
            col2.pyplot(fig)

