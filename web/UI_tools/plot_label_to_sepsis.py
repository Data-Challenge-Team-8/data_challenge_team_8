from matplotlib import pyplot as plt
import streamlit as st
from IO.data_reader import DataReader
from objects.training_set import TrainingSet


class PlotLabelToSepsis:

    LABELS = ["Temp", "HR", "pH", "age", "gender"]

    def __init__(self, option):
        self.__option = option
        self.__training_set = None
        col1, col2 = st.columns(2)
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
        if TrainingSet("descriptive_statistics", self.__training_set, [self.__option, selected_label]).is_cached():

            plot_data = TrainingSet(
                "descriptive_statistics",  # id
                self.__training_set,  # data set
                [self.__option, selected_label]  # keys of selected options
            ).get_plot_label_to_sepsis(selected_label)

        else:
            dr = DataReader()
            if selected_set == "Set A":  # user can select his set
                self.__training_set = dr.training_setA
            elif selected_set == "Set B":
                self.__training_set = dr.training_setB
            else:
                self.__training_set = dr.combined_training_set

            plot_data = TrainingSet(
                "descriptive_statistics",  # id
                self.__training_set,  # data set
                [self.__option, selected_label]  # keys of selected options
            ).get_plot_label_to_sepsis(selected_label)

        if selected_sepsis == "positive + negative":  # user can select the sepsis state
            fig, ax1 = plt.subplots()
            ax1.hist(plot_data[0], density=True, bins=50, color="r")
            ax1.hist(plot_data[1], density=True, bins=50, color="g", alpha=0.4)
            col2.pyplot(fig)

        elif selected_sepsis == "positive":
            fig, ax1 = plt.subplots()
            ax1.hist(plot_data[0], density=True, bins=50, color="r")
            col2.pyplot(fig)

        elif selected_sepsis == "negative":
            fig, ax1 = plt.subplots()
            ax1.hist(plot_data[1], density=True, bins=50, color="g")
            col2.pyplot(fig)

