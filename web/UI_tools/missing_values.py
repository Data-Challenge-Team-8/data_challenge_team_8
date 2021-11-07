from IO.data_reader import DataReader
from objects.training_set import TrainingSet
import matplotlib.pyplot as plt
import streamlit as st


class MissingValues:

    def __init__(self, selected_tool: str, selected_label: str, selected_set: str, col3=None, col2=None, display: bool = True):
        self.__training_set = {}
        self.__selected_tool = selected_tool
        self.__selected_label = selected_label
        self.__selected_set = selected_set
        self.__col2 = col2
        self.__col3 = col3

        if display:
            missing_val = self.get_missing_vals()
            self.create_plot_missing_vals(missing_val)

    def get_missing_vals(self):
        if not TrainingSet(
                "exploratory_data_analysis_missing_values",
                self.__training_set,
                [self.__selected_tool, self.__selected_label, self.__selected_set]
        ).is_cached():
            dr = DataReader()
            if self.__selected_set == "Set A":
                self.__training_set = dr.training_setA
            elif self.__selected_set == "Set B":
                self.__training_set = dr.training_setB
            else:
                self.__training_set = dr.combined_training_set

        missing_vals_rel = TrainingSet(
            "exploratory_data_analysis_missing_values",
            self.__training_set,
            [self.__selected_tool, self.__selected_label, self.__selected_set]
        ).get_avg_rel_NaN_amount_for_label(self.__selected_label)
        return missing_vals_rel

    def create_plot_missing_vals(self, missing_vals_rel):
        fig, ax = plt.subplots()
        ax.pie([missing_vals_rel, 1 - missing_vals_rel], explode=[0.2, 0], colors=['r', 'g'])
        self.__col2.pyplot(fig)
        self.__col3.metric("Missing (red)", str(round((missing_vals_rel * 100))) + "%")
        self.__col3.metric("Not Missing (green)", str(round(((1 - missing_vals_rel) * 100))) + "%")
