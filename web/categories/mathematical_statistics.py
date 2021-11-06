from matplotlib import pyplot as plt
import streamlit as st
from IO.data_reader import DataReader
from objects.training_set import TrainingSet


class MathematicalStatistics:

    def __init__(self):
        self.__training_setA = {}
        self.create_temperature_sepsis()

    def create_temperature_sepsis(self):

        if TrainingSet("descriptive_statistics", self.__training_setA, ["create_temperature_sepsis"]).is_cached():
            temps = TrainingSet("descriptive_statistics", self.__training_setA, ["create_temperature_sepsis"]).get_temperature_sepsis()
        else:
            dr = DataReader()
            self.__training_setA = dr.training_setA
            temps = TrainingSet("descriptive_statistics", self.__training_setA, ["create_temperature_sepsis"]).get_temperature_sepsis()

        fig, ax1 = plt.subplots()
        ax1.hist(temps["sick"], density=True, bins=50, color="r")
        ax1.hist(temps["healthy"], density=True, bins=50, color="g", alpha=0.4)
        st.pyplot(fig)
