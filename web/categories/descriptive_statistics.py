import streamlit as st
from matplotlib import pyplot as plt
from IO.data_reader import DataReader
from objects.training_set import TrainingSet


class DescriptiveStatistics:
    DESCRIPTION = 'A descriptive statistic is a summary statistic that quantitatively describes or summarizes ' \
                  'features from a collection of information, while descriptive statistics (in the mass noun sense) ' \
                  'is the process of using and analysing those statistics. '

    def __init__(self, option: str):
        self.create_description()
        self.__training_setA = {}

    def create_description(self):
        st.write(self.DESCRIPTION)
