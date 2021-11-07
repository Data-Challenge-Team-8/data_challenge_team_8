from matplotlib import pyplot as plt
import streamlit as st
from IO.data_reader import DataReader
from objects.training_set import TrainingSet
from web.UI_tools.plot_label_to_sepsis import PlotLabelToSepsis


class MathematicalStatistics:

    def __init__(self, option):
        self.__option = option
        self.__training_set = None
        PlotLabelToSepsis(option)
