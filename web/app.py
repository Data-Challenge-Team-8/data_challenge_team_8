import streamlit as st

from web.categories.general_information import GeneralInformation
from web.categories.descriptive_statistics import DescriptiveStatistics
from web.categories.exploratory_data_analysis import ExploratoryDataAnalysis
from web.categories.mathematical_statistics import MathematicalStatistics


def create_app():
    title = st.title("Dashboard for Sepsis Analysis")

    methode = st.selectbox(
        'Choose your way of analysing the data:',
        (
            'General Information',
            'Descriptive Statistics',
            'Exploratory Data Analysis',
            'Mathematical Statistics'
        )
    )

    if methode == 'General Information':
        general_info = GeneralInformation(methode)

    if methode == 'Descriptive Statistics':
        desc_stat = DescriptiveStatistics(methode)

    if methode == 'Exploratory Data Analysis':
        expl_ana = ExploratoryDataAnalysis(methode)

    if methode == 'Mathematical Statistics':
        math_stat = MathematicalStatistics(methode)


create_app()
