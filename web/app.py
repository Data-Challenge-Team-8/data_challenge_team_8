import streamlit as st

from web.categories.landing_page import LandingPage
from web.categories.descriptive_statistics import DescriptiveStatistics
from web.categories.exploratory_data_analysis import ExploratoryDataAnalysis
from web.categories.mathematical_statistics import MathematicalStatistics


def create_app():
    title = st.title("Dashboard for Sepsis Analysis")

    st.sidebar.write("Test")

    methode = st.sidebar.selectbox(
        'Choose your way of analysing the data:',
        (
            'General Information',
            'Descriptive Statistics',
            'Exploratory Data Analysis',
            'Mathematical Statistics'
        )
    )

    if methode == 'General Information':
        landing_page = LandingPage(methode)

    if methode == 'Descriptive Statistics':
        desc_stat = DescriptiveStatistics(methode)

    if methode == 'Exploratory Data Analysis':
        expl_ana = ExploratoryDataAnalysis(methode)

    if methode == 'Mathematical Statistics':
        math_stat = MathematicalStatistics(methode)


create_app()
