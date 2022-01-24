import streamlit as st

from web.categories.landing_page import LandingPage
from web.categories.exploratory_data_analysis import ExploratoryDataAnalysis
from web.categories.sepsis_research import SepsisResearch
from web.categories.ts_analysis import TimeSeriesAnalysis


def create_app():
    # title = st.title("Dashboard for Sepsis Analysis")
    st.sidebar.write("Dashboard for Sepsis Analysis")

    methode = st.sidebar.selectbox(
        'Choose your way of analysing the data:',
        (
            'General Information',
            'Exploratory Data Analysis',
            'Sepsis Research',
            'Timeseries Analysis'
        )
    )

    if methode == 'General Information':
        landing_page = LandingPage()
    if methode == 'Exploratory Data Analysis':
        expl_ana = ExploratoryDataAnalysis()
    if methode == 'Sepsis Research':
        math_stat = SepsisResearch()
    if methode == 'Timeseries Analysis':
        math_stat = TimeSeriesAnalysis()


create_app()
