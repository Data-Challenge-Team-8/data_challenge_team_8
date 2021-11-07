import streamlit as st

from web.categories.descriptive_statistics import DescriptiveStatistics
from web.categories.exploratory_data_analysis import ExploratoryDataAnalysis
from web.categories.mathematical_statistics import MathematicalStatistics
from web.data_loader import DataLoader


def create_app():
    del_input = st.empty()
    sets = st.multiselect(
        'Choose what set to load before the app starts. This can save loading Time',
        ['Set A', 'Set B', 'Set A + B'],
        []
    )
    options = st.multiselect(
        'Choose what tool to load before the app starts. This can save loading Time',
        ['Min, Max, Average', 'Plots', 'Missing Values'],
        []
    )

    if st.button('Load Data'):

        if sets:
            if options:
                DataLoader().load_before(sets, options)

    methode = st.selectbox(
        'Choose your way of analysing the data:',
        (
            'Descriptive statistics',
            'Exploratory data analysis',
            'Mathematical statistics'
        )
    )

    if methode == 'Descriptive statistics':
        desc_stat = DescriptiveStatistics(methode)

    if methode == 'Exploratory data analysis':
        expl_ana = ExploratoryDataAnalysis(methode)

    if methode == 'Mathematical statistics':
        math_stat = MathematicalStatistics(methode)


create_app()
