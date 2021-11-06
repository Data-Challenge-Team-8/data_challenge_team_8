import streamlit as st

from web.categories.descriptive_statistics import DescriptiveStatistics
from web.categories.exploratory_data_analysis import ExploratoryDataAnalysis
from web.categories.mathematical_statistics import MathematicalStatistics


def create_app():
    option = st.selectbox(
        'Choose your way of analysing the data:',
        (
            'Descriptive statistics',
            'Exploratory data analysis',
            'Mathematical statistics'
        )
    )
    if option == 'Descriptive statistics':
        desc_stat = DescriptiveStatistics(option)

    if option == 'Exploratory data analysis':
        expl_ana = ExploratoryDataAnalysis()

    if option == 'Mathematical statistics':
        math_stat = MathematicalStatistics()


create_app()
