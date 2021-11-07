import streamlit as st

from PIL import Image
from web.data_loader import DataLoader


def load_data_upfront():
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


def write_info_text():  # maybe: if questions or contact: our emails?
    info_p1 = "This dashboard was developed by students to display the data from the **PhysioNet Challenge 2019**. The goal of this challenge was the early prediction of sepsis based on clinical data. Data set A and B contain real patient data from two respective hospitals. We hope the implementation of this dashboard will be helpful to discover different patterns within the datasets."
    info_p2 = "For further information visit: https://physionet.org/content/challenge-2019/1.0.0/."
    info_p3 = "The following graphic displays the features collected in the dataset and their respective descriptions:"
    st.markdown(info_p1)
    st.markdown(info_p2)
    st.markdown(info_p3)


def display_feature_graphic():
    feature_graphic = Image.open(r'./data/feature_graphic.jpg')
    st.image(feature_graphic, caption='Descriptions for each feature from the underlying PhysioNet paper')


class LandingPage:
    def __init__(self, option: str):
        st.header("Helpful to load the data upfront:")
        load_data_upfront()
        st.header("General Information")
        write_info_text()
        display_feature_graphic()
