import streamlit as st

from PIL import Image
from web.data_loader import DataLoader


def load_data_upfront():
    st.markdown("<h2 style='text-align: left; color: black;'>Recommended to load the data upfront:</h2>", unsafe_allow_html=True)
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


def display_feature_graphic():
    feature_graphic = Image.open(r'./data/feature_graphic.jpg')
    st.image(feature_graphic, caption='Descriptions for each feature from the underlying PhysioNet paper')


def display_table():  # TODO: get our own general info per dataset (amount of patients etc)
    feature_graphic = Image.open(r'./data/descriptive_table.jpg')
    st.image(feature_graphic, caption='Descriptions for each feature from the underlying PhysioNet paper')


def write_info_text():  # maybe: if you have any questions or contact: our emails?
    st.markdown("<h2 style='text-align: left; color: black;'>Project Description</h2>", unsafe_allow_html=True)
    info_p1 = "This dashboard was developed by students to display the data from the **PhysioNet Challenge 2019**. " \
              "The goal of this challenge was the early prediction of sepsis based on clinical data. " \
              "Data set A and B contain real patient data from two respective hospitals. " \
              "We hope the implementation of this dashboard will be helpful to discover " \
              "different patterns within the datasets."
    st.markdown(info_p1)
    info_p2 = "For further information visit: https://physionet.org/content/challenge-2019/1.0.0/."
    st.markdown(info_p2)
    info_p3 = 'The following table offers an overview of general descriptive statistics about the datasets:'
    st.markdown(info_p3)
    display_table()
    info_p4 = "Below the features collected in the dataset and their respective descriptions are displayed:"
    st.markdown(info_p4)
    display_feature_graphic()


class LandingPage:
    def __init__(self, option: str):
        load_data_upfront()
        write_info_text()







