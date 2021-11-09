import streamlit as st
from PIL import Image


def write_info_text():
    info_p1 = 'A descriptive statistic is a summary statistic that quantitatively describes or summarizes features.'
    st.markdown(info_p1)
    info_p2 = 'The following table offers an overview of general descriptive statistics about the datasets:'
    st.markdown(info_p2)


def display_table():                # TODO: get our own general info per dataset (amount of patients etc)
    feature_graphic = Image.open(r'./data/descriptive_table.jpg')
    st.image(feature_graphic, caption='Descriptions for each feature from the underlying PhysioNet paper')


class DescriptiveStatistics:
    def __init__(self, option: str):
        st.markdown("<h2 style='text-align: left; color: black;'>Descriptive Statistics</h2>", unsafe_allow_html=True)
        write_info_text()
        display_table()
