import streamlit as st

from web.UI_tools.plot_label_to_sepsis import PlotLabelToSepsis


class SepsisResearch:

    def __init__(self, option: str):
        self.__option = option
        st.markdown("<h2 style='text-align: left; color: black;'>Mathematical Statistics</h2>", unsafe_allow_html=True)
        PlotLabelToSepsis(option)
