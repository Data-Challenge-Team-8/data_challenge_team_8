import streamlit as st
import pandas as pd
import numpy as np

option = st.selectbox(
     'Choos your tool.',
     ('min/max', 'average', 'Amount of missing values', 'subgroup', 'time series information'))
if option == 'subgroup':
    values = st.slider(
        'Select a range of values',
        0, 100, (25, 75))
    st.write('Values:', values)

if option == 'min/max':
    genre = st.radio("What's your favorite movie genre", ('Age', 'pH', 'HR'))



