import numpy
import numpy as np
import pandas as pd
import streamlit as st
import statistics
from matplotlib import pyplot as plt

from objects.patient import Patient
from objects.training_set import TrainingSet


class TimeSeriesAnalysis:

    def __init__(self):
        # Client selects the dataset he wants to analyse
        option_set = st.selectbox(
            'Select a data set:',
            ('Set A', 'Set B', 'Set A + B'))

        # we load the dataset
        dataset = TrainingSet.get_training_set(option_set)

        # Client enters a patient ID
        option_patient = st.text_input('Enter a patient ID (pXXXXXX):', 'p017475')
        # Client selects features
        option_features = st.multiselect(
            'Select a feature:',
            Patient.LABELS,
            ['HR', 'O2Sat'])

        # We load the patient
        patient = dataset.get_patient_form_id(option_patient)
        data_list = []
        for feature in option_features:
            series_feature = getattr(patient, feature)
            data = series_feature.interpolate()
            data_list.append(data)
        chart_data = pd.concat(data_list, axis=1)

        st.area_chart(chart_data)
        st.area_chart(getattr(patient, "sepsis_label"))
