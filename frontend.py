import streamlit as st
import datetime

from IO.data_reader import DataReader
from objects.analyse_tools import AnalyseTool

if __name__ == '__main__':
    readData = DataReader()
    training_setA = readData.training_setA
    tool = AnalyseTool(training_setA)
option = st.selectbox(
    'Choos your tool.',
    ('min/max', 'average', 'Amount of missing values', 'subgroup', 'time series information'))
if option == 'subgroup':
    values = st.slider(
        'Select a range of values',
        0, 100, (25, 75))
    st.write('Values:', values)

if option == 'min/max':
    col1, col2 = st.columns(2)
    feature = col2.selectbox("Choose your feature", (
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
        "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
        "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets", "age", "Gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
        "SepsisLabel"))

    type = col1.radio("Choose your quarry type", (
        "single patient", "all patient"))
    type_func = col1.radio("Choose your function", (
        "min", "max"))

    if type_func == "min":
        if type == "single patient":
            number = col2.number_input('Insert a patient id')
            num = tool.min_single(feature, number)
            st.write(num)

    if type_func == "min":
        if type == "all patient":
            num = tool.min_all(feature)
            st.write(num)

    if type_func == "max":
        if type == "single patient":
            number = col2.number_input('Insert a patient id')
            num = tool.max_single(feature, number)
            st.write(num)

    if type_func == "max":
        if type == "all patient":
            num = tool.max_all(feature)
            st.write(num)
