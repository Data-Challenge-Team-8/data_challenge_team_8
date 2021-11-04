import streamlit as st
from matplotlib import pyplot as plt
from matplotlib.pyplot import figure

from read_data import ReadData
import numpy as np

read_data = ReadData()
option = st.selectbox(
    'Choos your tool.',
    ('min, max and average all data', 'Amount of missing values', 'single patient', 'time series information'))

if option == 'Amount of missing values':
    col1, col2 = st.columns(2)
    feature = col1.selectbox("Choose your feature", (
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
        "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
        "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets", "age", "gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
        "SepsisLabel"))
    missing_vals = read_data.missing_val_labels(feature)
    col2.text("This label is missing")
    col2.text(missing_vals)
    col2.text("values.")

    labels = [key for key in read_data.missing["missing_vals"]]
    x = [read_data.missing["missing_vals"][key] for key in read_data.missing["missing_vals"]]
    x, labels = zip(*sorted(zip(x, labels)))
    y_pos = np.arange(len(labels))
    color = []
    for l in labels:
        if l == feature:
            color.append('r')
        else:
            color.append('green')

    fig, ax = plt.subplots()
    ax.barh(y_pos, x, color=color)
    fig.set_size_inches(10, 15)
    ax.set_yticks(y_pos)
    ax.set_yticklabels(labels)
    ax.set_ylabel('Labels')
    ax.set_xlabel('Amount of missing values')
    st.pyplot(fig)

if option == 'min, max and average all data':
    col1, col2 = st.columns(2)
    feature = col2.selectbox("Choose your feature", (
        "HR", "O2Sat", "Temp", "SBP", "MAP", "DBP", "Resp", "EtCO2", "BaseExcess", "FiO2", "pH", "PaCO2", "SaO2",
        "AST", "BUN", "Alkalinephos", "Calcium", "Chloride", "Creatinine", "Bilirubin_direct", "Glucose",
        "Lactate", "Magnesium", "Phosphate", "Potassium", "Bilirubin_total", "TroponinI", "Hct", "Hgb",
        "PTT", "WBC", "Fibrinogen", "Platelets", "age", "gender", "Unit1", "Unit2", "HospAdmTime", "ICULOS",
        "SepsisLabel"))

    type_func = col1.radio("Choose your function", (
        "min", "max", "avg"))

    if type_func == "min":
        min_val = read_data.min_val_labels(feature)
        st.text(min_val)

    if type_func == "max":
        max_val = read_data.max_val_labels(feature)
        st.text(max_val)

    if type_func == "avg":
        avg_val = read_data.avg_val_labels(feature)
        st.text(avg_val)
