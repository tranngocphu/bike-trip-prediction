import streamlit as st
from joblib import load
import pandas as pd
import numpy as np
import os

current_path = os.path.dirname(os.path.realpath(__file__))

from src.preprocessing import AddPeakAndMinutes  # this's required to load the pipeline from file
data_pipe = load(current_path + "/src/preprocess.joblib")
print("Data preprocessing pipeline loaded: ", data_pipe)

xgb_model = load(current_path + "/src/xgb_V1.joblib")
print("Prediction model loaded: ", xgb_model)


def predict_duration(input_features):
    pred = xgb_model.predict(input_features)
    print("Predicted duration:", int(pred[0]))
    return int(pred[0])


st.set_page_config(
    page_title='Seoul Bike Trip Duration Predictor',
    page_icon="🚲",
    layout="wide"
)

with st.form("input_form"):
    
    st.header("Enter data for prediction:")

    distance = st.number_input("Distance: ", value=10, format="%d")
    haversine = st.number_input("Haversine: ", value=10, format="%d")
    pmonth = st.slider("Pickup Month: ", 1, 12, value=6, format="%d") 
    pday = st.slider("Pickup Date: ", 1, 31, value=15, format="%d")
    phour = st.slider("Pickup Hour: ", 0, 23, value=12, format="%d")
    pmin = st.slider("Pickup Minute: ", 0, 59, value=30, format="%d")
    pweek = st.slider("Pickup Weekdays: ", 0, 6, value=3, format="%d")    
    gtemp = st.number_input("Ground Temp: ", value=1)
    humid = st.number_input("Humid: ", value=1)
    solar = st.number_input("Solar: ", value=1)
    dust = st.number_input("Dust: ", value=1)
    wind = st.number_input("Wind: ", value=1)

    submission = st.form_submit_button("Predict Duration")


if submission:

    input_df = pd.DataFrame(data=[{
        'Unnamed: 0': np.nan,
        'Distance': distance,
        'PLong': np.nan,
        'PLatd': np.nan,
        'DLong': np.nan,
        'DLatd': np.nan,
        'Haversine': haversine,
        'Pmonth': pmonth,
        'Pday': pday,
        'Phour': phour,
        'Pmin': pmin,
        'PDweek': pweek,
        'Dmonth': np.nan,
        'Dday': np.nan,
        'Dhour': np.nan,
        'Dmin': np.nan,
        'DDweek': np.nan,
        'Temp': np.nan,
        'Precip': np.nan,
        'Wind': wind,
        'Humid': humid,
        'Solar': solar,
        'Snow': np.nan,        
        'GroundTemp': gtemp,
        'Dust': dust,        
    }])

    prediction = predict_duration(data_pipe.transform(input_df))

    st.header('Results:')
    st.success(f"The predicted duration is {prediction} mins")

