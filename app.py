import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained XGBoost model
with open("Xgboost_Regression.pkl", "rb") as file:
    model = pickle.load(file)

# Title and description
st.title("California Housing Price Prediction")
st.write(
    "This web app uses a trained XGBoost model to predict housing prices. "
    "Please enter the values for the following features to get a prediction."
)
st.markdown("---")

# Input fields for the model's features
st.sidebar.header("Input Features")

med_inc = st.sidebar.number_input("Median Income (in tens of thousands)", min_value=0.0, value=3.87)
house_age = st.sidebar.number_input("House Age (in years)", min_value=1.0, value=28.0)
ave_rooms = st.sidebar.number_input("Average Number of Rooms", min_value=0.0, value=5.4)
ave_bedrms = st.sidebar.number_input("Average Number of Bedrooms", min_value=0.0, value=1.1)
population = st.sidebar.number_input("Population", min_value=0.0, value=1425.0)
ave_occup = st.sidebar.number_input("Average House Occupancy", min_value=0.0, value=3.0)
latitude = st.sidebar.number_input("Latitude", min_value=32.0, max_value=42.0, value=35.6)
longitude = st.sidebar.number_input("Longitude", min_value=-125.0, max_value=-114.0, value=-119.5)


# Collect inputs into a DataFrame
# The feature names are based on the training data of the model.
input_data = pd.DataFrame(
    [
        [
            med_inc,
            house_age,
            ave_rooms,
            ave_bedrms,
            population,
            ave_occup,
            latitude,
            longitude,
        ]
    ],
    columns=[
        "MedInc",
        "HouseAge",
        "AveRooms",
        "AveBedrms",
        "Population",
        "AveOccup",
        "Latitude",
        "Longitude",
    ],
)

st.subheader("Input Features")
st.write(input_data)

# Predict when button is clicked
if st.button("Predict House Price"):
    prediction = model.predict(input_data)
    st.success(f"**Predicted Housing Price:** ${prediction[0]:,.2f}")