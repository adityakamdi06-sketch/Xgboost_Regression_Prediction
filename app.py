Here is the complete and final code for your Streamlit application.

### \#\# Application Code (`app.py`)

Save this code in a file named `app.py`. Make sure your `Xgboost_Regression.pkl` file is in the same folder.

```python
import streamlit as st
import numpy as np
import pickle
import pandas as pd

# Load the trained XGBoost model
try:
    with open("Xgboost_Regression.pkl", "rb") as file:
        model = pickle.load(file)

except FileNotFoundError:
    st.error("Model file (Xgboost_Regression.pkl) not found. Please ensure it's in the same directory as the app.")
    st.stop()


# --- App Title and Description ---
st.title("California Housing Price Prediction 🏠")
st.write(
    "This web app uses a trained XGBoost model to predict housing prices in California. "
    "Enter the values for the features in the sidebar to get a prediction."
)
st.markdown("---")


# --- Sidebar for Inputs ---
st.sidebar.header("Input Features")

# A dictionary to hold default values for a more organized approach
default_values = {
    "MedInc": 3.87,
    "HouseAge": 28.0,
    "AveRooms": 5.4,
    "AveBedrms": 1.1,
    "Population": 1425.0,
    "AveOccup": 3.0,
    "Latitude": 35.6,
    "Longitude": -119.5,
}

# Create input fields using the feature names from the model
med_inc = st.sidebar.number_input(
    "Median Income (in tens of thousands)",
    min_value=0.0,
    value=default_values["MedInc"],
    step=0.1,
)
house_age = st.sidebar.number_input(
    "House Age (in years)", min_value=1.0, value=default_values["HouseAge"], step=1.0
)
ave_rooms = st.sidebar.number_input(
    "Average Number of Rooms",
    min_value=0.0,
    value=default_values["AveRooms"],
    step=0.1,
)
ave_bedrms = st.sidebar.number_input(
    "Average Number of Bedrooms",
    min_value=0.0,
    value=default_values["AveBedrms"],
    step=0.1,
)
population = st.sidebar.number_input(
    "Population", min_value=0.0, value=default_values["Population"], step=100.0
)
ave_occup = st.sidebar.number_input(
    "Average House Occupancy",
    min_value=0.0,
    value=default_values["AveOccup"],
    step=0.1,
)
latitude = st.sidebar.number_input(
    "Latitude", min_value=32.0, max_value=42.0, value=default_values["Latitude"]
)
longitude = st.sidebar.number_input(
    "Longitude",
    min_value=-125.0,
    max_value=-114.0,
    value=default_values["Longitude"],
)


# --- Prediction Logic ---
# Collect inputs into a DataFrame. The column names must match the model's expected features.
feature_names = [
    "MedInc",
    "HouseAge",
    "AveRooms",
    "AveBedrms",
    "Population",
    "AveOccup",
    "Latitude",
    "Longitude",
]
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
    columns=feature_names,
)

# Display the user's input
st.subheader("Your Input Features")
st.dataframe(input_data)

# Predict when the button is clicked
if st.button("Predict House Price", type="primary"):
    try:
        prediction = model.predict(input_data)
        st.success(f"**Predicted Housing Price:** `${prediction[0]:,.2f}`")
    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

```