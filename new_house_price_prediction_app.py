import streamlit as st
import pandas as pd
import joblib
import numpy as np
import urllib.request
import os

# URLs to your model and pipeline files hosted online
MODEL_URL = "YOUR_PUBLIC_MODEL_FILE_URL_HERE"
PIPELINE_URL = "YOUR_PUBLIC_PIPELINE_FILE_URL_HERE"
MODEL_FILENAME = "final_model.pkl"
PIPELINE_FILENAME = "full_pipeline.pkl"

@st.cache_resource # Cache the loading process
def load_resources():
    # Download the model file
    if not os.path.exists(MODEL_FILENAME):
        st.info(f"Downloading {MODEL_FILENAME}...")
        urllib.request.urlretrieve(MODEL_URL, MODEL_FILENAME)
        st.success(f"Downloaded {MODEL_FILENAME}")

    # Download the pipeline file
    if not os.path.exists(PIPELINE_FILENAME):
        st.info(f"Downloading {PIPELINE_FILENAME}...")
        urllib.request.urlretrieve(PIPELINE_URL, PIPELINE_FILENAME)
        st.success(f"Downloaded {PIPELINE_FILENAME}")

    # Load the saved pipeline and model
    try:
        full_pipeline = joblib.load(PIPELINE_FILENAME)
        final_model = joblib.load(MODEL_FILENAME)
        return full_pipeline, final_model
    except Exception as e:
        st.error(f"Error loading model or pipeline: {e}")
        st.stop()

# Load the resources (this function is cached)
full_pipeline, final_model = load_resources()


# Streamlit App Title and Description
st.title("California Housing Price Predictor")
st.write("Enter the details of a house in California to get a predicted median house value.")

# Input Widgets for Features (sliders)
st.header("House Features")

# Define reasonable ranges for sliders based on your data exploration
# For features where ranges are highly variable (like total_rooms), number_input might be better than sliders
longitude = st.slider("Longitude", min_value=-124.35, max_value=-114.31, value=-118.24, help="A measure of how far west a house is; a higher value is farther west")
latitude = st.slider("Latitude", min_value=32.54, max_value=41.95, value=34.05, help="A measure of how far north a house is; a higher value is farther north")
housing_median_age = st.slider("Housing Median Age", min_value=1, max_value=52, value=25, help="Median age of a house block")
total_rooms = st.slider("Total Rooms", min_value=2, max_value=39320, value=2000, help="Total number of rooms within a block")
total_bedrooms = st.slider("Total Bedrooms", min_value=1, max_value=6445, value=400, help="Total number of bedrooms within a block")
population = st.slider("Population", min_value=3, max_value=35682, value=1000, help="Total number of people residing within a block")
households = st.slider("Households", min_value=1, max_value=6082, value=350, help="Total number of households, a group of people residing within a home unit")
median_income = st.slider("Median Income", min_value=0.4999, max_value=15.0001, value=3.5, format="%.4f", help="Median income for households within a block") # Adjusted format for precision

# Categorical feature
ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
ocean_proximity = st.selectbox("Ocean Proximity", ocean_proximity_options, help="Location of the house with respect to the ocean")

# Prediction Button
if st.button("Predict Median House Value"):
    # Prepare Input Data
    input_data = pd.DataFrame([[
        longitude,
        latitude,
        housing_median_age,
        total_rooms,
        total_bedrooms,
        population,
        households,
        median_income,
        ocean_proximity
    ]], columns=[
        'longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income',
        'ocean_proximity'
    ])

    # Use the loaded pipeline to transform the input data
    try:
        prepared_input_data = full_pipeline.transform(input_data)

        # Make Prediction
        prediction = final_model.predict(prepared_input_data)

        # Display Prediction
        st.success(f"The predicted median house value is: ${prediction[0]:,.2f}")

    except Exception as e:
        st.error(f"An error occurred during prediction: {e}")

# Section to Evaluate Model on a Known Example 
st.markdown("---")
st.header("Evaluate Model on a Known Example")
st.write("Enter the details of a house with a known value to see how the model performs.")

# Input widgets for a known example (you can use sliders or number inputs here)
eval_longitude = st.slider("Known Example Longitude", min_value=-124.35, max_value=-114.31, value=-118.24)
eval_latitude = st.slider("Known Example Latitude", min_value=32.54, max_value=41.95, value=34.05)
eval_housing_median_age = st.slider("Known Example Housing Median Age", min_value=1, max_value=52, value=25)
eval_total_rooms = st.slider("Known Example Total Rooms", min_value=2, max_value=39320, value=2000)
eval_total_bedrooms = st.slider("Known Example Total Bedrooms", min_value=1, max_value=6445, value=400)
eval_population = st.slider("Known Example Population", min_value=3, max_value=35682, value=1000)
eval_households = st.slider("Known Example Households", min_value=1, max_value=6082, value=350)
eval_median_income = st.slider("Known Example Median Income", min_value=0.4999, max_value=15.0001, value=3.5, format="%.4f")
eval_ocean_proximity_options = ['<1H OCEAN', 'INLAND', 'NEAR OCEAN', 'NEAR BAY', 'ISLAND']
eval_ocean_proximity = st.selectbox("Known Example Ocean Proximity", eval_ocean_proximity_options)
true_house_value = st.number_input("Known True House Value", min_value=14999.00, format="%.2f", help="The actual median house value for this example.") # Set min based on your data

if st.button("Evaluate Model"):
    # Create DataFrame for the known example
    eval_input_data = pd.DataFrame([[
        eval_longitude,
        eval_latitude,
        eval_housing_median_age,
        eval_total_rooms,
        eval_total_bedrooms,
        eval_population,
        eval_households,
        eval_median_income,
        eval_ocean_proximity
    ]], columns=[
        'longitude',
        'latitude',
        'housing_median_age',
        'total_rooms',
        'total_bedrooms',
        'population',
        'households',
        'median_income',
        'ocean_proximity'
    ])

    # Transform the input data
    try:
        prepared_eval_data = full_pipeline.transform(eval_input_data)

        # Make prediction
        eval_prediction = final_model.predict(prepared_eval_data)

        # Display results
        st.write(f"True Median House Value: ${true_house_value:,.2f}")
        st.write(f"Model's Predicted Median House Value: ${eval_prediction[0]:,.2f}")

        # Calculate and display the difference
        difference = eval_prediction[0] - true_house_value
        st.write(f"Difference (Prediction - True): ${difference:,.2f}")
        st.write(f"Percentage Difference: {((difference / true_house_value) * 100):.2f}%")

    except Exception as e:
        st.error(f"An error occurred during evaluation: {e}")