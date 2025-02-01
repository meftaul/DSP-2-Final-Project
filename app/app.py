import streamlit as st
import pandas as pd
import joblib

st.title('House Price Prediction App')

st.sidebar.header('User Input Parameters')

longitude =  st.sidebar.slider('Longitude', -124.35, -114.31, -124.23)
latitude = st.sidebar.slider('Latitude', 32.54, 42.01, 40.54)
housing_median_age = st.sidebar.text_input('Housing Median Age', 52.0)
total_rooms = st.sidebar.text_input('Total Rooms', 2694.0)
total_bedrooms = st.sidebar.text_input('Total Bedrooms', 453.0)
population = st.sidebar.text_input('Population', 1152.0)
households = st.sidebar.text_input('Households', 435.0)
median_income = st.sidebar.text_input('Median Income', 3.0806)
median_house_value = st.sidebar.text_input('Median House Value', 106700.0)
ocean_proximity = st.sidebar.selectbox('Ocean Proximity', ['<1H OCEAN', 'INLAND', 'ISLAND', 'NEAR BAY', 'NEAR OCEAN'])


input_data = {
    'longitude': longitude, 
    'latitude': latitude, 
    'housing_median_age': housing_median_age, 
    'total_rooms': total_rooms, 
    'total_bedrooms': total_bedrooms, 
    'population': population, 
    'households': households, 
    'median_income': median_income, 
    'median_house_value': median_house_value, 
    'ocean_proximity': ocean_proximity
}

input_data_df = pd.DataFrame([input_data])

model = joblib.load('model_with_pipeline.pkl')

result = model.predict(input_data_df)

st.table(input_data_df)

st.metric('Predicted House Price', f'{result[0]:,.2f}')