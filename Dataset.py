# Import the necessary libraries
import streamlit as st
import pandas as pd

# Configure the Streamlit page settings
st.set_page_config(page_title="Weather Prediction App Project", page_icon=":sun_behind_rain_cloud:")

st.header("Dataset")

# Explain the dataset
st.write("This dataset contains weather-related information, sourced from Kaggle, collected over a period of time and portrays the historical weather conditions experienced in Seattle. It includes various attributes such as temperature, precipitation, wind speed, and weather conditions. The dataset's columns provide details about maximum and minimum temperatures, as well as precipitation and wind data. The data has been organized into a tabular format to facilitate analysis and visualization. Feel free to explore the dataset to gain insights into the relationships between different weather parameters and conditions.")

st.write("---")

# Read data from the CSV file into a Pandas DataFrame
data = pd.read_csv("Dataset.csv")

# Display the DataFrame in a tabular format using Streamlit's dataframe function
st.dataframe(data)
