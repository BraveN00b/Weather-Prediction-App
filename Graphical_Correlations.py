# Import the necessary libraries
import streamlit as st
import plotly.express as px
import pandas as pd

# Configure the Streamlit page settings
st.set_page_config(page_title="Weather Prediction App Project", page_icon=":sun_behind_rain_cloud:")

# Read data from the CSV file into a Pandas DataFrame
data = pd.read_csv("Dataset.csv")

st.header("Graphical Correlations:")

st.write("In this section of the Weather Prediction App, we leverage the power of Plotly Express and Streamlit to visualize key relationships in our weather dataset. We use scatter plots to uncover insights about temperature, precipitation, and wind conditions. Each scatter plot offers a unique perspective, showing how different weather attributes are interconnected. The scatter plots are color-coded based on weather conditions, allowing us to easily identify patterns and trends. By presenting these visualizations, we enhance the app's ability to provide users with a comprehensive understanding of the data underlying our weather predictions.")

st.write()

# Create a scatter plot comparing temp_max and temp_min with color differentiation based on weather
plot_temp = px.scatter(
        data,
        x="temp_max",
        y="temp_min",
        color="weather",
        title="Temperature Dependence"
)

# Create a scatter plot comparing precipitation and wind with color differentiation based on weather
plot_pvw = px.scatter(
        data,
        x="precipitation",
        y="wind",
        color="weather",
        title="Precipitation VS Wind"
)

# Create a scatter plot comparing wind and temp_max with color differentiation based on weather
plot_wind_maxtemp = px.scatter(
        data,
        x="wind",
        y="temp_max",
        color="weather",
        title="Wind and Maximum Temperature"
)

# Create a scatter plot comparing wind and temp_min with color differentiation based on weather
plot_wind_mintemp = px.scatter(
        data,
        x="wind",
        y="temp_min",
        color="weather",
        title="Wind and Minimum Temperature"
)

# Create a scatter plot comparing precipitation and temp_max with color differentiation based on weather
plot_precipitation_maxtemp = px.scatter(
        data,
        x="precipitation",
        y="temp_max",
        color="weather",
        title="Precipitation and Maximum Temperature"
)

# Create a scatter plot comparing precipitation and temp_min with color differentiation based on weather
plot_precipitation_mintemp = px.scatter(
        data,
        x="precipitation",
        y="temp_min",
        color="weather",
        title="Precipitation and Minimum Temperature"
)

# Display the scatter plots using Streamlit's plotly_chart function
st.plotly_chart(plot_temp)
st.plotly_chart(plot_pvw)

st.write("---")

st.plotly_chart(plot_wind_maxtemp)
st.plotly_chart(plot_wind_mintemp)

st.write("---")

st.plotly_chart(plot_precipitation_maxtemp)
st.plotly_chart(plot_precipitation_mintemp)

st.write("---")
