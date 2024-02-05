# Import Libraries
import numpy as np
import pandas as pd
from math import sqrt
from sklearn.metrics import accuracy_score,precision_score,recall_score,f1_score,classification_report,mean_squared_error,r2_score,confusion_matrix
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split
from xgboost import XGBClassifier
import streamlit as st
from PIL import Image

# Load the dataset (assuming you have it in the same directory)
data = pd.read_csv("Dataset.csv")

# Feature Engineering
df = data.drop('date', axis=1)
X = df.iloc[:, :-1].values  # Independent variables
Y = df.iloc[:, -1].values  # Target Column

LE=LabelEncoder()
Y = LE.fit_transform(Y)
Y = Y.astype('int')

# Feature Scaling
sc1 = StandardScaler()
X = sc1.fit_transform(X)

# Splitting the data for Training and Testing
X_train, X_test, Y_train, Y_test = train_test_split(X, Y, test_size=0.3, random_state=2)

# Train the XGBoost model

xgb = XGBClassifier(n_estimators=100, max_depth=4,eta=0.1, subsample=1, colsample_bytree=1)
xgb.fit(X_train,Y_train)
Y_pred = xgb.predict(X_test)

# Streamlit UI
st.set_page_config(page_title="Weather Prediction App Project", page_icon=":sun_behind_rain_cloud:")

st.title("Weather Prediction App")

st.write("""Welcome to the Weather Prediction App! This app allows you to predict weather conditions based on specific input parameters. By providing values for precipitation, maximum and minimum temperatures, and wind speed, the app will use a machine learning model to predict the likely weather condition for that combination of inputs. Whether you're planning an outdoor activity or simply curious about the weather, this app provides you with an insightfulforecast. Just adjust the sliders below to input the desired parameters and get a prediction!""")

st.write("---")

# Enter the parameters
st.header("Input Parameters")
precipitation = st.slider("Precipitation (mm)", 0.0, 60.0, 0.0)
temp_max = st.slider("Max Temperature (°C)", -10.0, 40.0, 20.0)
temp_min = st.slider("Min Temperature (°C)", -10.0, 40.0, 10.0)
wind = st.slider("Wind Speed (km/h)", 0.0, 10.0, 2.0)

input_data = pd.DataFrame([[precipitation, temp_max, temp_min, wind]])
input_data_scaled = sc1.transform(input_data)

# Make a prediction using the XGBoost model
prediction = xgb.predict(input_data_scaled)

# Reverse label encoding to get the weather prediction
predicted_weather = LE.inverse_transform(prediction)

# Display images based on the predicted weather
image_path = "images/"  # Folder containing weather condition images

# Mapping between weather labels and image filenames
weather_image_mapping = {
    'drizzle': 'drizzle.png',
    'fog': 'fog.png',
    'rain': 'rain.png',
    'snow': 'snow.png',
    'sun': 'sun.png'
}

# Get the corresponding image filename based on predicted weather
image_filename = weather_image_mapping.get(predicted_weather[0], 'default.png')
image = Image.open(image_path + image_filename)

with st.container():
    st.write("---")
    left_column, right_column = st.columns(2)
    
    # Left column
    with left_column:
        # Print the Predicted Weather
        st.header("Predicted Weather:")
        st.markdown(f"<span style='font-size:40px'>{predicted_weather[0]}<span>", unsafe_allow_html=True)
    # Right column
    with right_column:
        # Display the image
        st.image(image, caption=f"Predicted Weather: {predicted_weather[0]}", width=200)

st.write("---")

accu = accuracy_score(Y_test, Y_pred)
st.write('Accuracy (XGB Classifier):',accu)
r2_score = r2_score(Y_test, Y_pred)
st.write("R2 score (XGB Classifier): ",r2_score )
rmse = sqrt(mean_squared_error(Y_test, Y_pred))
st.write('RMSE (XGB Classifier): ',rmse)
st.write('Precision:', precision_score(Y_test, y_pred=Y_pred,average='micro'))
st.write('Recall:',recall_score(Y_test, y_pred=Y_pred,average='micro'))
st.write('F1 Score:',f1_score(Y_test, y_pred=Y_pred,average='micro'))
