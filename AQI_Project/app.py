import streamlit as st
import pandas as pd
from sklearn.ensemble import RandomForestRegressor

# Load dataset
df = pd.read_csv("aqi_dataset.csv")

# Prepare data
df = df.dropna()
df['Date'] = pd.to_datetime(df['Date'])
df['Month'] = df['Date'].dt.month
df['Day'] = df['Date'].dt.day
df = df.drop('Date', axis=1)

X = df[['PM2.5','PM10','NO2','SO2','CO','O3',
        'Temperature','Humidity','WindSpeed','Month','Day']]
y = df['AQI']

# Train model
model = RandomForestRegressor(n_estimators=100)
model.fit(X, y)

# UI
st.title("🌍 AQI Prediction System")

st.header("Enter Pollution Values")

pm25 = st.number_input("PM2.5", 0, 500, 100)
pm10 = st.number_input("PM10", 0, 500, 150)
no2 = st.number_input("NO2", 0, 200, 50)
so2 = st.number_input("SO2", 0, 100, 20)
co = st.number_input("CO", 0.0, 10.0, 1.0)
o3 = st.number_input("O3", 0, 200, 50)
temp = st.number_input("Temperature", -10, 50, 25)
humidity = st.number_input("Humidity", 0, 100, 50)
wind = st.number_input("Wind Speed", 0.0, 20.0, 5.0)

month = st.slider("Month", 1, 12, 6)
day = st.slider("Day", 1, 31, 15)

# Prediction
if st.button("Predict AQI"):
    input_data = [[pm25, pm10, no2, so2, co, o3,
                   temp, humidity, wind, month, day]]

    prediction = model.predict(input_data)

    st.success(f"Predicted AQI: {int(prediction[0])}")