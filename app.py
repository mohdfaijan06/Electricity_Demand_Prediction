import joblib
import pandas as pd
import streamlit as st

from datetime import datetime


st.title("Electricity Demand Prediction")

# Date input
date_input = st.date_input("Select Date", datetime.today())

# Hour input (0 to 23)
hour_input = st.number_input("Hour (0–23)", min_value=0, max_value=23, step=1)

# Temperature input
temperature_input = st.number_input("Temperature (°C) upto 55 °C", format="%.1f")

# Humidity input
humidity_input = st.number_input("Humidity (%) upto 100 %", format="%.1f")

# Model selection
model_options = ["LinearRegression", "DecisionTreeRegressor", "RandomForestRegressor", "GradientBoostingRegressor"]
model_input = st.selectbox("Select Model", model_options)

def convert_to_model_input(temperature_input, humidity_input, date_input, hour_input):
    # Get day name from date
    dayofweek = date_input.strftime("%a")  # Mon, Tue, Wed, etc.
    hour_input = int(hour_input)
    
    # Extract components
    year = date_input.year
    month = date_input.month
    date = date_input.day
    
    d = {
        "Temperature": [temperature_input],
        "Humidity": [humidity_input],
        "hour": [hour_input],
        "dayofweek": [dayofweek],
        "month": [month],
        "year": [year],
        "date": [date]
    }
    
    print(d)
    df = pd.DataFrame(d)
    categorical_columns = ['hour', 'dayofweek', 'month', 'year', 'date']
    df_input = pd.get_dummies(df, columns = categorical_columns, drop_first=True)
    
    feature_columns = joblib.load("columns/feature_columns.joblib")
    
    # Reindex to match training features, fill missing with 0
    df_input = df_input.reindex(columns=feature_columns, fill_value=0)
    
    # Load a scaler from preprocess.py
    scaler = joblib.load("scaler/MinMaxScaler.joblib")
    # df_input is unseen data and we dont know about it at present. so we only use .transform() in df_input, since it has already been fit on train data in preprocess.py
    print(list(df_input.columns))
    df_input_scaled = scaler.transform(df_input) # returns numpy array
    
    return df_input_scaled

def load_model_to_predict(model_input, df_input_scaled):
    model = joblib.load(f"models/{model_input}.joblib")
    pred = model.predict(df_input_scaled)
    
    if model_input == "LinearRegression":
        pred = pred[0][0]
    else:
        pred = pred[0]
    
        
    return pred if pred >= 0 else 0
    
# Submit button
if st.button("Submit"):
    # Validation: Check if all fields are filled
    if date_input and hour_input is not None and temperature_input != 0.0 and humidity_input != 0.0:
        
        with st.spinner("Predicting..."):
            df_input_scaled = convert_to_model_input(temperature_input, humidity_input, date_input, hour_input)
            pred = load_model_to_predict(model_input, df_input_scaled)
        
        # st.success(f"""**Data Captured ✅**  
        # **Date:** {date_input}  
        # **Hour:** {hour_input}  
        # **Temperature:** {temperature_input} °C  
        # **Humidity:** {humidity_input} %  
        # **Selected Model:** {model_input}  
        # **Prediction:** {pred:.2f} units""")
        
        st.metric(label="Predicted Demand (in kWh)", value=f"{pred:.2f}")
    else:
        st.error("⚠ Please fill in all fields before submitting.")
