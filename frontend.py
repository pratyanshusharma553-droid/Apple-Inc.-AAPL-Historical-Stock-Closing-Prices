import streamlit as st
import requests

API_URL = "https://apple-inc-aapl-historical-stock-closing.onrender.com/predict"

st.set_page_config(page_title="Time Series Forecast", layout="centered")

st.title("📈 Time Series Forecasting")
st.write("Enter the last 30 prices to predict the next value.")

model_choice = st.selectbox("Choose Model", ["transformer", "lstm"])

prices_input = st.text_area(
    "Enter 30 values separated by commas",
    "182.1,183.2,181.5,180.9,182.3,184.0,183.7,185.2,186.1,187.0,"
    "188.2,189.3,190.1,191.4,192.0,193.2,194.1,195.5,196.0,197.3,"
    "198.2,199.1,200.0,201.5,202.1,203.2,204.0,205.3,206.2,207.1"
)

if st.button("Predict"):
    try:
        prices = [float(x.strip()) for x in prices_input.split(",")]

        if len(prices) != 30:
            st.error("Please enter exactly 30 values.")
        else:
            response = requests.post(f"{API_URL}/{model_choice}", json=prices)

            if response.status_code == 200:
                result = response.json()
                st.success(f"Predicted Next Value: {result['predicted_next_value']:.2f}")
            else:
                st.error(response.json())

    except:
        st.error("Invalid input. Please enter numeric values only.")
