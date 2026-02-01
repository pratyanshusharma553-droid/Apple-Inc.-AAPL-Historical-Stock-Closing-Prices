from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI(title="Time Series Forecasting API")

# Load trained models and scaler
lstm_model = tf.keras.models.load_model("lstm_model.keras", compile=False)
transformer_model = tf.keras.models.load_model("transformer_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

SEQ_LEN = 30


@app.get("/")
def home():
    return {"message": "Forecasting API is live 🚀"}


@app.post("/predict/{model_name}")
def predict(model_name: str, prices: list[float]):

    if len(prices) != SEQ_LEN:
        return {"error": f"Exactly {SEQ_LEN} values required"}

    data = np.array(prices).reshape(-1, 1)
    data_scaled = scaler.transform(data)
    X = data_scaled.reshape(1, SEQ_LEN, 1)

    if model_name.lower() == "lstm":
        pred_scaled = lstm_model.predict(X)
    elif model_name.lower() == "transformer":
        pred_scaled = transformer_model.predict(X)
    else:
        return {"error": "Model must be 'lstm' or 'transformer'"}

    pred = scaler.inverse_transform(pred_scaled)

    return {
        "model_used": model_name.lower(),
        "predicted_next_value": float(pred[0][0])
    }
