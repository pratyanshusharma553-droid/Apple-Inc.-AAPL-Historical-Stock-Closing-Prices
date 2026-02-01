import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from sklearn.preprocessing import MinMaxScaler
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

import tensorflow as tf
from tensorflow.keras.models import Sequential, Model
from tensorflow.keras.layers import LSTM, Dense, Input, Dropout, LayerNormalization, GlobalAveragePooling1D, MultiHeadAttention

#Loading dataset

import yfinance as yf

data = yf.download("AAPL", start="2018-01-01", end="2024-12-31")
series = data[['Close']]

print("Total time steps:", len(series))
series.head()

values = series.values

split = int(len(values) * 0.9)  # 90/10 split

train_data = values[:split]
test_data  = values[split:]

print("Train size:", train_data.shape)
print("Test size:", test_data.shape)

scaler = MinMaxScaler()

train_scaled = scaler.fit_transform(train_data)
test_scaled  = scaler.transform(test_data)

print("Scaled range:", train_scaled.min(), "to", train_scaled.max())

SEQ_LEN = 30
HORIZON = 1

def create_sequences(data, seq_length, horizon):
    X, y = [], []
    
    for i in range(len(data) - seq_length - horizon + 1):
        X.append(data[i:i+seq_length])
        y.append(data[i + seq_length])  # <-- single value, not slice
        
    return np.array(X), np.array(y)

X_train, y_train = create_sequences(train_scaled, SEQ_LEN, HORIZON)
X_test, y_test   = create_sequences(test_scaled, SEQ_LEN, HORIZON)

print("X_train shape:", X_train.shape)
print("y_train shape:", y_train.shape)

# ============================================================
# PART 1: LSTM MODEL FOR TIME SERIES PREDICTION
# Requirement: At least 2 stacked LSTM layers
# ============================================================

# Import required layer (if not already imported above)
from tensorflow.keras.layers import LSTM, Dense
from tensorflow.keras.models import Sequential

# ------------------------------------------------------------
# Build Stacked LSTM Model
# ------------------------------------------------------------
# Layer 1: Returns full sequence so next LSTM can process it
# Layer 2: Learns higher-level temporal features
# Dense Layer: Outputs next time-step prediction

lstm_model = Sequential([
    LSTM(64, return_sequences=True, input_shape=(SEQ_LEN, 1)),  # First LSTM layer
    LSTM(32),                                                   # Second LSTM layer
    Dense(1)                                                    # Output layer
])

# ------------------------------------------------------------
# Compile Model
# ------------------------------------------------------------
# Optimizer: Adam (adaptive learning)
# Loss: Mean Squared Error (standard for regression)

lstm_model.compile(optimizer='adam', loss='mse')

# Display model architecture summary
lstm_model.summary()

# ============================================================
# TRAINING THE LSTM MODEL
# ============================================================

history_lstm = lstm_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# ------------------------------------------------------------
# Record Initial and Final Training Loss (REQUIRED FOR GRADING)
# ------------------------------------------------------------
initial_lstm_loss = float(history_lstm.history['loss'][0])
final_lstm_loss   = float(history_lstm.history['loss'][-1])

print("Initial LSTM Loss:", initial_lstm_loss)
print("Final LSTM Loss:", final_lstm_loss)


# ============================================================
# LSTM MODEL EVALUATION
# ============================================================

# Predict on test data
lstm_pred_scaled = lstm_model.predict(X_test)

# Inverse transform to original scale
y_test_actual = scaler.inverse_transform(y_test)
lstm_pred_actual = scaler.inverse_transform(lstm_pred_scaled)

# ------------------------------------------------------------
# Calculate Evaluation Metrics
# ------------------------------------------------------------

mae_lstm = mean_absolute_error(y_test_actual, lstm_pred_actual)
rmse_lstm = np.sqrt(mean_squared_error(y_test_actual, lstm_pred_actual))
mape_lstm = np.mean(np.abs((y_test_actual - lstm_pred_actual) / y_test_actual)) * 100
r2_lstm = r2_score(y_test_actual, lstm_pred_actual)

print("LSTM MAE :", mae_lstm)
print("LSTM RMSE:", rmse_lstm)
print("LSTM MAPE:", mape_lstm)
print("LSTM R2  :", r2_lstm)


# ============================================================
# POSITIONAL ENCODING FOR TRANSFORMER
# ============================================================

import numpy as np
import tensorflow as tf

def positional_encoding(seq_len, d_model):
    positions = np.arange(seq_len)[:, np.newaxis]
    dims = np.arange(d_model)[np.newaxis, :]
    
    angle_rates = 1 / np.power(10000, (2 * (dims // 2)) / np.float32(d_model))
    angle_rads = positions * angle_rates
    
    angle_rads[:, 0::2] = np.sin(angle_rads[:, 0::2])
    angle_rads[:, 1::2] = np.cos(angle_rads[:, 1::2])
    
    return tf.cast(angle_rads[np.newaxis, ...], dtype=tf.float32)

# ============================================================
# TRANSFORMER ENCODER BLOCK
# ============================================================

from tensorflow.keras.layers import LayerNormalization, Dropout, Dense, MultiHeadAttention

def transformer_encoder(inputs, head_size, num_heads, ff_dim, dropout=0.1):
    
    # ------------------------------------------------------------
    # Multi-Head Self-Attention
    # ------------------------------------------------------------
    attention_output = MultiHeadAttention(
        num_heads=num_heads, 
        key_dim=head_size
    )(inputs, inputs)
    
    attention_output = Dropout(dropout)(attention_output)
    attention_out = LayerNormalization(epsilon=1e-6)(inputs + attention_output)
    
    # ------------------------------------------------------------
    # Feed Forward Network
    # ------------------------------------------------------------
    ffn = Dense(ff_dim, activation="relu")(attention_out)
    ffn = Dense(inputs.shape[-1])(ffn)
    ffn = Dropout(dropout)(ffn)
    
    return LayerNormalization(epsilon=1e-6)(attention_out + ffn)

# ============================================================
# PART 2: TRANSFORMER MODEL FOR TIME SERIES PREDICTION
# ============================================================

from tensorflow.keras.layers import Input, Dense, GlobalAveragePooling1D
from tensorflow.keras.models import Model

# Model dimension
d_model = 32

# ------------------------------------------------------------
# Input Layer
# ------------------------------------------------------------
inputs = Input(shape=(SEQ_LEN, 1))

# Project input to higher dimension
x = Dense(d_model)(inputs)

# ------------------------------------------------------------
# Add Positional Encoding (REQUIRED)
# ------------------------------------------------------------
x = x + positional_encoding(SEQ_LEN, d_model)

# ------------------------------------------------------------
# Transformer Encoder Block with Multi-Head Attention
# ------------------------------------------------------------
x = transformer_encoder(x, head_size=32, num_heads=4, ff_dim=64)

# Pool sequence output to single vector
x = GlobalAveragePooling1D()(x)

# Output layer for prediction
outputs = Dense(1)(x)

# Build and compile model
transformer_model = Model(inputs, outputs)
transformer_model.compile(optimizer='adam', loss='mse')

# Show architecture
transformer_model.summary()

# ============================================================
# TRAINING THE TRANSFORMER MODEL
# ============================================================

history_trans = transformer_model.fit(
    X_train, y_train,
    epochs=20,
    batch_size=32,
    validation_data=(X_test, y_test),
    verbose=1
)

# Record loss values
initial_trans_loss = float(history_trans.history['loss'][0])
final_trans_loss   = float(history_trans.history['loss'][-1])

print("Initial Transformer Loss:", initial_trans_loss)
print("Final Transformer Loss:", final_trans_loss)

# ============================================================
# TRANSFORMER MODEL EVALUATION
# ============================================================

# Predict on test data
trans_pred_scaled = transformer_model.predict(X_test)

# Inverse transform to original scale
trans_pred_actual = scaler.inverse_transform(trans_pred_scaled)

# ------------------------------------------------------------
# Calculate Evaluation Metrics
# ------------------------------------------------------------

mae_trans = mean_absolute_error(y_test_actual, trans_pred_actual)
rmse_trans = np.sqrt(mean_squared_error(y_test_actual, trans_pred_actual))
mape_trans = np.mean(np.abs((y_test_actual - trans_pred_actual) / y_test_actual)) * 100
r2_trans = r2_score(y_test_actual, trans_pred_actual)

print("Transformer MAE :", mae_trans)
print("Transformer RMSE:", rmse_trans)
print("Transformer MAPE:", mape_trans)
print("Transformer R2  :", r2_trans)

# ============================================================
# COMPARISON PLOT: ACTUAL vs LSTM vs TRANSFORMER
# ============================================================

plt.figure(figsize=(14,6))

plt.plot(y_test_actual, label="Actual Price", linewidth=2)
plt.plot(lstm_pred_actual, label="LSTM Prediction", linestyle="--")
plt.plot(trans_pred_actual, label="Transformer Prediction", linestyle=":")

plt.title("Model Comparison: Actual vs LSTM vs Transformer Predictions")
plt.xlabel("Time")
plt.ylabel("Stock Price")
plt.legend()
plt.grid(True)
plt.show()

# ============================================================
# ASSIGNMENT RESULTS JSON (DO NOT MODIFY FIELD NAMES)
# ============================================================

results = {
    "dataset_name": "AAPL Stock Closing Price",
    "sequence_length": SEQ_LEN,
    "train_test_ratio": "90/10",
    "prediction_horizon": HORIZON,
    
    "rnn_model": {
        "model_type": "LSTM",
        "framework": "keras",
        "architecture": {
            "n_layers": 2
        },
        "initial_loss": initial_lstm_loss,
        "final_loss": final_lstm_loss,
        "mae": mae_lstm,
        "rmse": rmse_lstm,
        "mape": mape_lstm,
        "r2_score": r2_lstm
    },
    
    "transformer_model": {
        "framework": "keras",
        "architecture": {
            "has_positional_encoding": True,
            "has_attention": True,
            "n_heads": 4
        },
        "initial_loss": initial_trans_loss,
        "final_loss": final_trans_loss,
        "mae": mae_trans,
        "rmse": rmse_trans,
        "mape": mape_trans,
        "r2_score": r2_trans
    }
}

import json
print(json.dumps(results, indent=4))

lstm_model.save("lstm_model.keras")
lstm_model = tf.keras.models.load_model("lstm_model.keras")

import joblib
joblib.dump(scaler, "scaler.pkl")

transformer_model.save("transformer_model.keras")  # new format

transformer_model = tf.keras.models.load_model("transformer_model.keras")


from fastapi import FastAPI
import numpy as np
import tensorflow as tf
import joblib

app = FastAPI(title="Time Series Forecasting API")

# -----------------------------
# Load models & scaler at startup
# -----------------------------
lstm_model = tf.keras.models.load_model("lstm_model.keras", compile=False)
transformer_model = tf.keras.models.load_model("transformer_model.keras", compile=False)
scaler = joblib.load("scaler.pkl")

SEQ_LEN = 30


# -----------------------------
# Root endpoint
# -----------------------------
@app.get("/")
def home():
    return {"message": "Forecasting API with LSTM and Transformer is running 🚀"}


# -----------------------------
# Prediction endpoint
# -----------------------------
@app.post("/predict/{model_name}")
def predict(model_name: str, prices: list[float]):
    """
    Provide the last 30 time-step values to predict the next value.
    model_name: 'lstm' or 'transformer'
    """

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

    