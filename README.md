# Apple-Inc.-AAPL-Historical-Stock-Closing-Prices
📈 Time Series Forecasting with LSTM and Transformer

This project explores deep learning approaches for time series forecasting by comparing a stacked LSTM (Recurrent Neural Network) with a Transformer model using self-attention. The goal is to predict future values in a financial time series by learning temporal patterns from historical data.

🚀 Project Overview

Time series data contains sequential dependencies that traditional machine learning models often struggle to capture. This project demonstrates how modern deep learning architectures can model these patterns effectively.

Two architectures were implemented and evaluated:

LSTM Model – Learns temporal dependencies step-by-step using recurrent memory

Transformer Model – Uses multi-head self-attention and positional encoding to analyze the entire sequence at once

⚙️ Workflow

The project follows a complete end-to-end pipeline:

Data Collection & Visualization
Historical financial time series data was used and visualized to understand trends.

Preprocessing

Temporal train–test split (no shuffling)

Data normalization

Sliding window sequence creation

Model Development

Stacked LSTM with multiple recurrent layers

Transformer encoder with positional encoding and multi-head attention

Training
Both models were trained while tracking loss convergence.

Evaluation
Performance was measured using:

MAE (Mean Absolute Error)

RMSE (Root Mean Squared Error)

MAPE (Mean Absolute Percentage Error)

R² Score

Visualization
Plots comparing actual vs predicted values were generated for both models.

<img width="1160" height="545" alt="image" src="https://github.com/user-attachments/assets/3a2c4c44-4246-4413-a4f6-6234bf4dafc5" />


📊 Results

Both models successfully learned meaningful temporal patterns. However, the Transformer consistently achieved lower prediction errors and a higher R² score. Its attention mechanism allowed it to better capture long-term dependencies and important historical signals.

🧠 Key Takeaways

LSTMs are strong baselines for sequential modeling

Attention mechanisms allow models to focus on the most relevant past information

Transformers are powerful alternatives to recurrent networks for time series forecasting

🛠 Tech Stack

Python

TensorFlow / Keras

NumPy, Pandas

Matplotlib

Scikit-learn

📌 Future Improvements

Try multivariate time series inputs

Experiment with deeper Transformer architectures

Add hyperparameter tuning

Deploy as a real-time forecasting API
