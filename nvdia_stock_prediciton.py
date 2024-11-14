import yfinance as yf
import pandas as pd
import numpy as np
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LinearRegression
import matplotlib.pyplot as plt
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import LSTM, Dense, Dropout
from sklearn.metrics import mean_squared_error, mean_absolute_error

# NVIDIA Stock Data
ticker_symbol = "NVDA"
nvidia = yf.Ticker(ticker_symbol)
data = nvidia.history(period="5y")
data = data[['Close', 'Volume']]

# Technical indicators
data['MA20'] = data['Close'].rolling(window=20).mean()
data['MA50'] = data['Close'].rolling(window=50).mean()
delta = data['Close'].diff(1)
gain = delta.where(delta > 0, 0)
loss = -delta.where(delta < 0, 0)
average_gain = gain.rolling(window=14).mean()
average_loss = loss.rolling(window=14).mean()
rs = average_gain / average_loss
data['RSI'] = 100 - (100 / (1 + rs))
data['EMA12'] = data['Close'].ewm(span=12, adjust=False).mean()
data['EMA26'] = data['Close'].ewm(span=26, adjust=False).mean()
data['MACD'] = data['EMA12'] - data['EMA26']
data['Signal_Line'] = data['MACD'].ewm(span=9, adjust=False).mean()

# Target column
data['Target'] = data['Close'].shift(-1)
data.dropna(inplace=True)

# Data for LSTM
scaler = MinMaxScaler()
scaled_data = scaler.fit_transform(data[['Close', 'Volume', 'MA20', 'MA50', 'RSI', 'MACD', 'Signal_Line', 'Target']])

X = scaled_data[:, :-1]
y = scaled_data[:, -1]

# Reshaping X for LSTM input (samples, timesteps, features)
X_reshaped = X.reshape((X.shape[0], 1, X.shape[1]))

# Splitting data into training and test sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, shuffle=False)

# Training linear regression model
lr_model = LinearRegression()
lr_model.fit(X_train, y_train)
y_pred_lr_scaled = lr_model.predict(X_test)

# Linear regression predictions
y_pred_lr_scaled = y_pred_lr_scaled.reshape(-1, 1)
y_test = y_test.reshape(-1, 1)
y_test_original = scaler.inverse_transform(np.concatenate((X_test, y_test), axis=1))[:, -1]
y_pred_lr_original = scaler.inverse_transform(np.concatenate((X_test, y_pred_lr_scaled), axis=1))[:, -1]

# Training LSTM
lstm_model = Sequential()
lstm_model.add(LSTM(units=50, return_sequences=True, input_shape=(X_reshaped.shape[1], X_reshaped.shape[2])))
lstm_model.add(Dropout(0.2))
lstm_model.add(LSTM(units=50, return_sequences=False))
lstm_model.add(Dropout(0.2))
lstm_model.add(Dense(units=1))

lstm_model.compile(optimizer='adam', loss='mean_squared_error')
lstm_model.fit(X_train.reshape(X_train.shape[0], 1, X_train.shape[1]), y_train, epochs=50, batch_size=32, validation_data=(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]), y_test), verbose=1)

# LSTM Predictions
y_pred_lstm_scaled = lstm_model.predict(X_test.reshape(X_test.shape[0], 1, X_test.shape[1]))

# LSTM predictions
y_pred_lstm_scaled = y_pred_lstm_scaled.reshape(-1, 1)
y_pred_lstm_original = scaler.inverse_transform(np.concatenate((X_test, y_pred_lstm_scaled), axis=1))[:, -1]

# Plotting actual vs linear regression and LSTM predictions
plt.figure(figsize=(14, 7))
plt.plot(y_test_original, label='Actual Prices', color='blue')
plt.plot(y_pred_lr_original, label='Linear Regression Predictions', color='orange')
plt.plot(y_pred_lstm_original, label='LSTM Predictions', color='red')
plt.title("Actual vs Predicted Closing Prices (Linear Regression vs LSTM)")
plt.xlabel("Days")
plt.ylabel("Closing Price")
plt.legend()
plt.savefig("NVIDIA_Actual_vs_Predicted_LSTM_and_LR.png")
plt.show()

# Printing mae and rmse
rmse_lstm = np.sqrt(mean_squared_error(y_test_original, y_pred_lstm_original))
mae_lstm = mean_absolute_error(y_test_original, y_pred_lstm_original)
print(f"LSTM Model - RMSE: {rmse_lstm}")
print(f"LSTM Model - MAE: {mae_lstm}")

rmse_lr = np.sqrt(mean_squared_error(y_test_original, y_pred_lr_original))
mae_lr = mean_absolute_error(y_test_original, y_pred_lr_original)
print(f"Linear Regression Model - RMSE: {rmse_lr}")
print(f"Linear Regression Model - MAE: {mae_lr}")

# Saving metrics to excel file
metrics = {
    "Model": ["Linear Regression", "LSTM"],
    "RMSE": [rmse_lr, rmse_lstm],
    "MAE": [mae_lr, mae_lstm]
}

metrics_df = pd.DataFrame(metrics)
metrics_df.to_excel("NVIDIA_Price_Prediction_KPIs.xlsx", index=False)
print("Metrics saved to NVIDIA_Price_Prediction_KPIs.xlsx")
