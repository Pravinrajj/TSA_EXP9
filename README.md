# EX.NO.09        A project on Time series analysis on weather forecasting using ARIMA model 
### Date: 

### AIM:
To Create a project on Time series analysis on XAUUSD forecasting using ARIMA model in  Python and compare with other models.
### ALGORITHM:
1. Explore the dataset of XAUUSD 
2. Check for stationarity of time series time series plot
   ACF plot and PACF plot
   ADF test
   Transform to stationary: differencing
3. Determine ARIMA models parameters p, q
4. Fit the ARIMA model
5. Make time series predictions
6. Auto-fit the ARIMA model
7. Evaluate model predictions
### PROGRAM:
```py
import pandas as pd
import matplotlib.pyplot as plt
from pmdarima import auto_arima
from sklearn.metrics import mean_squared_error
import numpy as np

# Load and preprocess data
data = pd.read_csv('XAUUSD_2010-2023.csv')
data['time'] = pd.to_datetime(data['time'], format='%d-%m-%Y %H:%M')
data.set_index('time', inplace=True)
close_prices = data['close']

# Plot the closing prices
plt.figure(figsize=(10, 5))
plt.plot(close_prices, label="Closing Prices")
plt.title("XAUUSD Closing Prices Over Time")
plt.xlabel("Date")
plt.ylabel("Close Price")
plt.legend()
plt.show()

# Use auto_arima to find the best ARIMA parameters and fit the model
model = auto_arima(close_prices, seasonal=False, trace=True, error_action='ignore', suppress_warnings=True)
print(f"Best ARIMA Model Order: {model.order}")

# Split data into train and test sets
train_size = int(len(close_prices) * 0.8)
train, test = close_prices[:train_size], close_prices[train_size:]

# Fit the model on the training data
model.fit(train)

# Forecast future values
forecast = model.predict(n_periods=len(test))

# Plot actual vs forecasted values
plt.figure(figsize=(10, 5))
plt.plot(test, label="Actual Closing Prices")
plt.plot(test.index, forecast, label="Forecasted Prices", color='orange')
plt.title("Actual vs Forecasted Closing Prices")
plt.legend()
plt.show()

# Evaluate model accuracy
mse = mean_squared_error(test, forecast)
rmse = np.sqrt(mse)
print(f"Mean Squared Error: {mse}")
print(f"Root Mean Squared Error: {rmse}")
```
### OUTPUT:
![image](https://github.com/user-attachments/assets/ffad4f2a-2ee0-4d52-ab7f-e142d3a976c9)

### RESULT:
Thus the program run successfully based on the ARIMA model using python.
