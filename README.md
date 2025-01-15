# Stock-Price-Prediction-using-Linear-Regression-Model

import numpy as np
import pandas as pd
import plotly.graph_objects as go
from autots import AutoTS

# Specify the path to your Excel file
file_path = 'C:\\Users\\anand\\OneDrive\\Desktop\\AI ML\\AAPL.csv'

# Read the CSV file into a DataFrame
data = pd.read_csv(file_path)

# Convert 'Date' column to datetime format
data['Date'] = pd.to_datetime(data['Date'])

# Display the first few rows of the DataFrame
# print("Original DataFrame:")
# print(data.head())

figure = go.Figure(data=[go.Candlestick(x=data["Date"],
                                        open=data["Open"], high=data["High"],
                                        low=data["Low"], close=data["Close"])])
figure.update_layout(title = "Apple Stock Price Analysis", xaxis_rangeslider_visible=False)
figure.show()

# Print the correlation matrix
# print(data.corr())

# Initialize and fit AutoTS model
model = AutoTS(forecast_length=5, frequency='infer', ensemble='simple')
model = model.fit(data, date_col='Date', value_col='Close', id_col=None)

# Make predictions
prediction = model.predict()
forecast = prediction.forecast

# Print the forecasted values
print("Forecasted values:")
# print(forecast)
