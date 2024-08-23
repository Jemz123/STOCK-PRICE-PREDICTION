import pandas as pd
import numpy as np
import yfinance as yf
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.linear_model import LinearRegression
from sklearn.metrics import mean_squared_error, r2_score

# Step 1: Fetch Historical Stock Data
# You can change the stock ticker symbol and date range as needed
stock_symbol = 'AAPL'  # Example: Apple Inc.
start_date = '2020-01-01'
end_date = '2024-01-01'

print(f"Fetching data for {stock_symbol} from {start_date} to {end_date}...")
stock_data = yf.download(stock_symbol, start=start_date, end=end_date)

# Display the first few rows of the data
print("\nHistorical Stock Data:")
print(stock_data.head())

# Step 2: Prepare Data for Modeling
# Use 'Close' price as the target variable
stock_data['Target'] = stock_data['Close'].shift(-1)  # Predicting the next day's closing price

# Drop rows with NaN values (last row will have NaN target)
stock_data = stock_data.dropna()

# Features and target variable
X = stock_data[['Close']]
y = stock_data['Target']

# Split data into training and testing sets
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Step 3: Train a Linear Regression Model
model = LinearRegression()
model.fit(X_train, y_train)

# Step 4: Make Predictions
y_pred = model.predict(X_test)

# Step 5: Evaluate the Model
print("\nModel Evaluation:")
print("Mean Squared Error:", mean_squared_error(y_test, y_pred))
print("R^2 Score:", r2_score(y_test, y_pred))

# Step 6: Visualize the Results
# Plot predictions vs actual values
plt.figure(figsize=(14, 7))

# Plot actual prices
plt.plot(stock_data.index[-len(y_test):], y_test, label='Actual Prices', color='blue')

# Plot predicted prices
plt.plot(stock_data.index[-len(y_test):], y_pred, label='Predicted Prices', color='red', linestyle='dashed')

# Add labels and title
plt.title(f'Stock Price Prediction for {stock_symbol}')
plt.xlabel('Date')
plt.ylabel('Price')
plt.legend()

# Show plot
plt.show()
