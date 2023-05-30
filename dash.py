import pandas as pd
import yfinance as yf

# Step 1: Gather Data

# Download historical data as pandas DataFrame
data = yf.download('AAPL','2020-01-01','2023-12-31')

# Step 2: Feature Engineering

data['Rolling'] = data['Close'].rolling(window=5).mean()
