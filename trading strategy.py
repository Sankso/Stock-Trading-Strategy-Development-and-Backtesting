import pandas as pd
import numpy as np
import requests

# Function to compute EMA
def compute_ema(df, span):
    df[f'EMA_{span}'] = df['close'].ewm(span=span, adjust=False).mean()
    return df

# Function to compute RSI
def compute_rsi(df, window=14):
    delta = df['close'].diff()
    gain = (delta.where(delta > 0, 0)).rolling(window=window).mean()
    loss = (-delta.where(delta < 0, 0)).rolling(window=window).mean()
    rs = gain / loss
    df['RSI'] = 100 - (100 / (1 + rs))
    return df

# Function to compute Bollinger Bands
def compute_bollinger_bands(df, window=20, num_std_dev=2):
    df['MA'] = df['close'].rolling(window=window).mean()
    df['Upper Band'] = df['MA'] + (df['close'].rolling(window=window).std() * num_std_dev)
    df['Lower Band'] = df['MA'] - (df['close'].rolling(window=window).std() * num_std_dev)
    return df

# Function to fetch stock data
def fetch_stock_data(symbol, api_key):
    url = f'https://www.alphavantage.co/query?function=TIME_SERIES_DAILY&symbol={symbol}&apikey={api_key}'
    response = requests.get(url)
    data = response.json()
    time_series = data['Time Series (Daily)']
    df = pd.DataFrame.from_dict(time_series, orient='index')
    df = df.astype(float)
    df.index = pd.to_datetime(df.index)
    df.columns = ['open', 'high', 'low', 'close', 'volume']
    df = df.sort_index()
    return df

# Function to generate signals
def generate_signals(df):
    df['signal'] = 0
    buy_signal = (df['RSI'] < 30) | (df['close'] < df['Lower Band'])  # Using Bollinger Bands
    sell_signal = (df['RSI'] > 70) | (df['close'] > df['Upper Band'])
    df.loc[buy_signal, 'signal'] = 1
    df.loc[sell_signal, 'signal'] = -1
    return df

# Backtesting strategy
def backtest_strategy(df, initial_capital=10000):
    df['position'] = df['signal'].shift()
    df['daily_return'] = df['close'].pct_change()
    df['strategy_return'] = df['daily_return'] * df['position']
    df['cumulative_return'] = (1 + df['strategy_return']).cumprod()
    df['cumulative_strategy_return'] = initial_capital * df['cumulative_return']
    return df

# Main execution
api_key = '97K72T1VE9OD7Y5L'  # Replace with your Alpha Vantage API key
symbol = 'AAPL'  # You can try different stock symbols
df = fetch_stock_data(symbol, api_key)

# Compute indicators
df = compute_ema(df, 50)  # Calculate 50-period EMA
df = compute_rsi(df)
df = compute_bollinger_bands(df)  # Calculate Bollinger Bands

# Generate trading signals
df = generate_signals(df)

# Backtest strategy
df = backtest_strategy(df)

# Calculate key performance metrics
initial_capital = 10000
cumulative_return = df['cumulative_strategy_return'].iloc[-1] - initial_capital
annual_return = (df['strategy_return'].mean() * 252) * 100
annual_volatility = (df['strategy_return'].std() * (252**0.5)) * 100
sharpe_ratio = annual_return / annual_volatility if annual_volatility != 0 else np.nan

# Output results
print(f"Buy signals: {df['signal'].value_counts().get(1, 0)}")
print(f"Sell signals: {df['signal'].value_counts().get(-1, 0)}")
print(f"Cumulative Return: ${cumulative_return:.2f}")
print(f"Annual Return: {annual_return:.2f}%")
print(f"Annual Volatility: {annual_volatility:.2f}%")
print(f"Sharpe Ratio: {sharpe_ratio:.2f}")
