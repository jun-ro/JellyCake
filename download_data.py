import yfinance as yf
import pandas as pd
from datetime import datetime, timedelta

# Download recent 1-min BTC data (max ~7 days for 1-min granularity)
print('Downloading recent 1-minute BTC-USD data...')
ticker = yf.Ticker("BTC-USD")

# Get data from last 7 days (yfinance limit for 1m; adjust period to "2wk" if needed)
end_date = datetime.now()
start_date = end_date - timedelta(days=7)

df = ticker.history(start=start_date, end=end_date, interval="1m", prepost=False, auto_adjust=True)

if df.empty:
    raise ValueError("No data downloaded - check internet or try '60m' interval")

print(f'Downloaded {len(df)} rows from {df.index[0]} to {df.index[-1]}')
print(f'Price range: ${df["Low"].min():.2f} - ${df["High"].max():.2f}')
print(f'Std dev of Close: ${df["Close"].std():.2f} (volatile!)')

# Convert to your CSV format
df = df.reset_index().rename(columns={'Datetime': 'Datetime'})  # If it has Datetime
df['Timestamp'] = (df['Datetime'].astype('int64') // 10**9).astype(float)  # Unix seconds
df = df[['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']]

# Save
output_path = 'data/recent_btc_1min.csv'
df.to_csv(output_path, index=False)
print(f'\nSaved {len(df)} rows to {output_path}')
print('Data is ready for training - expect starting loss ~0.2-0.8')