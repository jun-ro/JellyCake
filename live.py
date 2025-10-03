import torch
import torch.nn as nn
from torch.amp import autocast
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
import time
from datetime import datetime
from collections import deque
from tqdm import tqdm
from model import GRUModel

# Paths
MODEL_PATH = "model/model.pth"
SCALER_PATH = "model/scaler.pkl"
SEQ_LENGTH = 60

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

class LiveBTCGRUPredictor:
    def __init__(self, model_path, scaler_path, seq_length):
        checkpoint = torch.load(model_path, map_location=device)
        self.model = GRUModel(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f'Loaded model from {model_path}')
        
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f'Loaded scaler from {scaler_path}')
        
        self.seq_length = seq_length
        self.history = deque(maxlen=seq_length * 2)
        
        print('Live predictor ready!')

    def fetch_history(self, period='5d', interval='1m'):
        """Fetch recent historical data"""
        try:
            ticker = yf.Ticker("BTC-USD")
            df = ticker.history(period=period, interval=interval, prepost=False)
            
            if df.empty:
                raise ValueError("No data from yfinance")
            
            # Keep datetime index and OHLCV columns
            df = df[['Open', 'High', 'Low', 'Close', 'Volume']]
            
            # Print date range using index directly
            min_date = df.index[0].strftime('%Y-%m-%d %H:%M:%S')
            max_date = df.index[-1].strftime('%Y-%m-%d %H:%M:%S')
            print(f'Fetched {len(df)} {interval} candles from {min_date} to {max_date}')
            
            # Take last seq_length * 2 rows
            df = df.tail(self.seq_length * 2)
            
            # Build history deque
            self.history = deque([{
                'datetime': idx.strftime('%Y-%m-%d %H:%M:%S'),
                'open': row['Open'],
                'high': row['High'],
                'low': row['Low'],
                'close': row['Close'],
                'volume': row['Volume']
            } for idx, row in df.iterrows()], maxlen=self.seq_length * 2)
            
            return df.tail(self.seq_length)
        
        except Exception as e:
            print(f'Error fetching history: {e}')
            if interval == '1m':
                return self.fetch_history(period=period, interval='5m')
            return pd.DataFrame()
    
    def update_history(self):
        """Fetch and append the latest 1-min candle"""
        try:
            ticker = yf.Ticker("BTC-USD")
            df = ticker.history(period='1d', interval='1m', prepost=False)
            if df.empty:
                return False
            
            # Get last row
            latest_idx = df.index[-1]
            latest = df.iloc[-1]
            
            new_candle = {
                'datetime': latest_idx.strftime('%Y-%m-%d %H:%M:%S'),
                'open': latest['Open'],
                'high': latest['High'],
                'low': latest['Low'],
                'close': latest['Close'],
                'volume': latest['Volume']
            }
            
            # Append to history
            self.history.append(new_candle)
            
            print(f'Updated: {new_candle["datetime"]} | Close ${new_candle["close"]:.2f} | Vol {new_candle["volume"]:.0f}')
            return True
            
        except Exception as e:
            print(f'Update error: {e}')
            return False

    def prepare_sequence(self):
        """Prepare latest history as scaled tensor"""
        if len(self.history) < self.seq_length:
            print('History too short—fetching more...')
            return None
        
        recent = list(self.history)[-self.seq_length:]
        seq_array = np.array([[
            c['open'], c['high'], c['low'], c['close'], c['volume']
        ] for c in recent])
        
        seq_scaled = self.scaler.transform(seq_array)
        seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
        
        return seq_tensor
    
    def predict_next_close(self):
        """Predict next closing price (raw delta model)"""
        seq_tensor = self.prepare_sequence()
        if seq_tensor is None:
            return None
        
        current_close = self.history[-1]['close']
        
        with torch.no_grad():
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                # Model outputs raw delta (NOT scaled)
                pred_delta = self.model(seq_tensor).cpu().numpy()[0, 0]
        
        # Simply add delta to current price
        pred_price = current_close + pred_delta
        
        return float(pred_price)
    
    def run_live(self, poll_interval=60, num_predictions=10):
        """Live loop"""
        history_df = self.fetch_history(period='5d', interval='1m')
        if history_df.empty:
            print('Failed initial fetch—exiting.')
            return
        
        pred = self.predict_next_close()
        if pred is not None:
            last_close = self.history[-1]['close']
            change_pct = ((pred - last_close) / last_close * 100)
            print(f'\nInitial prediction: Next close ${pred:.2f} (change: {change_pct:+.2f}%)')
        
        print(f'\nLive mode started! Polling every {poll_interval}s.')
        print('Time | Last Close | Predicted Next | Change %')
        print('-' * 60)
        
        prediction_count = 0
        while prediction_count < num_predictions:
            now = datetime.now()
            
            if self.update_history():
                pred = self.predict_next_close()
                if pred is not None:
                    last_close = self.history[-1]['close']
                    change_pct = ((pred - last_close) / last_close * 100)
                    print(f'{now.strftime("%H:%M:%S")} | ${last_close:.2f} | ${pred:.2f} | {change_pct:+.2f}%')
                    prediction_count += 1
            
            time.sleep(poll_interval)
        
        print('Live session complete!')

if __name__ == '__main__':
    predictor = LiveBTCGRUPredictor(MODEL_PATH, SCALER_PATH, SEQ_LENGTH)
    predictor.run_live(poll_interval=60, num_predictions=10)