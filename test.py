import torch
import torch.nn as nn
from torch.amp import autocast
import yfinance as yf
import pandas as pd
import numpy as np
import pickle
from model import GRUModel

# Paths
MODEL_PATH = "model/model.pth"
SCALER_PATH = "model/scaler.pkl"
SEQ_LENGTH = 60

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

class ModelTester:
    def __init__(self, model_path, scaler_path, seq_length):
        # Load model
        checkpoint = torch.load(model_path, map_location=device)
        self.model = GRUModel(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )
        self.model.load_state_dict(checkpoint['model_state_dict'])
        self.model.to(device)
        self.model.eval()
        print(f'✓ Loaded model from {model_path}')
        
        # Load scaler
        with open(scaler_path, 'rb') as f:
            self.scaler = pickle.load(f)
        print(f'✓ Loaded scaler from {scaler_path}')
        
        self.seq_length = seq_length
        self.history = None
        
    def fetch_data(self):
        """Fetch recent BTC data"""
        print('\nFetching live data...')
        ticker = yf.Ticker("BTC-USD")
        df = ticker.history(period='5d', interval='1m', prepost=False)
        
        if df.empty:
            raise ValueError("Failed to fetch data")
        
        df = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(self.seq_length)
        
        print(f'✓ Fetched {len(df)} candles')
        print(f'  Date range: {df.index[0]} to {df.index[-1]}')
        print(f'  Last close: ${df["Close"].iloc[-1]:.2f}')
        
        self.history = df
        return df
    
    def test_scaler_info(self):
        """Display scaler information"""
        print('\n' + '='*60)
        print('SCALER INFORMATION')
        print('='*60)
        
        print(f'Scaler type: {type(self.scaler).__name__}')
        
        if hasattr(self.scaler, 'mean_'):
            print(f'\nStandardScaler detected')
            print(f'  Close mean: {self.scaler.mean_[3]:.2f}')
            print(f'  Close std: {self.scaler.scale_[3]:.2f}')
        elif hasattr(self.scaler, 'min_'):
            print(f'\nMinMaxScaler detected')
            print(f'  Close min: {self.scaler.data_min_[3]:.2f}')
            print(f'  Close max: {self.scaler.data_max_[3]:.2f}')
            print(f'  Close range: {self.scaler.data_max_[3] - self.scaler.data_min_[3]:.2f}')
            print(f'  Scale factor: {self.scaler.scale_[3]:.6f}')
            print(f'  Min offset: {self.scaler.min_[3]:.6f}')
        
        # Test scaling
        print(f'\nScaling test (using Close column):')
        test_values = [100, 1000, 10000, 100000]
        for val in test_values:
            test_arr = np.zeros((1, 5))
            test_arr[0, 3] = val
            scaled = self.scaler.transform(test_arr)[0, 3]
            unscaled = self.scaler.inverse_transform(
                np.array([[0, 0, 0, scaled, 0]])
            )[0, 3]
            print(f'  ${val:>6.0f} -> {scaled:>8.6f} -> ${unscaled:>6.0f}')
    
    def test_model_output(self):
        """Test what the model actually outputs"""
        if self.history is None:
            self.fetch_data()
        
        print('\n' + '='*60)
        print('MODEL OUTPUT DIAGNOSIS')
        print('='*60)
        
        # Prepare input
        seq_array = self.history[['Open', 'High', 'Low', 'Close', 'Volume']].values
        seq_scaled = self.scaler.transform(seq_array)
        seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
        
        current_close = self.history['Close'].iloc[-1]
        
        # Get model output
        with torch.no_grad():
            with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                raw_output = self.model(seq_tensor).cpu().numpy()[0, 0]
        
        print(f'\nCurrent close: ${current_close:.2f}')
        print(f'Raw model output: {raw_output:.6f}')
        print(f'\nTesting 4 interpretation methods:\n')
        
        # Method 1: Absolute price
        dummy = np.zeros((1, 5))
        dummy[0, 3] = raw_output
        pred_abs = self.scaler.inverse_transform(dummy)[0, 3]
        change_abs = ((pred_abs - current_close) / current_close * 100)
        print(f'1. Absolute Price Interpretation:')
        print(f'   Prediction: ${pred_abs:.2f}')
        print(f'   Change: {change_abs:+.2f}%')
        print(f'   ✓ REASONABLE' if abs(change_abs) < 5 else '   ✗ UNREALISTIC')
        
        # Method 2: Delta (range scaling only)
        if hasattr(self.scaler, 'min_'):  # MinMaxScaler
            close_range = 1.0 / self.scaler.scale_[3]
            delta_range = raw_output * close_range
            pred_range = current_close + delta_range
            change_range = (delta_range / current_close * 100)
            print(f'\n2. Delta (Range Scaling Only):')
            print(f'   Delta: ${delta_range:+.2f}')
            print(f'   Prediction: ${pred_range:.2f}')
            print(f'   Change: {change_range:+.2f}%')
            print(f'   ✓ REASONABLE' if abs(change_range) < 5 else '   ✗ UNREALISTIC')
        
        # Method 3: Delta (full inverse transform)
        if hasattr(self.scaler, 'min_'):  # MinMaxScaler
            dummy_delta = np.zeros((1, 5))
            dummy_delta[0, 3] = raw_output
            delta_full = self.scaler.inverse_transform(dummy_delta)[0, 3] - self.scaler.data_min_[3]
            pred_full = current_close + delta_full
            change_full = (delta_full / current_close * 100)
            print(f'\n3. Delta (Full Inverse Transform):')
            print(f'   Delta: ${delta_full:+.2f}')
            print(f'   Prediction: ${pred_full:.2f}')
            print(f'   Change: {change_full:+.2f}%')
            print(f'   ✓ REASONABLE' if abs(change_full) < 5 else '   ✗ UNREALISTIC')
        
        # Method 4: Raw output as delta
        pred_raw = current_close + raw_output
        change_raw = (raw_output / current_close * 100)
        print(f'\n4. Raw Output as Delta (No Scaling):')
        print(f'   Delta: ${raw_output:+.2f}')
        print(f'   Prediction: ${pred_raw:.2f}')
        print(f'   Change: {change_raw:+.2f}%')
        print(f'   ✓ REASONABLE' if abs(change_raw) < 5 else '   ✗ UNREALISTIC')
        
        print('\n' + '='*60)
        print('RECOMMENDATION:')
        print('='*60)
        methods = [
            ('Absolute Price', pred_abs, change_abs),
            ('Delta (Range)', pred_range if hasattr(self.scaler, 'min_') else None, 
             change_range if hasattr(self.scaler, 'min_') else 999),
            ('Delta (Full)', pred_full if hasattr(self.scaler, 'min_') else None,
             change_full if hasattr(self.scaler, 'min_') else 999),
            ('Raw Delta', pred_raw, change_raw)
        ]
        
        reasonable = [(name, pred, chg) for name, pred, chg in methods 
                      if pred is not None and abs(chg) < 5]
        
        if reasonable:
            best = min(reasonable, key=lambda x: abs(x[2]))
            print(f'✓ Use Method: {best[0]}')
            print(f'  Prediction: ${best[1]:.2f} ({best[2]:+.2f}%)')
        else:
            print('✗ No method produces reasonable results!')
            print('  Your model may need retraining.')
    
    def test_multiple_predictions(self, n=5):
        """Make multiple predictions to check consistency"""
        if self.history is None:
            self.fetch_data()
        
        print('\n' + '='*60)
        print(f'CONSISTENCY TEST ({n} predictions)')
        print('='*60)
        
        predictions = []
        for i in range(n):
            seq_array = self.history[['Open', 'High', 'Low', 'Close', 'Volume']].values
            seq_scaled = self.scaler.transform(seq_array)
            seq_tensor = torch.FloatTensor(seq_scaled).unsqueeze(0).to(device)
            
            with torch.no_grad():
                with autocast('cuda' if device.type == 'cuda' else 'cpu'):
                    raw_output = self.model(seq_tensor).cpu().numpy()[0, 0]
            
            predictions.append(raw_output)
        
        print(f'Predictions (raw): {predictions}')
        print(f'Mean: {np.mean(predictions):.6f}')
        print(f'Std: {np.std(predictions):.6f}')
        print(f'Range: {np.max(predictions) - np.min(predictions):.6f}')
        
        if np.std(predictions) < 1e-6:
            print('✓ Deterministic (good)')
        else:
            print('✗ Non-deterministic (dropout may be enabled)')

def run_all_tests():
    """Run complete test suite"""
    print('='*60)
    print('MODEL DIAGNOSTIC TEST SUITE')
    print('='*60)
    
    tester = ModelTester(MODEL_PATH, SCALER_PATH, SEQ_LENGTH)
    
    # Test 1: Scaler info
    tester.test_scaler_info()
    
    # Test 2: Fetch data
    tester.fetch_data()
    
    # Test 3: Model output diagnosis
    tester.test_model_output()
    
    # Test 4: Consistency check
    tester.test_multiple_predictions(n=5)
    
    print('\n' + '='*60)
    print('TESTS COMPLETE')
    print('='*60)

if __name__ == '__main__':
    run_all_tests()