import time
import requests
import torch
import numpy as np
import pandas as pd
import pickle
import signal
import sys
import os
from datetime import datetime
import matplotlib.pyplot as plt
from model import GRUModel

# === Config ===
SEQ_LEN = 60
RETRY_DELAY = 5
MAX_RETRIES = 5
POLL_INTERVAL = 10  # seconds to check for new candle
ACCURACY_TRACK_SIZE = 100  # number of predictions to track for accuracy
OUTPUT_DIR = "performance_results"

# === Global variables for handling interruption ===
running = True
prediction_data = []

# === Signal handler for Ctrl+C ===
def signal_handler(sig, frame):
    global running
    print("\nCtrl+C detected. Generating performance report...")
    running = False
    generate_performance_report()
    sys.exit(0)

# === Register the signal handler ===
signal.signal(signal.SIGINT, signal_handler)

# === Load trained model + scalers ===
def load_model():
    try:
        checkpoint = torch.load("model/upgraded_model.pth", map_location="cpu")
        model = GRUModel(
            input_size=checkpoint['input_size'],
            hidden_size=checkpoint['hidden_size'],
            num_layers=checkpoint['num_layers']
        )
        model.load_state_dict(checkpoint['model_state_dict'])
        model.eval()
        
        with open("model/scaler.pkl", "rb") as f:
            scaler = pickle.load(f)
        with open("model/y_scaler.pkl", "rb") as f:
            y_scaler = pickle.load(f)
            
        return model, scaler, y_scaler
    except Exception as e:
        print(f"Error loading model or scalers: {e}")
        return None, None, None

# === Fetch BTC minute OHLCV from CryptoCompare ===
def fetch_minute_ohlcv(minutes=SEQ_LEN + 1):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {"fsym": "BTC", "tsym": "USD", "limit": minutes}
    
    for attempt in range(1, MAX_RETRIES + 1):
        try:
            resp = requests.get(url, params=params, timeout=10)
            resp.raise_for_status()
            data = resp.json()
            
            if "Data" not in data or "Data" not in data["Data"]:
                raise ValueError("No OHLCV data in CryptoCompare response")

            df = pd.DataFrame(data["Data"]["Data"])
            df = df[["time", "open", "high", "low", "close", "volumefrom"]]
            df.rename(columns={"volumefrom": "volume", "time": "timestamp"}, inplace=True)
            df = df.astype(float)
            
            if len(df) < minutes:
                raise ValueError(f"Not enough rows: got {len(df)}, need {minutes}")
                
            return df.tail(minutes).reset_index(drop=True)
        except Exception as e:
            print(f"[{datetime.utcnow()}] Fetch attempt {attempt} failed: {e}")
            if attempt < MAX_RETRIES:
                time.sleep(RETRY_DELAY)
            else:
                raise e

# === Sequence prep ===
def make_sequence(df, scaler, seq_len=SEQ_LEN):
    values = df[["open", "high", "low", "close", "volume"]].values
    scaled = scaler.transform(values)
    seq = scaled[-seq_len:]
    return torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, features)

# === Track prediction accuracy ===
class AccuracyTracker:
    def __init__(self, max_size=ACCURACY_TRACK_SIZE):
        self.price_predictions = []
        self.direction_predictions = []
        self.price_actuals = []
        self.direction_actuals = []
        self.timestamps = []
        self.max_size = max_size
    
    def add_prediction(self, price_pred, direction_pred, price_actual, direction_actual, timestamp):
        self.price_predictions.append(price_pred)
        self.direction_predictions.append(direction_pred)
        self.price_actuals.append(price_actual)
        self.direction_actuals.append(direction_actual)
        self.timestamps.append(timestamp)
        
        # Keep only the most recent predictions
        if len(self.price_predictions) > self.max_size:
            self.price_predictions.pop(0)
            self.direction_predictions.pop(0)
            self.price_actuals.pop(0)
            self.direction_actuals.pop(0)
            self.timestamps.pop(0)
    
    def calculate_accuracy(self):
        if len(self.price_predictions) == 0:
            return None, None, None, None
            
        # Price accuracy (MAPE)
        price_errors = []
        for pred, actual in zip(self.price_predictions, self.price_actuals):
            if actual != 0:
                price_errors.append(abs(pred - actual) / abs(actual))
        price_mape = np.mean(price_errors) * 100 if price_errors else 0
        
        # Direction accuracy
        direction_correct = sum(1 for pred, actual in zip(self.direction_predictions, self.direction_actuals) if pred == actual)
        direction_accuracy = direction_correct / len(self.direction_predictions) * 100
        
        # Mean absolute error for price
        mae = np.mean([abs(pred - actual) for pred, actual in zip(self.price_predictions, self.price_actuals)])
        
        return price_mape, direction_accuracy, mae, len(self.price_predictions)

# === Generate performance report ===
def generate_performance_report():
    global prediction_data
    
    if not prediction_data:
        print("No prediction data to report.")
        return
    
    # Create output directory if it doesn't exist
    os.makedirs(OUTPUT_DIR, exist_ok=True)
    
    # Convert to DataFrame
    df = pd.DataFrame(prediction_data)
    
    # Save to CSV
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    csv_path = os.path.join(OUTPUT_DIR, f"performance_{timestamp}.csv")
    df.to_csv(csv_path, index=False)
    print(f"Performance data saved to {csv_path}")
    
    # Generate plots
    create_performance_plots(df, timestamp)

# === Create performance plots ===
def create_performance_plots(df, timestamp):
    # Create figure with subplots
    fig, axes = plt.subplots(3, 1, figsize=(12, 15))
    fig.suptitle('BTC Price Prediction Performance', fontsize=16)
    
    # Convert timestamp to datetime for plotting
    df['datetime'] = pd.to_datetime(df['timestamp'], unit='s')
    
    # Plot 1: Price predictions vs actual
    axes[0].plot(df['datetime'], df['actual_close'], label='Actual Price', color='blue')
    axes[0].plot(df['datetime'], df['predicted_close'], label='Predicted Price', color='red', linestyle='--')
    axes[0].set_title('Price Predictions vs Actual')
    axes[0].set_ylabel('Price (USD)')
    axes[0].legend()
    axes[0].grid(True)
    
    # Plot 2: Price error over time
    df['price_error_pct'] = (df['predicted_close'] - df['actual_close']) / df['actual_close'] * 100
    axes[1].plot(df['datetime'], df['price_error_pct'], color='green')
    axes[1].axhline(y=0, color='black', linestyle='-')
    axes[1].set_title('Prediction Error (%)')
    axes[1].set_ylabel('Error (%)')
    axes[1].grid(True)
    
    # Plot 3: Direction accuracy over time (rolling average)
    df['direction_correct'] = (df['predicted_direction'] == df['actual_direction']).astype(int)
    df['rolling_accuracy'] = df['direction_correct'].rolling(window=min(20, len(df))).mean() * 100
    axes[2].plot(df['datetime'], df['rolling_accuracy'], color='purple')
    axes[2].axhline(y=50, color='red', linestyle='--', label='Random Guess')
    axes[2].set_title('Direction Prediction Accuracy (Rolling Average)')
    axes[2].set_ylabel('Accuracy (%)')
    axes[2].set_ylim(0, 100)
    axes[2].legend()
    axes[2].grid(True)
    
    # Adjust layout
    plt.tight_layout()
    
    # Save plot
    plot_path = os.path.join(OUTPUT_DIR, f"performance_plot_{timestamp}.png")
    plt.savefig(plot_path, dpi=100)
    print(f"Performance plot saved to {plot_path}")
    
    # Calculate overall statistics
    overall_direction_accuracy = df['direction_correct'].mean() * 100
    mean_abs_error = np.mean(np.abs(df['price_error_pct']))
    
    print("\nOverall Performance Statistics:")
    print(f"Direction Accuracy: {overall_direction_accuracy:.2f}%")
    print(f"Mean Absolute Error: {mean_abs_error:.2f}%")
    print(f"Total Predictions: {len(df)}")

# === Live prediction loop ===
def live_loop():
    global running, prediction_data
    
    model, scaler, y_scaler = load_model()
    if model is None:
        print("Failed to load model. Exiting.")
        return
        
    accuracy_tracker = AccuracyTracker()
    
    print("Starting live prediction loop...")
    print(f"Fetching last {SEQ_LEN} candles for initial prediction")
    print("Press Ctrl+C to stop and generate performance report")
    
    while running:
        try:
            # Fetch the most recent data
            df = fetch_minute_ohlcv(SEQ_LEN + 1)
            
            # Get the timestamp of the last candle
            last_timestamp = df["timestamp"].iloc[-1] # type: ignore
            last_close = df["close"].iloc[-1] # type: ignore
            
            # Prepare sequence for prediction
            seq = make_sequence(df, scaler)
            
            # Make prediction
            with torch.no_grad():
                price_delta_scaled, direction_logit = model(seq)
                
                # Convert scaled price delta back to original scale
                price_delta = y_scaler.inverse_transform(price_delta_scaled.cpu().numpy())[0, 0] # type: ignore
                
                # Convert direction logit to probability and then to binary prediction
                direction_prob = torch.sigmoid(direction_logit).item()
                direction_pred = 1 if direction_prob > 0.5 else 0
                
            # Calculate predicted next close price
            predicted_next_close = last_close + price_delta
            
            # Print prediction
            direction_str = "UP" if direction_pred == 1 else "DOWN"
            print(f"[{datetime.now()}] Last close: ${last_close:.2f}")
            print(f"  Predicted delta: ${price_delta:.2f} ({price_delta/last_close*100:.2f}%)")
            print(f"  Predicted next close: ${predicted_next_close:.2f}")
            print(f"  Predicted direction: {direction_str} (confidence: {direction_prob:.2f})")
            
            # Wait for the next candle
            print(f"Waiting for next candle (polling every {POLL_INTERVAL}s)...")
            new_candle = False
            
            while not new_candle and running:
                time.sleep(POLL_INTERVAL)
                
                try:
                    # Fetch the latest data to check if we have a new candle
                    latest_df = fetch_minute_ohlcv(2)
                    latest_timestamp = latest_df["timestamp"].iloc[-1] # type: ignore
                    
                    if latest_timestamp > last_timestamp:
                        # We have a new candle
                        new_candle = True
                        actual_next_close = latest_df["close"].iloc[-1] # type: ignore
                        
                        # Calculate actual direction
                        actual_direction = 1 if actual_next_close > last_close else 0
                        
                        # Calculate error
                        price_error = abs(predicted_next_close - actual_next_close)
                        price_error_pct = price_error / last_close * 100
                        
                        # Store prediction data for later analysis
                        prediction_data.append({
                            'timestamp': latest_timestamp,
                            'actual_close': actual_next_close,
                            'predicted_close': predicted_next_close,
                            'predicted_direction': direction_pred,
                            'actual_direction': actual_direction,
                            'direction_confidence': direction_prob,
                            'price_error': price_error,
                            'price_error_pct': price_error_pct
                        })
                        
                        # Print results
                        actual_direction_str = "UP" if actual_direction == 1 else "DOWN"
                        print(f"  Actual next close: ${actual_next_close:.2f}")
                        print(f"  Actual direction: {actual_direction_str}")
                        print(f"  Price error: ${price_error:.2f} ({price_error_pct:.2f}%)")
                        print(f"  Direction prediction: {'✓' if direction_pred == actual_direction else '✗'}")
                        
                        # Update accuracy tracker
                        accuracy_tracker.add_prediction(
                            predicted_next_close, direction_pred, 
                            actual_next_close, actual_direction, latest_timestamp
                        )
                        
                        # Calculate and print accuracy
                        price_mape, direction_acc, mae, count = accuracy_tracker.calculate_accuracy()
                        if count > 0: # type: ignore
                            print(f"  Recent accuracy (last {count} predictions):")
                            print(f"    Price MAPE: {price_mape:.2f}%")
                            print(f"    Price MAE: ${mae:.2f}")
                            print(f"    Direction accuracy: {direction_acc:.2f}%")
                        
                        print("-" * 50)
                        
                except Exception as e:
                    print(f"Error checking for new candle: {e}")
                    continue
                    
        except Exception as e:
            print(f"Error in prediction loop: {e}")
            time.sleep(RETRY_DELAY)

if __name__ == "__main__":
    live_loop()