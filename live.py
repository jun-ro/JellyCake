import time
import requests
import torch
import numpy as np
import pandas as pd
import pickle
from datetime import datetime
from model import GRUModel

# === Config ===
SEQ_LEN = 60
RETRY_DELAY = 5
MAX_RETRIES = 5

# === Load trained model + scalers ===
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

# === Fetch BTC minute OHLCV from CryptoCompare ===
def fetch_minute_ohlcv(minutes=SEQ_LEN + 1):
    url = "https://min-api.cryptocompare.com/data/v2/histominute"
    params = {"fsym": "BTC", "tsym": "USD", "limit": minutes}
    resp = requests.get(url, params=params)
    resp.raise_for_status()
    data = resp.json()
    if "Data" not in data or "Data" not in data["Data"]:
        raise ValueError("No OHLCV data in CryptoCompare response")

    df = pd.DataFrame(data["Data"]["Data"])
    df = df[["open", "high", "low", "close", "volumefrom"]]
    df.rename(columns={"volumefrom": "volume"}, inplace=True)
    df = df.astype(float)

    if len(df) < minutes:
        raise ValueError(f"Not enough rows: got {len(df)}, need {minutes}")

    return df.tail(minutes).reset_index(drop=True)

# === Sequence prep ===
def make_sequence(df, seq_len=SEQ_LEN):
    values = df[["open", "high", "low", "close", "volume"]].values
    scaled = scaler.transform(values)
    seq = scaled[-seq_len:]
    return torch.FloatTensor(seq).unsqueeze(0)  # (1, seq_len, features)

# === Live loop ===
def live_loop():
    while True:
        # Retry fetch
        for attempt in range(1, MAX_RETRIES + 1):
            try:
                df = fetch_minute_ohlcv(SEQ_LEN + 1)
                break
            except Exception as e:
                print(f"[{datetime.utcnow()}] Fetch attempt {attempt} failed: {e}")
                if attempt == MAX_RETRIES:
                    print("Max retries reached, skipping this round.")
                    time.sleep(RETRY_DELAY)
                    continue
                time.sleep(RETRY_DELAY)
        else:
            continue  # skip this round if all retries failed

        seq = make_sequence(df)

        # Predict delta
        with torch.no_grad():
            pred_scaled = model(seq).cpu().numpy()
        pred_delta = y_scaler.inverse_transform(pred_scaled)[0, 0]

        # Apply delta
        last_close = df["close"].iloc[-1]
        predicted_next_close = last_close + pred_delta

        # Diagnostic print
        print(f"[{datetime.now()}] Last close: {last_close:.2f} | "
              f"Predicted Δ: {pred_delta:.2f} | "
              f"Predicted next: {predicted_next_close:.2f}")

        last_time = df.index[-1]  # last candle index
        
        while True:
            df2 = fetch_minute_ohlcv(2)
            if df2.index[-1] != last_time:  # new candle available
                break
            time.sleep(5)  # poll every 5 seconds

        # Fetch next candle for actual comparison
        try:
            df2 = fetch_minute_ohlcv(2)
            actual_next = df2["close"].iloc[-1]
            error = abs(predicted_next_close - actual_next)
            print(f"    Actual next close: {actual_next:.2f} | Error: {error:.2f}\n")
        except Exception as e:
            print(f"Failed to fetch next actual close: {e}\n")

if __name__ == "__main__":
    live_loop()
