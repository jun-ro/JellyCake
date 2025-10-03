import torch
import numpy as np
import pandas as pd
import pickle
from model import GRUModel
from datetime import datetime
from types import SimpleNamespace

# Configuration
MODEL_PATH = "model.pth"
SCALER_PATH = "scaler.pkl"
CSV_PATH = "data/btcusd_1-min_data.csv"
SEQ_LENGTH = 60

# Device
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')


def load_model(model_path=MODEL_PATH):
    """Load trained model"""
    checkpoint = torch.load(model_path, map_location=device)
    model = GRUModel(
        input_size=checkpoint['input_size'],
        hidden_size=checkpoint['hidden_size'],
        num_layers=checkpoint['num_layers']
    )
    model.load_state_dict(checkpoint['model_state_dict'])
    model.to(device)
    model.eval()
    return model


def load_scaler(scaler_path=SCALER_PATH):
    """Load scaler"""
    with open(scaler_path, 'rb') as f:
        return pickle.load(f)


def inference():
    """Predict the next closing price and return as object"""
    # Load model and scaler
    model = load_model()
    scaler = load_scaler()
    
    # Load last 60 candles
    df = pd.read_csv(CSV_PATH)
    recent_data = df[['Open', 'High', 'Low', 'Close', 'Volume']].tail(SEQ_LENGTH).values
    
    # Get current time and price
    last_timestamp = df['Timestamp'].iloc[-1]
    last_datetime = datetime.fromtimestamp(last_timestamp)
    current_price = float(df['Close'].iloc[-1])
    
    # Scale and predict
    scaled_data = scaler.transform(recent_data)
    input_tensor = torch.FloatTensor(scaled_data).unsqueeze(0).to(device)
    
    with torch.no_grad():
        scaled_prediction = model(input_tensor).cpu().numpy()
    
    # Inverse transform
    dummy = np.zeros((1, 5))
    dummy[:, 3] = scaled_prediction.flatten()
    predicted_price = float(scaler.inverse_transform(dummy)[0, 3])
    
    # Calculate change
    price_change = predicted_price - current_price
    percent_change = (price_change / current_price) * 100
    
    # Return as object with dot notation access
    return SimpleNamespace(
        current_time=last_datetime.strftime('%Y-%m-%d %H:%M:%S'),
        timestamp=last_timestamp,
        current_price=current_price,
        predicted_price=predicted_price,
        price_change=price_change,
        percent_change=percent_change
    )


if __name__ == '__main__':
    result = inference()
    print(f"Current Time:      {result.current_time}")
    print(f"Current Price:     ${result.current_price:.2f}")
    print(f"Predicted Price:   ${result.predicted_price:.2f}")
    print(f"Change:            ${result.price_change:+.2f} ({result.percent_change:+.2f}%)")