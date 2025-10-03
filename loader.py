import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class BTCDataLoader:
    def __init__(self, csv_path, max_rows=None, seq_length=60, 
                 test_size=0.2, scale=True):
        """
        Load and prepare Bitcoin OHLCV data for GRU model
        
        Args:
            csv_path: path to CSV file
            max_rows: maximum number of rows to load (None = all rows)
            seq_length: number of past candles to use for prediction
            test_size: proportion of data for testing
            scale: whether to normalize the data
        """
        self.csv_path = csv_path
        self.max_rows = max_rows
        self.seq_length = seq_length
        self.test_size = test_size
        self.scale = scale
        self.scaler = MinMaxScaler() if scale else None
        
        # Load and prepare data
        self.df = self._load_data()
        self.X_train, self.X_test, self.y_train, self.y_test = self._prepare_data()
        
# In loader.py, inside the BTCDataLoader class, update _load_data method:

    def _load_data(self):
        """Load CSV, filter to volatile periods, and take recent data (backwards)"""
        # Read full data (or cap at safe large nrows to avoid memory issues)
        max_read = 5_000_000  # Safe cap for Kaggle file (~3M rows total)
        if self.max_rows and self.max_rows > 1_000_000:
            # For large max_rows, read more to ensure enough after filtering
            max_read = min(self.max_rows * 2, 5_000_000)
        
        df = pd.read_csv(self.csv_path, nrows=max_read)
        
        # Handle Kaggle columns (drop extras if present)
        required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        if 'Volume BTC' in df.columns:
            df['Volume'] = df['Volume BTC']
            required_cols = ['Timestamp', 'Open', 'High', 'Low', 'Close', 'Volume']
        
        # Select only needed columns
        df = df[required_cols]
        
        # Filter to volatile 2020+ period
        start_timestamp = 1577836800  # 2020-01-01
        df = df[df['Timestamp'] >= start_timestamp].reset_index(drop=True)
        
        # Sort by timestamp ASCENDING (old to new) to ensure proper order
        df = df.sort_values('Timestamp').reset_index(drop=True)
        
        # Now, take MOST RECENT rows for max_rows (backwards logic: newest first)
        if self.max_rows and len(df) > self.max_rows:
            df = df.tail(self.max_rows).reset_index(drop=True)  # Last N = most recent
            print(f'Limited to most recent {self.max_rows} rows (newest data first for volatility)')
        
        # Convert timestamp
        df['Datetime'] = pd.to_datetime(df['Timestamp'], unit='s')
        
        # Check for empty after all steps
        if len(df) == 0:
            raise ValueError("No data after filtering! Check timestamp range or CSV content. "
                            "Try max_rows=None or verify post-2020 data exists.")
        
        print(f"Loaded {len(df)} rows (post-2020, most recent {'N/A' if not self.max_rows else self.max_rows} for volatility)")
        print(f"Date range: {df['Datetime'].min()} to {df['Datetime'].max()}")
        print(f"Price range: ${df['Low'].min():.2f} - ${df['High'].max():.2f}")
        
        return df
    
    def _create_sequences(self, data):
        """Create sequences for time series prediction"""
        X, y = [], []
        
        for i in range(self.seq_length, len(data)):
            # X: past seq_length candles (OHLCV)
            X.append(data[i-self.seq_length:i])
            # y: next closing price
            y.append(data[i, 3] - data[i-1, 3])  # index 3 is Close
        
        return np.array(X), np.array(y)
    
    def _prepare_data(self):
        """Prepare and split data for training (with delta for y)"""
        # Select OHLCV columns
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        data = self.df[features].values
        
        # Scale data if requested
        if self.scale:
            # Scale X features (OHLCV) as usual
            self.scaler = MinMaxScaler()  # Ensure it's OHLCV scaler
            data = self.scaler.fit_transform(data)
            
            # Create sequences (y now deltas)
            X, y = self._create_sequences(data)
            
            # Scale y deltas separately (deltas are small, ~ -0.1 to 0.1 after X scaling)
            y_scaler = MinMaxScaler()
            y = y_scaler.fit_transform(y.reshape(-1, 1)).flatten()
            self.y_scaler = y_scaler  # Save for inverse transform (important!)
        else:
            # No scaling
            X, y = self._create_sequences(data)
        
        # Split into train and test
        X_train, X_test, y_train, y_test = train_test_split(
            X, y, test_size=self.test_size, shuffle=False  # no shuffle for time series
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_train = torch.FloatTensor(y_train).reshape(-1, 1)
        y_test = torch.FloatTensor(y_test).reshape(-1, 1)
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Sequence shape: {X_train.shape}")
        
        return X_train, X_test, y_train, y_test
    
    def get_dataloaders(self, batch_size=32):
        """Create PyTorch DataLoaders for training"""
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(self.X_train, self.y_train)
        test_dataset = TensorDataset(self.X_test, self.y_test)
        
        train_loader = DataLoader(
            train_dataset, 
            batch_size=batch_size, 
            shuffle=True
        )
        test_loader = DataLoader(
            test_dataset, 
            batch_size=batch_size, 
            shuffle=False
        )
        
        return train_loader, test_loader
    
    def inverse_transform_predictions(self, predictions, is_delta=False):
        """Convert scaled predictions back to original scale (handles delta y if separate scaler)"""
        if not self.scale:
            return predictions
        
        # Create dummy array for OHLCV inverse
        dummy = np.zeros((len(predictions), 5))
        dummy[:, 3] = predictions.flatten()  # Put in 'Close' column (or delta)
        
        # Inverse X scaler (OHLCV)
        original = self.scaler.inverse_transform(dummy)
        
        # Handle y delta if needed (separate scaler)
        if is_delta and hasattr(self, 'y_scaler'):
            delta_pred = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            original[:, 3] = delta_pred  # Replace with inverse delta
        
        return original[:, 3]  # Return Close/delta column