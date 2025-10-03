import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.model_selection import train_test_split

class BTCDataLoader:
    def __init__(self, csv_path, max_rows=None, seq_length=60, 
                 test_size=0.2, scale=True):
        """
        Load and prepare Bitcoin OHLCV data for GRU model with dual outputs
        
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
        self.X_train, self.X_test, self.y_price_train, self.y_direction_train, self.y_price_test, self.y_direction_test = self._prepare_data()
    
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
    
    def _create_sequences(self, scaled_data, price_changes, directions):
        """Create sequences using pre-calculated price changes and directions"""
        X, y_price, y_direction = [], [], []
        
        for i in range(self.seq_length, len(scaled_data)):
            # X: past seq_length candles (OHLCV) - use scaled data
            X.append(scaled_data[i-self.seq_length:i])
            
            # Use the ORIGINAL price changes and directions (calculated before scaling)
            y_price.append(price_changes[i])
            y_direction.append(directions[i])
        
        return np.array(X), np.array(y_price), np.array(y_direction)
    
    def _prepare_data(self):
        """Prepare and split data for training (with delta and direction for y)"""
        # Select OHLCV columns
        features = ['Open', 'High', 'Low', 'Close', 'Volume']
        original_data = self.df[features].values
        
        # Calculate price changes and directions from ORIGINAL data (before scaling)
        price_changes = np.zeros(len(original_data))
        directions = np.zeros(len(original_data), dtype=int)
        
        for i in range(1, len(original_data)):
            price_change = original_data[i, 3] - original_data[i-1, 3]  # Close price difference
            price_changes[i] = price_change
            directions[i] = 1 if price_change > 0 else 0
        
        # Scale the features (OHLCV)
        if self.scale:
            self.scaler = MinMaxScaler()  # OHLCV scaler
            scaled_data = self.scaler.fit_transform(original_data)
        else:
            scaled_data = original_data
        
        # Create sequences using scaled features but original price changes and directions
        X, y_price, y_direction = self._create_sequences(scaled_data, price_changes, directions)
        
        # Scale y deltas separately
        if self.scale:
            y_scaler = MinMaxScaler()
            y_price = y_scaler.fit_transform(y_price.reshape(-1, 1)).flatten()
            self.y_scaler = y_scaler  # Save for inverse transform
        
        # Split into train and test
        X_train, X_test, y_price_train, y_price_test, y_direction_train, y_direction_test = train_test_split(
            X, y_price, y_direction, test_size=self.test_size, shuffle=False
        )
        
        # Convert to PyTorch tensors
        X_train = torch.FloatTensor(X_train)
        X_test = torch.FloatTensor(X_test)
        y_price_train = torch.FloatTensor(y_price_train).reshape(-1, 1)
        y_price_test = torch.FloatTensor(y_price_test).reshape(-1, 1)
        y_direction_train = torch.LongTensor(y_direction_train)
        y_direction_test = torch.LongTensor(y_direction_test)
        
        print(f"\nTrain set: {X_train.shape[0]} samples")
        print(f"Test set: {X_test.shape[0]} samples")
        print(f"Sequence shape: {X_train.shape}")
        
        return X_train, X_test, y_price_train, y_direction_train, y_price_test, y_direction_test
    
    def get_dataloaders(self, batch_size=32):
        """Create PyTorch DataLoaders for training with dual outputs"""
        from torch.utils.data import DataLoader, TensorDataset
        
        train_dataset = TensorDataset(
            self.X_train, 
            self.y_price_train,
            self.y_direction_train
        )
        
        test_dataset = TensorDataset(
            self.X_test, 
            self.y_price_test,
            self.y_direction_test
        )
        
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
        original = self.scaler.inverse_transform(dummy) #type: ignore
        
        # Handle y delta if needed (separate scaler)
        if is_delta and hasattr(self, 'y_scaler'):
            delta_pred = self.y_scaler.inverse_transform(predictions.reshape(-1, 1)).flatten()
            original[:, 3] = delta_pred  # Replace with inverse delta
        
        return original[:, 3]  # Return Close/delta column
    
    # Add this to your data loader after creating sequences
    def analyze_direction_distribution(self):
        """Analyze direction distribution and print insights"""
        direction_counts = np.bincount(self.y_direction_train.numpy())
        total = len(self.y_direction_train)
        
        print("Direction distribution in training set:")
        print(f"  Up (0): {direction_counts[0]} ({direction_counts[0]/total:.2%})")
        print(f"  Down (1): {direction_counts[1]} ({direction_counts[1]/total:.2%})")
        print(f"  Flat (2): {direction_counts[2]} ({direction_counts[2]/total:.2%})")
        
        # Calculate class imbalance ratio
        max_count = np.max(direction_counts)
        min_count = np.min(direction_counts)
        imbalance_ratio = max_count / min_count
        
        print(f"Class imbalance ratio: {imbalance_ratio:.2f}")
        
        if imbalance_ratio > 2.0:
            print("Significant class imbalance detected!")
            return False
        return True

    def check_binary_direction_data(self):
        """Verify binary direction data is correctly prepared"""
        print("Checking binary direction data...")
        
        # We need to use the original, unscaled price changes for comparison
        # Let's recalculate them to be sure
        original_data = self.df[['Open', 'High', 'Low', 'Close', 'Volume']].values
        original_price_changes = np.zeros(len(original_data))
        
        for i in range(1, len(original_data)):
            price_change = original_data[i, 3] - original_data[i-1, 3]  # Close price difference
            original_price_changes[i] = price_change
        
        # Check a larger sample
        errors = 0
        for i in range(min(100, len(self.y_direction_train))):
            idx = np.random.randint(0, len(self.y_direction_train))
            
            # We need to adjust the index because sequences were created
            # with seq_length offset
            original_idx = idx + self.seq_length
            
            # Use the original price change
            price_change = original_price_changes[original_idx]
            direction = self.y_direction_train[idx].item()
            
            # Verify direction matches price change
            if (price_change > 0 and direction != 1) or (price_change <= 0 and direction != 0):
                print(f"ERROR: Sample {idx} (original idx: {original_idx}) - Price change: {price_change:.6f}, Direction: {direction}")
                errors += 1
        
        if errors > 0:
            print(f"Found {errors} mismatches out of 100 samples!")
            return False
        
        # Check distribution
        up_count = torch.sum(self.y_direction_train == 1).item()
        down_count = torch.sum(self.y_direction_train == 0).item()
        total = len(self.y_direction_train)
        
        print(f"Direction distribution: Up: {up_count} ({up_count/total:.2%}), Down: {down_count} ({down_count/total:.2%})")
        
        # Check for any NaN or extreme values
        if torch.isnan(self.y_price_train).any():
            print("WARNING: Found NaN values in price changes!")
            return False
        
        if torch.isinf(self.y_price_train).any():
            print("WARNING: Found infinite values in price changes!")
            return False
        
        print("Data check passed!")
        return True