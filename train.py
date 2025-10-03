import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler # type: ignore
from model import GRUModel
from tqdm import tqdm
import torch.nn.functional as F
import pickle
from loader import BTCDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # For diagnostics

# Variables
Epochs = 20  # Start small for testing
patience = 5
Output_file = "model/upgraded_model.pth"
Learning_rate = 0.001
Max_rows = 500_000  # Your bull run subset

# Use GPU
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print(f'Using device: {device}')

# Load and prepare dataset
data_loader = BTCDataLoader(
    csv_path="data/btcusd_1-min_data.csv",
    max_rows=Max_rows,
    seq_length=60,
    test_size=0.2,
    scale=True
)

# Get loaders with speed optimizations
train_loader, test_loader = data_loader.get_dataloaders(
    batch_size=128,  # Increased for speed (adjust to 64 if OOM)
)
print(f'Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')

# Model
model = GRUModel(input_size=5, hidden_size=64, num_layers=2)
model.to(device)

## Improved Loss Function

class DualLoss(nn.Module):
    def __init__(self, price_weight=0.7, direction_weight=0.3):
        super().__init__()
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.mse = nn.MSELoss()
        self.ce = nn.CrossEntropyLoss()
    
    def forward(self, price_pred, direction_pred, price_target, direction_target):
        price_loss = self.mse(price_pred, price_target)
        direction_loss = self.ce(direction_pred, direction_target)
        
        return self.price_weight * price_loss + self.direction_weight * direction_loss

def train():
    criterion = DualLoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    scaler = GradScaler('cuda')
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Epochs):
        model.train()
        total_loss = 0
        price_loss_total = 0
        direction_loss_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Epochs}')
        
        for batch_X, (batch_y_price, batch_y_direction) in progress_bar:
            batch_X = batch_X.to(device)
            batch_y_price = batch_y_price.to(device)
            batch_y_direction = batch_y_direction.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                price_pred, direction_pred = model(batch_X)
                loss = criterion(price_pred, direction_pred, batch_y_price, batch_y_direction)
                
                # Calculate individual losses for reporting
                price_loss = criterion.mse(price_pred, batch_y_price)
                direction_loss = criterion.ce(direction_pred, batch_y_direction)
            
            scaler.scale(loss).backward()
            scaler.unscale_(optimizer)
            nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            price_loss_total += price_loss.item()
            direction_loss_total += direction_loss.item()
            
            progress_bar.set_postfix({
                'Loss': f'{loss.item():.4f}',
                'Price': f'{price_loss.item():.4f}',
                'Dir': f'{direction_loss.item():.4f}'
            })
        
        avg_loss = total_loss / len(train_loader)
        avg_price_loss = price_loss_total / len(train_loader)
        avg_direction_loss = direction_loss_total / len(train_loader)
        
        print(f'Epoch [{epoch+1}/{Epochs}] Complete')
        print(f'  Total Loss: {avg_loss:.4f}')
        print(f'  Price Loss: {avg_price_loss:.4f}')
        print(f'  Direction Loss: {avg_direction_loss:.4f}')
        
        # Early stopping logic
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            print(f'New best: {best_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break

def save_model(model, filepath=Output_file):
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': 5, 'hidden_size': 64, 'num_layers': 2,
    }, filepath)
    print(f'Model saved to {filepath}')

def save_scalers(data_loader, x_path='model/scaler.pkl', y_path='model/y_scaler.pkl'):
    # Save input scaler
    with open(x_path, 'wb') as f:
        pickle.dump(data_loader.scaler, f)
    print(f'Input scaler saved to {x_path}')

    # Save y (delta) scaler if present
    if hasattr(data_loader, 'y_scaler'):
        with open(y_path, 'wb') as f:
            pickle.dump(data_loader.y_scaler, f)
        print(f'Y (delta) scaler saved to {y_path}')

# Run
train()
save_model(model)
save_scalers(data_loader)