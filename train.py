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
Output_file = "model/Aurora_v3.pth"
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

data_loader.check_binary_direction_data()

# Get loaders with speed optimizations
train_loader, test_loader = data_loader.get_dataloaders(
    batch_size=128,  # Increased for speed (adjust to 64 if OOM)
)


print(f'Train batches: {len(train_loader)} | Test batches: {len(test_loader)}')

# Model
model = GRUModel(input_size=5, hidden_size=64, num_layers=2)
model.to(device)

## Improved Loss Function

class ImprovedDualLoss(nn.Module):
    def __init__(self, price_weight=0.3, direction_weight=0.7):
        super().__init__()
        self.price_weight = price_weight
        self.direction_weight = direction_weight
        self.mse = nn.MSELoss()
        self.bce_with_logits = nn.BCEWithLogitsLoss()  # More numerically stable than BCE + Sigmoid
    
    def forward(self, price_pred, direction_logit, price_target, direction_target):
        price_loss = self.mse(price_pred, price_target)
        
        # Ensure direction_logit is the right shape and type
        direction_logit = direction_logit.squeeze()
        direction_target = direction_target.float()
        
        direction_loss = self.bce_with_logits(direction_logit, direction_target)
        
        return self.price_weight * price_loss + self.direction_weight * direction_loss

def train():
    criterion = ImprovedDualLoss(price_weight=0.3, direction_weight=0.7)
    
    # Different learning rates for different parts of the model
    optimizer = torch.optim.AdamW([
        {'params': model.price_feature_net.parameters()},
        {'params': model.direction_feature_net.parameters()},
        {'params': model.gru.parameters()},
        {'params': model.price_attention.parameters()},
        {'params': model.direction_attention.parameters()},
        {'params': model.price_head.parameters(), 'lr': 0.001},
        {'params': model.direction_head.parameters(), 'lr': 0.01}  # Higher LR for direction
    ], lr=0.001, weight_decay=0.01)
    
    # Use a scheduler that reduces LR when validation loss plateaus
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2,
    )
    
    scaler = GradScaler('cuda')
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Epochs):
        model.train()
        total_loss = 0
        price_loss_total = 0
        direction_loss_total = 0
        
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Epochs}')
        
        for batch_idx, (batch_X, batch_y_price, batch_y_direction) in enumerate(progress_bar):
            batch_X = batch_X.to(device)
            batch_y_price = batch_y_price.to(device)
            batch_y_direction = batch_y_direction.to(device)
            
            optimizer.zero_grad()
            
            with autocast('cuda'):
                price_pred, direction_logit = model(batch_X)  # Changed variable name
                loss = criterion(price_pred, direction_logit, batch_y_price, batch_y_direction)
                
                # Calculate individual losses for monitoring
                price_loss = criterion.mse(price_pred, batch_y_price)
                
                direction_logit_squeezed = direction_logit.squeeze()
                direction_target_float = batch_y_direction.float()
                direction_loss = criterion.bce_with_logits(direction_logit_squeezed, direction_target_float)
            
            scaler.scale(loss).backward()
            
            # Debugging: check gradients every 100 batches
            if batch_idx % 100 == 0:
                direction_head_grad_norm = 0
                for param in model.direction_head.parameters():
                    if param.grad is not None:
                        direction_head_grad_norm += param.grad.data.norm(2).item() ** 2
                direction_head_grad_norm = direction_head_grad_norm ** 0.5
                
                # Check predictions
                with torch.no_grad():
                    # Apply sigmoid to get probabilities
                    direction_prob = torch.sigmoid(direction_logit).mean().item()
                    direction_acc = ((torch.sigmoid(direction_logit) > 0.5).float() == batch_y_direction.float()).float().mean().item()
                
                print(f"\nBatch {batch_idx}: Direction head grad norm: {direction_head_grad_norm:.6f}")
                print(f"Average direction prediction: {direction_prob:.4f}")
                print(f"Direction accuracy in batch: {direction_acc:.4f}")
            
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
        
        # Calculate average losses for this epoch
        avg_train_loss = total_loss / len(train_loader)
        avg_price_loss = price_loss_total / len(train_loader)
        avg_direction_loss = direction_loss_total / len(train_loader)
        
        print(f'Epoch [{epoch+1}/{Epochs}] Complete')
        print(f'  Train Total Loss: {avg_train_loss:.4f}')
        print(f'  Train Price Loss: {avg_price_loss:.4f}')
        print(f'  Train Direction Loss: {avg_direction_loss:.4f}')
        
        # Evaluate on validation set
        val_loss, val_price_loss, val_direction_loss, val_direction_acc = evaluate_on_validation(
            model, test_loader, criterion, device
        )
        
        print(f'  Val Total Loss: {val_loss:.4f}')
        print(f'  Val Price Loss: {val_price_loss:.4f}')
        print(f'  Val Direction Loss: {val_direction_loss:.4f}')
        print(f'  Val Direction Accuracy: {val_direction_acc:.4f}')
        
        # Update scheduler based on validation loss
        scheduler.step(val_loss)
        
        # Early stopping logic
        if val_loss < best_loss:
            best_loss = val_loss
            patience_counter = 0
            print(f'New best: {best_loss:.4f}')
            # Save best model
            torch.save({
                'model_state_dict': model.state_dict(),
                'optimizer_state_dict': optimizer.state_dict(),
                'loss': best_loss,
            }, 'model/best_model.pth')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    print(f'Final best loss: {best_loss:.4f}')
    return best_loss

def evaluate_on_validation(model, val_loader, criterion, device):
    """Evaluate model on validation set"""
    model.eval()
    total_loss = 0
    price_loss_total = 0
    direction_loss_total = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for batch_X, batch_y_price, batch_y_direction in val_loader:
            batch_X = batch_X.to(device)
            batch_y_price = batch_y_price.to(device)
            batch_y_direction = batch_y_direction.to(device)
            
            price_pred, direction_logit = model(batch_X)  # Changed variable name
            loss = criterion(price_pred, direction_logit, batch_y_price, batch_y_direction)
            
            # Calculate individual losses
            price_loss = criterion.mse(price_pred, batch_y_price)
            
            direction_logit_squeezed = direction_logit.squeeze()
            direction_target_float = batch_y_direction.float()
            direction_loss = criterion.bce_with_logits(direction_logit_squeezed, direction_target_float)
            
            total_loss += loss.item()
            price_loss_total += price_loss.item()
            direction_loss_total += direction_loss.item()
            
            # Calculate direction accuracy
            predicted = (torch.sigmoid(direction_logit) > 0.5).float()  # Apply sigmoid before thresholding
            total += batch_y_direction.size(0)
            correct += (predicted.squeeze() == batch_y_direction.float()).sum().item()
    
    avg_loss = total_loss / len(val_loader)
    avg_price_loss = price_loss_total / len(val_loader)
    avg_direction_loss = direction_loss_total / len(val_loader)
    direction_accuracy = correct / total
    
    return avg_loss, avg_price_loss, avg_direction_loss, direction_accuracy

def save_model(model, filepath="model/upgraded_model.pth"):
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': 5, 
        'hidden_size': 64, 
        'num_layers': 2,
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