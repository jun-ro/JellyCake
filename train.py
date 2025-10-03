import torch
import torch.nn as nn
from torch.amp import autocast, GradScaler
from model import GRUModel
from tqdm import tqdm
import pickle
from loader import BTCDataLoader
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt  # For diagnostics

# Variables
Epochs = 20  # Start small for testing
patience = 5
Output_file = "model/model.pth"
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

# Diagnostics to verify data
def run_diagnostics(data_loader):
    print('\n=== DIAGNOSTICS ===')
    df = data_loader.df
    print(f'Rows: {len(df)}, Date range: {df["Datetime"].min()} to {df["Datetime"].max()}')
    print(f'Price range: ${df["Low"].min():.2f} - ${df["High"].max():.2f}')
    print(f'Std Close: ${df["Close"].std():.2f}')
    changes = df["Close"].diff().dropna()
    change_pct = len(changes[changes != 0]) / len(changes) * 100
    print(f'Price changes (% non-zero): {change_pct:.1f}%')
    # Quick plot
    plt.figure(figsize=(10, 4))
    sample = df.tail(5000)['Close']
    plt.plot(sample)
    plt.title('Sample Close Prices')
    plt.savefig('diagnostics_plot.png')
    plt.close()
    print('Plot saved as diagnostics_plot.png')

run_diagnostics(data_loader)

# Model
model = GRUModel(input_size=5, hidden_size=64, num_layers=2)
model.to(device)

def train():
    criterion = nn.MSELoss()
    optimizer = torch.optim.Adam(model.parameters(), lr=Learning_rate)
    scaler = GradScaler('cuda')  # AMP scaler
    
    best_loss = float('inf')
    patience_counter = 0
    
    for epoch in range(Epochs):
        model.train()
        total_loss = 0
        progress_bar = tqdm(train_loader, desc=f'Epoch {epoch+1}/{Epochs}')
        
        for batch_X, batch_y in progress_bar:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            
            optimizer.zero_grad()  # Clear BEFORE forward (fix #1)
            
            with autocast('cuda'):  # FP16 forward (no duplicate)
                predictions = model(batch_X)
                loss = criterion(predictions, batch_y)
            
            # AMP backward (fix #2: No duplicate loss)
            scaler.scale(loss).backward()
            scaler.step(optimizer)
            scaler.update()
            
            total_loss += loss.item()
            progress_bar.set_postfix({'Loss': f'{loss.item():.4f}'})
        
        avg_loss = total_loss / len(train_loader)
        print(f'Epoch [{epoch+1}/{Epochs}] Complete - Avg Loss: {avg_loss:.4f}')
        
        if avg_loss < best_loss:
            best_loss = avg_loss
            patience_counter = 0
            print(f'New best: {best_loss:.4f}')
        else:
            patience_counter += 1
            if patience_counter >= patience:
                print(f'Early stopping at epoch {epoch+1}')
                break
    
    print(f'Final best loss: {best_loss:.4f}')

def quick_eval(model, data_loader, device):
    """Corrected eval for delta model"""
    model.eval()
    all_preds, all_targets = [], []
    
    with torch.no_grad():
        for batch_X, batch_y in test_loader:
            batch_X = batch_X.to(device)
            batch_y = batch_y.to(device)
            with autocast('cuda'):
                preds = model(batch_X)
            all_preds.append(preds.cpu().numpy().flatten())
            all_targets.append(batch_y.cpu().numpy().flatten())
    
    predictions = np.concatenate(all_preds)
    targets = np.concatenate(all_targets)
    
    # Inverse deltas (using y_scaler)
    pred_delta = data_loader.inverse_transform_predictions(predictions, is_delta=True)
    target_delta = data_loader.inverse_transform_predictions(targets, is_delta=True)
    
    # Reconstruct full prices (key fix: add delta to previous close)
    # Get test part of df (approximate: after train split + seq_length offset)
    test_start_idx = len(data_loader.X_train.shape[0]) + data_loader.seq_length - 1  # Adjust for sequences
    test_df = data_loader.df.iloc[test_start_idx:].reset_index(drop=True)
    previous_closes = test_df['Close'].values[:-1]  # Previous for each test prediction
    full_targets = previous_closes + target_delta
    full_predictions = previous_closes + pred_delta
    
    # Metrics on full prices
    mae_full = mean_absolute_error(full_targets, full_predictions)
    rmse_full = np.sqrt(mean_squared_error(full_targets, full_predictions))
    r2_full = r2_score(full_targets, full_predictions)
    
    # Metrics on deltas (change error)
    mae_delta = mean_absolute_error(target_delta, pred_delta)
    rmse_delta = np.sqrt(mean_squared_error(target_delta, pred_delta))
    r2_delta = r2_score(target_delta, pred_delta)
    
    # Baseline delta (assume 0 change = previous price)
    baseline_mae = np.mean(np.abs(target_delta))  # Average |actual delta|
    
    # Directional accuracy (up/down correct)
    dir_actual = np.sign(target_delta)  # +1 up, -1 down, 0 flat
    dir_pred = np.sign(pred_delta)
    dir_acc = np.mean(dir_actual == dir_pred) * 100
    
    print(f'\n=== QUICK EVAL (DELTA MODEL) ===')
    print(f'Full Prices Metrics:')
    print(f'  MAE: ${mae_full:.2f} | RMSE: ${rmse_full:.2f} | R²: {r2_full:.4f}')
    print(f'Delta Change Metrics:')
    print(f'  MAE: ${mae_delta:.2f} | RMSE: ${rmse_delta:.2f} | R²: {r2_delta:.4f}')
    print(f'Baseline Delta MAE: ${baseline_mae:.2f} ({"✓ Beats" if mae_delta < baseline_mae else "✗ Worse"} by ${abs(baseline_mae - mae_delta):.2f})')
    print(f'Directional Accuracy (Up/Down Correct): {dir_acc:.1f}% (random = 50%)')
    
    # Save corrected CSV
    results_df = pd.DataFrame({
        'Previous_Close': previous_closes,
        'Predicted_Delta': pred_delta,
        'Actual_Delta': target_delta,
        'Predicted_Full': full_predictions,
        'Actual_Full': full_targets,
        'Error_Full': full_targets - full_predictions,
        'Dir_Correct': (dir_actual == dir_pred).astype(int)
    })
    results_df.to_csv('delta_eval_results.csv', index=False)
    print(f'Saved to delta_eval_results.csv ({len(results_df)} rows)')
    
    return mae_full, rmse_full, r2_full, mae_delta, dir_acc

def save_model(model, filepath=Output_file):
    torch.save({
        'model_state_dict': model.state_dict(),
        'input_size': 5, 'hidden_size': 64, 'num_layers': 2,
    }, filepath)
    print(f'Model saved to {filepath}')

def save_scaler(scaler, filepath='model/scaler.pkl'):
    with open(filepath, 'wb') as f:
        pickle.dump(scaler, f)
    print(f'Scaler saved to {filepath}')

# Run
train()
save_model(model)
save_scaler(data_loader.scaler)
mae, rmse, r2 = quick_eval(model, data_loader, device)
print(f'Complete! MAE=${mae:.2f}, R²={r2:.4f}')