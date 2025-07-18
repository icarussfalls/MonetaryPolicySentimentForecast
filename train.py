from model import FinBERTRegressor
from dataset import PolicyReturnDataset
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import shap

import random
import os

def set_seeds(seed):
    """Set seeds for reproducible results"""
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ['PYTHONHASHSEED'] = str(seed)

# Set seeds for reproducibility
set_seeds(42)


device = torch.device("cuda" if torch.cuda.is_available() else "cpu")

# Load and preprocess data
df = pd.read_csv("/Users/icarus/Desktop/event_driven/monetary_policy/index_30d_returns.csv")
report_embeddings = np.load("report_embeddings.npy", allow_pickle=True).item()

# Print initial data info
print(f"Initial data shape: {df.shape}")
print(f"Original FY format in DataFrame: {df['FY'].iloc[0]}")

# Convert fiscal year format from "YYYY-YY" to "YYYY_YY"
df['FY'] = df['FY'].str.replace('-', '_')
print(f"Available fiscal years in DataFrame: {sorted(df['FY'].unique())}")
print(f"Available fiscal years in embeddings: {sorted(report_embeddings.keys())}")

# Filter out NaN returns
df = df[df['Return_30d'].notna()].reset_index(drop=True)
print(f"Shape after removing NaN returns: {df.shape}")

# Filter out returns that are exactly zero (likely unlisted indices)
df = df[df['Return_30d'] != 0.0].reset_index(drop=True)
print(f"Shape after removing zero returns (unlisted indices): {df.shape}")

# Build sector vocabulary from non-NaN data
sectors = sorted(df["IndexName"].unique())
sector2id = {s: i for i, s in enumerate(sectors)}
print(f"Number of sectors: {len(sectors)}")

# Filter valid FYs
valid_fys = set(report_embeddings.keys())
df = df[df["FY"].isin(valid_fys)].reset_index(drop=True)
print(f"Shape after filtering valid FYs: {df.shape}")

# Verify we have data
if len(df) == 0:
    raise ValueError("No valid data points after filtering!")

# Prepare full dataset
dataset = PolicyReturnDataset(report_embeddings, df, sector2id)

# Cross-validation settings
k_folds = 5
num_epochs = 10
results = {}

# K-Fold Cross Validation
kfold = KFold(n_splits=k_folds, shuffle=True, random_state=42)

print(f"\nStarting {k_folds}-fold cross-validation")

# Store metrics for plotting
train_losses = []
val_losses = []
best_model = None
best_val_loss = float('inf')

# Store trained models and predictions during the original CV loop
trained_models = []
fold_predictions = []
fold_actuals = []

# Replace the training loop section with this improved version:
for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f"\nFOLD {fold+1}/{k_folds}")
    print('-' * 30)
    
    # Set seed for this fold
    set_seeds(42 + fold)
    
    # Sample elements randomly from a given list of ids, no replacement
    train_sampler = SubsetRandomSampler(train_ids)
    val_sampler = SubsetRandomSampler(val_ids)
    
    # Define data loaders for training and validation
    train_loader = DataLoader(dataset, batch_size=min(8, len(train_ids)), 
                             sampler=train_sampler)
    val_loader = DataLoader(dataset, batch_size=min(8, len(val_ids)),
                           sampler=val_sampler)
    
    # Init model, optimizer, loss function
    model = FinBERTRegressor(num_sectors=len(sector2id)).to(device)
    
    # FIXED: Better optimizer settings
    optimizer = torch.optim.AdamW(model.parameters(), lr=5e-5, weight_decay=0.001)
    
    # FIXED: More robust loss function
    loss_fn = torch.nn.HuberLoss(delta=1.0)  # More robust than MSE
    
    # FIXED: Add learning rate scheduler
    scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
        optimizer, mode='min', factor=0.5, patience=2
    )
    
    # Lists to track per-epoch losses
    fold_train_losses = []
    fold_val_losses = []
    
    # Training loop
    for epoch in range(num_epochs):
        # Training phase
        model.train()
        train_loss = 0.0
        for report_emb, sector_id, targets in train_loader:
            report_emb = report_emb.to(device)
            sector_id = sector_id.to(device)
            targets = targets.to(device)

            # Forward pass
            preds = model(report_emb, sector_id)
            loss = loss_fn(preds, targets)
            
            # Backward pass and optimize
            optimizer.zero_grad()
            loss.backward()
            
            # FIXED: Add gradient clipping for stability
            torch.nn.utils.clip_grad_norm_(model.parameters(), max_norm=1.0)
            
            optimizer.step()
            
            train_loss += loss.item()
            
        # Calculate average losses
        train_loss = train_loss / len(train_loader)
        fold_train_losses.append(train_loss)
        
        # Validation phase
        model.eval()
        val_loss = 0.0
        with torch.no_grad():
            for report_emb, sector_id, targets in val_loader:
                report_emb = report_emb.to(device)
                sector_id = sector_id.to(device)
                targets = targets.to(device)
                
                # Forward pass
                preds = model(report_emb, sector_id)
                loss = loss_fn(preds, targets)
                
                val_loss += loss.item()
                
        val_loss = val_loss / len(val_loader)
        fold_val_losses.append(val_loss)
        
        # FIXED: Update learning rate based on validation loss
        scheduler.step(val_loss)
        
        # Only print every 2 epochs to reduce noise
        if (epoch + 1) % 2 == 0 or epoch == 0:
            current_lr = optimizer.param_groups[0]['lr']
            print(f"Epoch {epoch+1}/{num_epochs} - Train: {train_loss:.6f}, Val: {val_loss:.6f}, LR: {current_lr:.2e}")
    
    # Store the fold results
    train_losses.append(fold_train_losses)
    val_losses.append(fold_val_losses)
    
    # Keep track of best model
    final_val_loss = fold_val_losses[-1]
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_model = model.state_dict().copy()  # Make a copy!
        print(f"New best model found (val_loss: {best_val_loss:.6f})")

    # SAVE the trained model immediately after training
    torch.save(model.state_dict(), f"policy_return_model_fold_{fold}.pt")
    print(f"Saved model for fold {fold+1}")
    
    # Get validation predictions for this fold (for metrics)
    model.eval()
    fold_preds = []
    fold_acts = []
    
    with torch.no_grad():
        for report_emb, sector_id, targets in val_loader:
            report_emb = report_emb.to(device)
            sector_id = sector_id.to(device)
            targets = targets.to(device)
            
            preds = model(report_emb, sector_id)
            fold_preds.extend(preds.cpu().numpy())
            fold_acts.extend(targets.cpu().numpy())
    
    fold_predictions.extend(fold_preds)
    fold_actuals.extend(fold_acts)

# Calculate and print average results across folds
avg_train_loss = np.mean([losses[-1] for losses in train_losses])
avg_val_loss = np.mean([losses[-1] for losses in val_losses])

print("\nTraining complete!")
print(f"Average final train loss across folds: {avg_train_loss:.6f}")
print(f"Average final validation loss across folds: {avg_val_loss:.6f}")

# Plot cross-validation results
plt.figure(figsize=(10, 6))
epochs = range(1, num_epochs + 1)

# Plot training & validation loss values
for fold in range(k_folds):
    plt.plot(epochs, train_losses[fold], 'b--', alpha=0.3, label=f'Fold {fold+1} Train' if fold==0 else "")
    plt.plot(epochs, val_losses[fold], 'r--', alpha=0.3, label=f'Fold {fold+1} Val' if fold==0 else "")

# Plot average across folds
plt.plot(epochs, np.mean(train_losses, axis=0), 'b-', linewidth=2, label='Avg Train')
plt.plot(epochs, np.mean(val_losses, axis=0), 'r-', linewidth=2, label='Avg Val')

plt.title('Training and Validation Loss')
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.legend()
plt.grid(True)
plt.savefig('cross_validation_results.png', dpi=300)
plt.close()

print("Cross-validation results plot saved to 'cross_validation_results.png'")

# Now use the stored predictions instead of recomputing
all_preds = np.array(fold_predictions)
all_actuals = np.array(fold_actuals)

# Calculate metrics on the ORIGINAL training results
directional_correct = ((all_preds > 0) == (all_actuals > 0)).mean()
print(f"Combined directional accuracy: {directional_correct:.4f} ({directional_correct*100:.2f}%)")

mae = np.mean(np.abs(all_preds - all_actuals))
print(f"Combined MAE: {mae:.6f}")

ss_total = np.sum((all_actuals - np.mean(all_actuals))**2)
ss_residual = np.sum((all_actuals - all_preds)**2)
r2 = 1 - (ss_residual / ss_total)
print(f"Combined R-squared: {r2:.6f}")

# Save the best model and sector mappings
torch.save(best_model, "policy_return_model.pt")
np.save("sector_mapping.npy", sector2id)
print("Best model and sector mapping saved!")
