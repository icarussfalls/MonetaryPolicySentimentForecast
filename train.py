from model import FinBERTRegressor
from dataset import PolicyReturnDataset
import torch
from torch.utils.data import DataLoader, SubsetRandomSampler
import pandas as pd
import numpy as np
from sklearn.model_selection import KFold
import matplotlib.pyplot as plt
import shap

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

for fold, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    print(f"\nFOLD {fold+1}/{k_folds}")
    print('-' * 30)
    
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
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-5, weight_decay=0.01)
    loss_fn = torch.nn.MSELoss()
    
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
        
        print(f"Epoch {epoch+1}/{num_epochs} - Train loss: {train_loss:.6f}, Val loss: {val_loss:.6f}")
    
    # Store the fold results
    train_losses.append(fold_train_losses)
    val_losses.append(fold_val_losses)
    
    # Keep track of best model
    final_val_loss = fold_val_losses[-1]
    if final_val_loss < best_val_loss:
        best_val_loss = final_val_loss
        best_model = model.state_dict()
        print(f"New best model found (val_loss: {best_val_loss:.6f})")

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

# Save models for all folds
print("\nSaving models for all folds...")
for fold_idx in range(k_folds):
    # Initialize a new model
    fold_model = FinBERTRegressor(num_sectors=len(sector2id)).to(device)
    
    # Get the ids for this fold
    _, (train_ids, val_ids) = list(enumerate(kfold.split(dataset)))[fold_idx]
    
    # Define data loaders
    train_sampler = SubsetRandomSampler(train_ids)
    train_loader = DataLoader(dataset, batch_size=min(8, len(train_ids)), sampler=train_sampler)
    
    # Train the model
    optimizer = torch.optim.AdamW(fold_model.parameters(), lr=2e-5)
    for epoch in range(num_epochs):
        fold_model.train()
        for report_emb, sector_id, targets in train_loader:
            report_emb, sector_id, targets = report_emb.to(device), sector_id.to(device), targets.to(device)
            optimizer.zero_grad()
            preds = fold_model(report_emb, sector_id)
            loss = torch.nn.MSELoss()(preds, targets)
            loss.backward()
            optimizer.step()
    
    # Save the model
    torch.save(fold_model.state_dict(), f"policy_return_model_fold_{fold_idx}.pt")
    print(f"Saved model for fold {fold_idx+1}")

# Also save the best model
torch.save(best_model, "policy_return_model.pt")

# Save sector mappings
np.save("sector_mapping.npy", sector2id)
print("Sector mapping saved for prediction!")

print("\nCalculating combined directional accuracy across all folds...")

# Initialize containers for all predictions and actual values
all_preds = []
all_actuals = []

# Run predictions on validation sets for each fold
for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    # Load the trained model for this fold
    fold_model = FinBERTRegressor(num_sectors=len(sector2id)).to(device)
    fold_model.load_state_dict(torch.load(f"policy_return_model_fold_{fold_idx}.pt", map_location=device))
    fold_model.eval()
    
    # Create validation loader for this fold
    val_sampler = SubsetRandomSampler(val_ids)
    val_loader = DataLoader(dataset, batch_size=min(8, len(val_ids)), sampler=val_sampler)
    
    # Get predictions
    with torch.no_grad():
        for report_emb, sector_id, targets in val_loader:
            report_emb = report_emb.to(device)
            sector_id = sector_id.to(device)
            targets = targets.to(device)
            
            # Get predictions
            preds = fold_model(report_emb, sector_id)
            
            # Store predictions and actual values
            all_preds.extend(preds.cpu().numpy())
            all_actuals.extend(targets.cpu().numpy())

# Convert to numpy arrays
all_preds = np.array(all_preds)
all_actuals = np.array(all_actuals)

# Calculate directional accuracy (positive vs negative)
directional_correct = ((all_preds > 0) == (all_actuals > 0)).mean()
print(f"Combined directional accuracy across all folds: {directional_correct:.4f} ({directional_correct*100:.2f}%)")

# Calculate Mean Absolute Error (MAE)
mae = np.mean(np.abs(all_preds - all_actuals))
print(f"Combined MAE across all folds: {mae:.6f}")

# Calculate R-squared
ss_total = np.sum((all_actuals - np.mean(all_actuals))**2)
ss_residual = np.sum((all_actuals - all_preds)**2)
r2 = 1 - (ss_residual / ss_total)
print(f"Combined R-squared across all folds: {r2:.6f}")

# Calculate sector-specific metrics
sector_metrics = {}
for fold_idx, (train_ids, val_ids) in enumerate(kfold.split(dataset)):
    # Load the trained model for this fold
    fold_model = FinBERTRegressor(num_sectors=len(sector2id)).to(device)
    fold_model.load_state_dict(torch.load(f"policy_return_model_fold_{fold_idx}.pt", map_location=device))
    fold_model.eval()
    
    # Create validation loader for this fold
    val_sampler = SubsetRandomSampler(val_ids)
    val_loader = DataLoader(dataset, batch_size=1, sampler=val_sampler)
    
    # Get predictions by sector
    with torch.no_grad():
        for report_emb, sector_id, targets in val_loader:
            report_emb = report_emb.to(device)
            sector_id = sector_id.to(device)
            targets = targets.to(device)
            
            # Get predictions
            preds = fold_model(report_emb, sector_id)
            
            # Get sector name
            sector_name = list(sector2id.keys())[sector_id.item()]
            
            # Initialize sector in the metrics dict if it doesn't exist
            if sector_name not in sector_metrics:
                sector_metrics[sector_name] = {'preds': [], 'actuals': []}
            
            # Store predictions and actual values
            sector_metrics[sector_name]['preds'].append(preds.item())
            sector_metrics[sector_name]['actuals'].append(targets.item())

# Calculate and print sector-specific directional accuracy
print("\nSector-specific directional accuracy:")
print("=====================================")
all_sector_accs = []
for sector_name, data in sector_metrics.items():
    sector_preds = np.array(data['preds'])
    sector_actuals = np.array(data['actuals'])
    
    # Skip sectors with too few samples
    if len(sector_preds) < 5:
        continue
        
    dir_acc = ((sector_preds > 0) == (sector_actuals > 0)).mean()
    mae = np.mean(np.abs(sector_preds - sector_actuals))
    all_sector_accs.append((sector_name, dir_acc, mae, len(sector_preds)))
    
# Sort by directional accuracy
all_sector_accs.sort(key=lambda x: x[1], reverse=True)

# Print in a table format
print(f"{'Sector':<15} {'Dir. Accuracy':<15} {'MAE':<10} {'Samples':<10}")
print("-" * 50)
for sector_name, dir_acc, mae, n_samples in all_sector_accs:
    print(f"{sector_name:<15} {dir_acc*100:>5.2f}%{'':<9} {mae:<10.6f} {n_samples:<10}")


# Calculating combined directional accuracy across all folds...
# Combined directional accuracy across all folds: 0.6316 (63.16%)
# Combined MAE across all folds: 0.056369
# Combined R-squared across all folds: 0.021674

# Sector-specific directional accuracy:
# =====================================
# Sector          Dir. Accuracy   MAE        Samples   
# --------------------------------------------------
# LIFEINSU        85.71%          0.052352   7         
# HYDROPOWER      78.57%          0.077600   14        
# OTHERS          71.43%          0.047259   14        
# MICROFINANCE    71.43%          0.036530   7         
# TRADING         66.67%          0.047060   9         
# FINANCE         64.29%          0.067343   14        
# HOTELS          64.29%          0.046127   14        
# BANKING         64.29%          0.054003   14        
# DEVBANK         61.54%          0.047332   13        
# NONLIFEINSU     57.14%          0.049129   7         
# NEPSE           54.55%          0.049716   22        
# MANUFACTURE     38.46%          0.072826   13
# 
# 
# 
# Sector-specific directional accuracy:
# =====================================
# Sector          Dir. Accuracy   MAE        Samples   
# --------------------------------------------------
# LIFEINSU        85.71%          0.132518   7         
# MANUFACTURE     76.92%          0.065170   13        
# BANKING         71.43%          0.065467   14        
# HYDROPOWER      71.43%          0.077146   14        
# NONLIFEINSU     57.14%          0.053593   7         
# MICROFINANCE    57.14%          0.111719   7         
# TRADING         55.56%          0.083539   9         
# HOTELS          50.00%          0.056879   14        
# NEPSE           50.00%          0.068538   22        
# FINANCE         50.00%          0.143178   14        
# DEVBANK         46.15%          0.102191   13        
# OTHERS          42.86%          0.068712   14       