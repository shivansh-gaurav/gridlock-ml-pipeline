import os
import pandas as pd
import numpy as np
import geohash
import torch
import torch.nn as nn
import torch.optim as optim
from torch.utils.data import Dataset, DataLoader

# Configuration
L = 24  # Lookback window: 6 hours (24 steps of 15 mins)
H = 48  # Horizon: 12 hours (48 steps of 15 mins)
BATCH_SIZE = 512
EPOCHS = 35
LR = 0.001
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

class TrafficMIMO(nn.Module):
    def __init__(self, history_len, static_dim, horizon):
        super(TrafficMIMO, self).__init__()
        
        self.fc1 = nn.Linear(history_len + static_dim, 256)
        self.bn1 = nn.BatchNorm1d(256)
        self.drop1 = nn.Dropout(0.2)
        
        self.fc2 = nn.Linear(256, 256)
        self.bn2 = nn.BatchNorm1d(256)
        self.drop2 = nn.Dropout(0.2)
        
        self.out = nn.Linear(256, horizon)
        self.relu = nn.ReLU()
        
    def forward(self, history, static):
        # history: (B, L)
        # static: (B, static_dim)
        x = torch.cat([history, static], dim=1)
        x = self.relu(self.bn1(self.fc1(x)))
        x = self.drop1(x)
        x = self.relu(self.bn2(self.fc2(x)))
        x = self.drop2(x)
        x = self.out(x)
        return x

class TrafficDataset(Dataset):
    def __init__(self, X_hist, X_static, Y=None):
        self.X_hist = torch.tensor(X_hist, dtype=torch.float32)
        self.X_static = torch.tensor(X_static, dtype=torch.float32)
        self.Y = torch.tensor(Y, dtype=torch.float32) if Y is not None else None
        
    def __len__(self):
        return len(self.X_hist)
        
    def __getitem__(self, idx):
        if self.Y is not None:
            return self.X_hist[idx], self.X_static[idx], self.Y[idx]
        return self.X_hist[idx], self.X_static[idx]

def main():
    print("Loading data...")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    # 1. Absolute Time Steps
    for df in [train, test]:
        df['time_minutes'] = df['timestamp'].apply(parse_minutes)
        df['absolute_step'] = (df['day'] - 48) * 96 + df['time_minutes'] // 15
        
    # Day 48: steps 0 to 95. Day 49 morning: 96 to 103. Total train steps: 104 (0 to 103)
    # Test steps: 105 to 151.
    
    # 2. Build the Dense Demand Matrix
    unique_geos = list(set(train['geohash'].unique()) | set(test['geohash'].unique()))
    geo_to_idx = {g: i for i, g in enumerate(unique_geos)}
    N_GEO = len(unique_geos)
    MAX_TRAIN_STEP = 103
    
    demand_matrix = np.full((N_GEO, MAX_TRAIN_STEP + 1), np.nan, dtype=np.float32)
    
    for _, row in train.iterrows():
        g_idx = geo_to_idx[row['geohash']]
        step = row['absolute_step']
        if step <= MAX_TRAIN_STEP:
            demand_matrix[g_idx, step] = row['demand']
            
    # Impute missing with global mean (mostly for unseen geohashes in train)
    global_mean = train['demand'].mean()
    np.nan_to_num(demand_matrix, copy=False, nan=global_mean)
    
    # 3. Build Static Features
    print("Building Static Features...")
    all_df = pd.concat([train, test]).drop_duplicates('geohash')
    all_df['lat'] = all_df['geohash'].apply(lambda x: geohash.decode(x)[0])
    all_df['lon'] = all_df['geohash'].apply(lambda x: geohash.decode(x)[1])
    
    # One-Hot Encode RoadType
    static_cols = ['lat', 'lon', 'NumberofLanes']
    categorical = ['RoadType', 'LargeVehicles']
    all_encoded = pd.get_dummies(all_df[['geohash'] + static_cols + categorical], columns=categorical)
    
    static_feature_cols = [c for c in all_encoded.columns if c != 'geohash']
    static_matrix = np.zeros((N_GEO, len(static_feature_cols)), dtype=np.float32)
    
    all_encoded.set_index('geohash', inplace=True)
    for g, g_idx in geo_to_idx.items():
        if g in all_encoded.index:
            static_matrix[g_idx] = all_encoded.loc[g].values
            
    # Normalize static features
    static_mean = static_matrix.mean(axis=0)
    static_std = static_matrix.std(axis=0) + 1e-6
    static_matrix = (static_matrix - static_mean) / static_std
    
    # 4. Create Sliding Window Dataset
    print(f"Creating Training Windows (L={L}, H={H})...")
    X_hist_list, X_static_list, Y_list = [], [], []
    
    # Slide from t=0 to t = 103 - L - H + 1
    # Example: L=24, H=48. 103 - 24 - 48 + 1 = 32
    for t in range(MAX_TRAIN_STEP - L - H + 2):
        for g_idx in range(N_GEO):
            # Only use geohashes that actually exist in train
            if demand_matrix[g_idx, t:t+L+H].mean() != global_mean:
                X_hist_list.append(demand_matrix[g_idx, t : t+L])
                X_static_list.append(static_matrix[g_idx])
                Y_list.append(demand_matrix[g_idx, t+L : t+L+H])
                
    train_dataset = TrafficDataset(X_hist_list, X_static_list, Y_list)
    train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    
    # 5. Train Model
    print(f"Training PyTorch MIMO Model on {len(train_dataset)} samples...")
    model = TrafficMIMO(history_len=L, static_dim=len(static_feature_cols), horizon=H).to(DEVICE)
    optimizer = optim.Adam(model.parameters(), lr=LR, weight_decay=1e-5)
    criterion = nn.MSELoss()
    
    model.train()
    for epoch in range(EPOCHS):
        total_loss = 0.0
        for batch_hist, batch_static, batch_y in train_loader:
            batch_hist, batch_static, batch_y = batch_hist.to(DEVICE), batch_static.to(DEVICE), batch_y.to(DEVICE)
            
            optimizer.zero_grad()
            preds = model(batch_hist, batch_static)
            loss = criterion(preds, batch_y)
            loss.backward()
            optimizer.step()
            
            total_loss += loss.item()
            
        if (epoch+1) % 5 == 0:
            print(f"Epoch {epoch+1:02d}/{EPOCHS} | Loss: {total_loss/len(train_loader):.6f}")
            
    # 6. Inference for Test Set
    print("Executing Direct Inference for Horizon...")
    model.eval()
    
    # We want to predict steps 104 to 151 (which is H=48 steps)
    # The last available train step is 103.
    # So our input history is from 103 - L + 1 to 103.
    # Example: L=24, 103 - 24 + 1 = 80. History is [80, 81, ..., 103] (length 24)
    start_hist = MAX_TRAIN_STEP - L + 1
    
    test_hist = demand_matrix[:, start_hist : MAX_TRAIN_STEP + 1]
    
    test_dataset = TrafficDataset(test_hist, static_matrix)
    test_loader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=False)
    
    all_preds = []
    with torch.no_grad():
        for batch_hist, batch_static in test_loader:
            batch_hist, batch_static = batch_hist.to(DEVICE), batch_static.to(DEVICE)
            preds = model(batch_hist, batch_static)
            all_preds.append(preds.cpu().numpy())
            
    all_preds = np.concatenate(all_preds, axis=0) # (N_GEO, H)
    all_preds = np.clip(all_preds, 0.0, 1.0)
    
    # Map predictions to the exact test rows
    predictions = []
    for _, row in test.iterrows():
        g_idx = geo_to_idx[row['geohash']]
        step = row['absolute_step']
        
        # We predicted steps 104 to 151.
        # Index in all_preds is step - 104
        pred_idx = step - 104
        
        if 0 <= pred_idx < H:
            val = all_preds[g_idx, pred_idx]
        else:
            val = global_mean # Fallback if outside horizon (shouldn't happen)
            
        predictions.append({'Index': int(row['Index']), 'demand': float(val)})
        
    sub = pd.DataFrame(predictions).sort_values('Index')
    output_path = "outputs/strategy10_mimo_pytorch.csv"
    sub.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

    # Check Variance
    print(f"Final MIMO Model Std Dev: {sub['demand'].std():.6f}")

if __name__ == "__main__":
    main()
