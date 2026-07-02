import os
import math
import pandas as pd
import numpy as np
import geohash
import torch
import torch.nn as nn
from lightgbm import LGBMRegressor

# Configuration
SEQ_LEN = 4
FOURIER_DIM = 64
DEVICE = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

# ==========================================
# 1. Fourier Spatial Mapping (PyTorch)
# ==========================================
class FourierFeatures(nn.Module):
    def __init__(self, in_features=2, out_features=FOURIER_DIM, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def extract_spatial_embeddings(df: pd.DataFrame) -> pd.DataFrame:
    print("Extracting PyTorch Fourier Spatial Embeddings...")
    lats = df['geohash'].apply(lambda x: geohash.decode(x)[0]).values
    lons = df['geohash'].apply(lambda x: geohash.decode(x)[1]).values
    
    # Normalize coords
    lat_norm = (lats - lats.mean()) / (lats.std() + 1e-6)
    lon_norm = (lons - lons.mean()) / (lons.std() + 1e-6)
    
    coords = np.stack([lat_norm, lon_norm], axis=1)
    coords_t = torch.tensor(coords, dtype=torch.float32).to(DEVICE)
    
    fourier = FourierFeatures().to(DEVICE)
    with torch.no_grad():
        embeddings = fourier(coords_t).cpu().numpy()
        
    emb_df = pd.DataFrame(embeddings, columns=[f'fourier_{i}' for i in range(FOURIER_DIM)])
    emb_df['geohash'] = df['geohash'].values
    # Remove duplicates since spatial features are static per geohash
    return emb_df.drop_duplicates('geohash')

# ==========================================
# 2. Asymmetric Huber Objective for LightGBM
# ==========================================
# This forces the model to care more about high-demand "shocks" (traffic jams)
# than about predicting zero-traffic roads perfectly.
def asymmetric_huber(preds, train_data):
    labels = train_data.get_label()
    delta = preds - labels
    # If we under-predict (delta < 0), apply a higher penalty weight
    weights = np.where(delta < 0, 1.5, 1.0) 
    
    # Huber logic
    huber_delta = 0.15 # Transition point
    is_large_error = np.abs(delta) > huber_delta
    
    grad = np.where(is_large_error, weights * huber_delta * np.sign(delta), weights * delta)
    hess = np.where(is_large_error, 0.0, weights)
    return grad, hess

# ==========================================
# 3. Master Pipeline
# ==========================================
def main():
    print("Loading Continuous Dataset...")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    train['time_minutes'] = train['timestamp'].apply(parse_minutes)
    test['time_minutes'] = test['timestamp'].apply(parse_minutes)
    train['absolute_time'] = train['day'] * 24 * 60 + train['time_minutes']
    test['absolute_time'] = test['day'] * 24 * 60 + test['time_minutes']
    
    # 1. Get Neural Spatial Embeddings
    all_geos = pd.DataFrame({'geohash': pd.concat([train['geohash'], test['geohash']]).unique()})
    spatial_embeddings = extract_spatial_embeddings(all_geos)
    
    # 2. Engineer Continuous Lags (No 24-Hour Anchor)
    demand_dict = {}
    for _, row in train.iterrows():
        geo = row['geohash']
        t = row['absolute_time']
        if geo not in demand_dict:
            demand_dict[geo] = {}
        demand_dict[geo][t] = row['demand']
        
    print("Engineering Momentum Features...")
    train_records = []
    
    for geo, group in train.groupby('geohash'):
        group = group.sort_values('absolute_time')
        times = group['absolute_time'].values
        demands = group['demand'].values
        
        for i in range(len(times)):
            t = times[i]
            l1 = demand_dict[geo].get(t - 15, np.nan)
            l2 = demand_dict[geo].get(t - 30, np.nan)
            l3 = demand_dict[geo].get(t - 45, np.nan)
            l4 = demand_dict[geo].get(t - 60, np.nan)
            
            row_data = group.iloc[i].to_dict()
            row_data['lag_1'] = l1
            row_data['lag_2'] = l2
            row_data['lag_3'] = l3
            row_data['lag_4'] = l4
            row_data['velocity'] = l1 - l2 if pd.notna(l1) and pd.notna(l2) else 0.0
            row_data['acceleration'] = (l1 - l2) - (l2 - l3) if pd.notna(l1) and pd.notna(l2) and pd.notna(l3) else 0.0
            
            train_records.append(row_data)
            
    train_df = pd.DataFrame(train_records)
    train_df = train_df.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'lag_4'])
    
    # Merge Spatial Embeddings
    train_df = train_df.merge(spatial_embeddings, on='geohash', how='left')
    
    # Define Features
    fourier_cols = [f'fourier_{i}' for i in range(FOURIER_DIM)]
    numeric_features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'velocity', 'acceleration', 'time_minutes', 'Temperature', 'NumberofLanes']
    categorical = ['RoadType', 'Weather', 'LargeVehicles']
    
    train_encoded = pd.get_dummies(train_df, columns=categorical)
    model_cols = [c for c in train_encoded.columns if c in numeric_features + fourier_cols or c.startswith(tuple([f"{cat}_" for cat in categorical]))]
    
    X_train = train_encoded[model_cols]
    y_train = train_encoded['demand']
    
    print(f"Training Master LGBM Model on {len(X_train)} rows...")
    model = LGBMRegressor(
        n_estimators=750, 
        learning_rate=0.035, 
        num_leaves=63,      # Reduced from 127 to prevent SegFault
        max_depth=6,        # Reduced from 8
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        verbose=-1
    )
    # Using default objective first to establish baseline; custom objective can be tricky with LGBM early stopping
    model.fit(X_train, y_train)
    
    print("Executing Recursive Inference (SOTA Rollout)...")
    test = test.sort_values(['absolute_time', 'geohash']).reset_index(drop=True)
    test = test.merge(spatial_embeddings, on='geohash', how='left')
    test_encoded = pd.get_dummies(test, columns=categorical)
    
    for c in model_cols:
        if c not in test_encoded.columns:
            test_encoded[c] = 0.0
            
    for c in ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'velocity', 'acceleration']:
        if c not in test_encoded.columns:
            test_encoded[c] = 0.0
        test_encoded[c] = test_encoded[c].astype(float)
            
    test_encoded = test_encoded[model_cols + ['Index', 'geohash', 'absolute_time', 'Temperature', 'RoadType']]
    
    unique_times = sorted(test['absolute_time'].unique())
    global_mean = train['demand'].mean()
    geo_means = train.groupby('geohash')['demand'].mean().to_dict()
    
    predictions = []
    
    # SOTA Component: Global Hourly Anchor Baseline
    train['hour'] = train['timestamp'].str.split(':').str[0].astype(int)
    day48_hourly_mean = train[train['day'] == 48].groupby('hour')['demand'].mean()
    
    for t in unique_times:
        current_rows = test_encoded[test_encoded['absolute_time'] == t].copy()
        
        for i, row in current_rows.iterrows():
            geo = row['geohash']
            geo_dict = demand_dict.get(geo, {})
            l1 = geo_dict.get(t - 15, np.nan)
            l2 = geo_dict.get(t - 30, np.nan)
            l3 = geo_dict.get(t - 45, np.nan)
            l4 = geo_dict.get(t - 60, np.nan)
            
            geo_mean = geo_means.get(geo, global_mean)
            
            current_rows.at[i, 'lag_1'] = l1 if pd.notna(l1) else geo_mean
            current_rows.at[i, 'lag_2'] = l2 if pd.notna(l2) else geo_mean
            current_rows.at[i, 'lag_3'] = l3 if pd.notna(l3) else geo_mean
            current_rows.at[i, 'lag_4'] = l4 if pd.notna(l4) else geo_mean
            current_rows.at[i, 'velocity'] = current_rows.at[i, 'lag_1'] - current_rows.at[i, 'lag_2']
            current_rows.at[i, 'acceleration'] = current_rows.at[i, 'velocity'] - (current_rows.at[i, 'lag_2'] - current_rows.at[i, 'lag_3'])
            
        X_test = current_rows[model_cols]
        preds = model.predict(X_test)
        
        # SOTA Component: Hourly Thermostat
        # 15 mins = 1 step. 4 steps per hour.
        hour = (t // 60) % 24
        if hour in day48_hourly_mean.index:
            current_mean = np.mean(preds)
            baseline_mean = day48_hourly_mean[hour]
            if current_mean > (baseline_mean * 1.15):
                # Pull back to exactly 115% of baseline
                correction_factor = (baseline_mean * 1.15) / current_mean
                preds *= correction_factor
                
        preds = np.clip(preds, 0.0, 1.0)
        
        for i, (idx, row) in enumerate(current_rows.iterrows()):
            geo = row['geohash']
            val = preds[i]
            if geo not in demand_dict:
                demand_dict[geo] = {}
            demand_dict[geo][t] = val
            
            # Save extra context for the Peak Optimizer
            predictions.append({
                'Index': int(row['Index']), 
                'demand': val,
                'Temperature': row['Temperature'],
                'RoadType': row['RoadType']
            })
            
    sub = pd.DataFrame(predictions).sort_values('Index')
    
    print("Applying SOTA Thermo-Spike Non-Linear Optimizer...")
    # 1. Hot Artery Spike
    extreme_heat_mask = sub["Temperature"] > 40.0
    artery_mask = sub["RoadType"].isin(["Highway", "Street"]) # SOTA Archetypes 1 and 2
    target_mask = extreme_heat_mask & artery_mask
    if target_mask.sum() > 0:
        sub.loc[target_mask, "demand"] = sub.loc[target_mask, "demand"] ** 0.70
        
    # 2. Variable Global Peak Preservation (0.93 Power Stretch)
    peak_mask = sub["demand"] > 0.40
    if peak_mask.sum() > 0:
        sub.loc[peak_mask, "demand"] = sub.loc[peak_mask, "demand"] ** 0.93
        
    sub["demand"] = sub["demand"].clip(0.0, 1.0)
    
    output_path = "outputs/master_expert_93.csv"
    sub[['Index', 'demand']].to_csv(output_path, index=False)
    
    print(f"Saved {output_path}")
    print(f"Final Model Std: {sub['demand'].std():.6f}")

if __name__ == "__main__":
    main()
