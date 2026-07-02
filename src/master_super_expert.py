import os
import math
import random
from pathlib import Path
import warnings

import numpy as np
import pandas as pd
import geohash
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler
import lightgbm as lgb

# CRITICAL to prevent macOS SegFault when mixing PyTorch & LightGBM
os.environ["OMP_NUM_THREADS"] = "1"
warnings.filterwarnings('ignore')

RANDOM_STATE = 42
TARGET = "demand"
SEQ_LEN = 4

def seed_everything(seed=RANDOM_STATE):
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)

def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)

# ==========================================
# 1. DATA PREP
# ==========================================
def prepare_data(train_df, test_df):
    print("Preparing Dataset...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    
    train_df["is_test"] = 0
    test_df["is_test"] = 1
    test_df[TARGET] = np.nan
    
    test_df["original_index"] = test_df["Index"]
    train_df["original_index"] = np.arange(len(train_df)) * -1
    
    full_df = pd.concat([train_df, test_df], ignore_index=True)
    
    full_df["hour"] = full_df["timestamp"].str.split(":").str[0].astype(int)
    full_df["minute"] = full_df["timestamp"].str.split(":").str[1].astype(int)
    coords = pd.DataFrame(full_df["geohash"].apply(decode_geohash).tolist(), index=full_df.index, columns=["lat", "lon"])
    full_df["lat"] = coords["lat"]
    full_df["lon"] = coords["lon"]
    
    full_df["minute_of_day"] = full_df["hour"] * 60 + full_df["minute"]
    full_df["slot_id"] = full_df["day"] * 96 + full_df["minute_of_day"] // 15
    
    full_df["hour_sin"] = np.sin(2 * np.pi * full_df["hour"] / 24)
    full_df["hour_cos"] = np.cos(2 * np.pi * full_df["hour"] / 24)
    full_df["minute_sin"] = np.sin(2 * np.pi * full_df["minute_of_day"] / 1440)
    full_df["minute_cos"] = np.cos(2 * np.pi * full_df["minute_of_day"] / 1440)
    
    full_df["RoadType"] = full_df["RoadType"].fillna("Unknown")
    full_df["Weather"] = full_df["Weather"].fillna("Unknown")
    full_df["Temperature"] = full_df["Temperature"].fillna(25.0)
    
    le_road = LabelEncoder()
    le_weather = LabelEncoder()
    full_df["RoadType_idx"] = le_road.fit_transform(full_df["RoadType"])
    full_df["Weather_idx"] = le_weather.fit_transform(full_df["Weather"])
    
    scaler_coords = StandardScaler()
    full_df[["lat", "lon"]] = scaler_coords.fit_transform(full_df[["lat", "lon"]])
    
    print("Generating Lags (including 24h Oracle)...")
    full_df = full_df.sort_values(["geohash", "slot_id"]).reset_index(drop=True)
    
    lag_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    lags_to_create = list(range(1, SEQ_LEN + 1)) + [96] # 96 slots = 24 hours
    
    for col in lag_cols:
        for lag in lags_to_create:
            full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
    for lag in lags_to_create:
        mask = full_df["geohash"] != full_df["geohash"].shift(lag)
        for col in lag_cols:
            full_df.loc[mask, f"{col}_lag{lag}"] = np.nan
            
    return full_df

# ==========================================
# 2. PYTORCH MODEL
# ==========================================
class FourierFeatures(nn.Module):
    def __init__(self, in_features=2, out_features=128, scale=10.0):
        super().__init__()
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SpatialMapper(nn.Module):
    def __init__(self, fourier_dim=128, hidden_dim=64):
        super().__init__()
        self.fourier = FourierFeatures(in_features=2, out_features=fourier_dim)
        self.mlp = nn.Sequential(
            nn.Linear(fourier_dim, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    def forward(self, coords):
        return self.mlp(self.fourier(coords))

class TemporalMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(in_features, hidden_dim * 2), nn.GELU(),
            nn.Linear(hidden_dim * 2, hidden_dim)
        )
    def forward(self, flat_seq):
        return self.mlp(flat_seq)

class NeuralFeatureExtractor(nn.Module):
    def __init__(self, spatial_dim=64, temporal_dim=64, seq_features=25): # 5 cols * 5 lags (1,2,3,4,96)
        super().__init__()
        self.spatial = SpatialMapper(fourier_dim=128, hidden_dim=spatial_dim)
        self.temporal = TemporalMLP(in_features=seq_features, hidden_dim=temporal_dim)
        self.head = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(),
            nn.Linear(32, 1)
        )
    def forward(self, coords, flat_seq):
        s_emb = self.spatial(coords)
        t_emb = self.temporal(flat_seq)
        combined = torch.cat([s_emb, t_emb], dim=-1)
        return combined, self.head(combined)

def get_tensors_from_df(df, device):
    coords = df[["lat", "lon"]].to_numpy(dtype=np.float32)
    seq_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    lags_to_use = [96] + list(range(SEQ_LEN, 0, -1)) # Order: 96, 4, 3, 2, 1
    
    lag_arrays = []
    for lag in lags_to_use:
        cols = [f"{col}_lag{lag}" for col in seq_cols]
        lag_arrays.append(df[cols].fillna(0.0).to_numpy(dtype=np.float32))
        
    seqs = np.hstack(lag_arrays)
    coords_t = torch.from_numpy(coords).to(device)
    seqs_t = torch.from_numpy(seqs).to(device)
    
    if TARGET in df.columns and not df[TARGET].isna().all():
        targets = df[TARGET].to_numpy(dtype=np.float32)[:, None]
        return coords_t, seqs_t, torch.from_numpy(targets).to(device)
    return coords_t, seqs_t, None

def train_pytorch(train_df, device, epochs=8):
    print(f"--- Training PyTorch Neural Feature Extractor ({epochs} epochs) ---")
    coords_t, seqs_t, targets_t = get_tensors_from_df(train_df, device)
    
    model = NeuralFeatureExtractor(seq_features=seqs_t.shape[1]).to(device)
    optimizer = torch.optim.AdamW(model.parameters(), lr=2e-3, weight_decay=1e-4)
    criterion = nn.MSELoss()
    
    batch_size = 512
    num_samples = len(coords_t)
    indices = np.arange(num_samples)
    
    model.train()
    for epoch in range(1, epochs + 1):
        np.random.shuffle(indices)
        total_loss = 0.0
        for start_idx in range(0, num_samples, batch_size):
            batch_idx = indices[start_idx:start_idx + batch_size]
            optimizer.zero_grad()
            _, preds = model(coords_t[batch_idx], seqs_t[batch_idx])
            loss = criterion(preds, targets_t[batch_idx])
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
        print(f"PyTorch Epoch {epoch} | Loss: {total_loss/(num_samples/batch_size):.6f}")
    return model

@torch.no_grad()
def extract_embeddings(model, df, device):
    model.eval()
    coords_t, seqs_t, _ = get_tensors_from_df(df, device)
    emb, _ = model(coords_t, seqs_t)
    return emb.cpu().numpy()

# ==========================================
# 3. MASTER PIPELINE
# ==========================================
def main():
    seed_everything(RANDOM_STATE)
    device = torch.device("cpu")
    
    train_raw = pd.read_csv("dataset/train.csv")
    test_raw = pd.read_csv("dataset/test.csv")
    
    full_df = prepare_data(train_raw, test_raw)
    
    train_full = full_df[full_df["is_test"] == 0].copy()
    test_full = full_df[full_df["is_test"] == 1].copy()
    
    # Require Oracle for stability
    train_clean = train_full.dropna(subset=[f"{TARGET}_lag96", f"{TARGET}_lag4"]).copy()
    
    pt_model = train_pytorch(train_clean, device, epochs=8)
    
    print("\n--- Extracting Stable Global Embeddings ---")
    train_emb = extract_embeddings(pt_model, train_clean, device)
    
    base_features_cols = ["Temperature", "NumberofLanes", "RoadType_idx", "Weather_idx", 
                          "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    X_train_base = train_clean[base_features_cols].values
    X_train_full = np.hstack([X_train_base, train_emb])
    y_train = train_clean[TARGET].values
    
    print("\n--- Training Master LGBM Model (Huber) ---")
    lgb_model = lgb.LGBMRegressor(objective='huber', n_estimators=1000, learning_rate=0.03, num_leaves=127, verbose=-1, random_state=RANDOM_STATE)
    lgb_model.fit(X_train_full, y_train)
    
    print("\n--- Executing Recursive SOTA Rollout Inference ---")
    test_slots = sorted(test_full["slot_id"].unique())
    dynamic_df = full_df.copy().set_index(["geohash", "slot_id"])
    predictions = []
    
    day48_hourly_mean = train_raw[train_raw['day'] == 48].copy()
    day48_hourly_mean["hour"] = day48_hourly_mean["timestamp"].str.split(":").str[0].astype(int)
    day48_hourly_mean = day48_hourly_mean.groupby("hour")["demand"].mean()
    
    for slot in test_slots:
        current_mask = (test_full["slot_id"] == slot)
        slot_df = test_full[current_mask].copy()
        if slot_df.empty: continue
            
        for lag in [1, 2, 3, 4, 96]:
            lag_slot = slot - lag
            slot_df[f"{TARGET}_lag{lag}"] = slot_df["geohash"].apply(
                lambda gh: dynamic_df.at[(gh, lag_slot), TARGET] if (gh, lag_slot) in dynamic_df.index else 0.0
            )
            for col in ["hour_sin", "hour_cos", "minute_sin", "minute_cos"]:
                 slot_df[f"{col}_lag{lag}"] = slot_df[f"{col}_lag{lag}"].fillna(0.0)
                 
        X_test_slot_emb = extract_embeddings(pt_model, slot_df, device)
        X_test_slot_base = slot_df[base_features_cols].values
        X_test_slot_full = np.hstack([X_test_slot_base, X_test_slot_emb])
        
        preds = lgb_model.predict(X_test_slot_full)
        
        # Thermostat Stabilization
        hour = (slot % 96) // 4
        if hour in day48_hourly_mean.index:
            current_mean = np.mean(preds)
            baseline_mean = day48_hourly_mean[hour]
            if current_mean > (baseline_mean * 1.15):
                preds *= (baseline_mean * 1.15) / current_mean
                
        preds = np.clip(preds, 0.0, 1.0)
        
        for i, gh in enumerate(slot_df["geohash"]):
            dynamic_df.at[(gh, slot), TARGET] = preds[i]
            
        for i, orig_idx in enumerate(slot_df["original_index"]):
            predictions.append({
                "Index": int(orig_idx), "demand": preds[i], 
                "Temperature": slot_df.iloc[i]["Temperature"], "RoadType": slot_df.iloc[i]["RoadType"]
            })
            
    sub = pd.DataFrame(predictions).sort_values("Index")
    
    print("\n--- Applying Thermo-Spike Non-Linear Optimizer ---")
    extreme_heat_mask = sub["Temperature"] > 40.0
    artery_mask = sub["RoadType"].isin(["Highway", "Street"])
    if (extreme_heat_mask & artery_mask).sum() > 0:
        sub.loc[extreme_heat_mask & artery_mask, "demand"] **= 0.70
        
    peak_mask = sub["demand"] > 0.40
    if peak_mask.sum() > 0:
        sub.loc[peak_mask, "demand"] **= 0.93
        
    sub["demand"] = sub["demand"].clip(0.0, 1.0)
    
    output_path = "outputs/master_super_expert.csv"
    sub[['Index', 'demand']].to_csv(output_path, index=False)
    print(f"Saved {output_path}. Final Std: {sub['demand'].std():.6f}")

if __name__ == "__main__":
    main()
