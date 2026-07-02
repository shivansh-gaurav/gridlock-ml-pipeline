import os
import math
import random
import numpy as np
import pandas as pd
import geohash
import torch
import torch.nn as nn
from sklearn.preprocessing import LabelEncoder, StandardScaler

os.environ["OMP_NUM_THREADS"] = "1"
TARGET = "demand"
SEQ_LEN = 4

def decode_geohash(value: str) -> tuple[float, float]:
    lat, lon = geohash.decode(str(value))
    return float(lat), float(lon)

def prepare_data(train_df, test_df):
    print("Preparing Dataset...")
    train_df = train_df.copy()
    test_df = test_df.copy()
    train_df["is_test"] = 0
    test_df["is_test"] = 1
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
    
    scaler = StandardScaler()
    full_df[["lat", "lon"]] = scaler.fit_transform(full_df[["lat", "lon"]])
    
    full_df = full_df.sort_values(["geohash", "slot_id"]).reset_index(drop=True)
    
    lag_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    lags = list(range(1, SEQ_LEN + 1)) + [96] 
    
    for col in lag_cols:
        for lag in lags:
            full_df[f"{col}_lag{lag}"] = full_df[col].shift(lag)
            
    for lag in lags:
        mask = full_df["geohash"] != full_df["geohash"].shift(lag)
        for col in lag_cols:
            full_df.loc[mask, f"{col}_lag{lag}"] = np.nan
            
    return full_df

class FourierFeatures(nn.Module):
    def __init__(self, in_features=2, out_features=128, scale=10.0):
        super().__init__()
        torch.manual_seed(42)
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
    def forward(self, x):
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

class SpatialMapper(nn.Module):
    def __init__(self, fourier_dim=128, hidden_dim=64):
        super().__init__()
        self.fourier = FourierFeatures(in_features=2, out_features=fourier_dim)
        self.mlp = nn.Sequential(nn.Linear(fourier_dim, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
    def forward(self, coords):
        return self.mlp(self.fourier(coords))

class TemporalMLP(nn.Module):
    def __init__(self, in_features, hidden_dim=64):
        super().__init__()
        self.mlp = nn.Sequential(nn.Linear(in_features, hidden_dim * 2), nn.GELU(), nn.Linear(hidden_dim * 2, hidden_dim))
    def forward(self, flat_seq):
        return self.mlp(flat_seq)

class NeuralFeatureExtractor(nn.Module):
    def __init__(self, spatial_dim=64, temporal_dim=64, seq_features=25):
        super().__init__()
        self.spatial = SpatialMapper(fourier_dim=128, hidden_dim=spatial_dim)
        self.temporal = TemporalMLP(in_features=seq_features, hidden_dim=temporal_dim)
        self.head = nn.Sequential(
            nn.Linear(spatial_dim + temporal_dim, 64), nn.GELU(),
            nn.Linear(64, 32), nn.GELU(), nn.Linear(32, 1)
        )
    def forward(self, coords, flat_seq):
        s_emb = self.spatial(coords)
        t_emb = self.temporal(flat_seq)
        combined = torch.cat([s_emb, t_emb], dim=-1)
        return combined, self.head(combined)

def get_tensors(df, device):
    coords = df[["lat", "lon"]].to_numpy(dtype=np.float32)
    seq_cols = [TARGET, "hour_sin", "hour_cos", "minute_sin", "minute_cos"]
    lags = [96, 4, 3, 2, 1] 
    
    lag_arrays = []
    for lag in lags:
        cols = [f"{col}_lag{lag}" for col in seq_cols]
        lag_arrays.append(df[cols].fillna(0.0).to_numpy(dtype=np.float32))
        
    seqs = np.hstack(lag_arrays)
    coords_t = torch.from_numpy(coords).to(device)
    seqs_t = torch.from_numpy(seqs).to(device)
    
    if not df[TARGET].isna().all():
        targets_t = torch.from_numpy(df[TARGET].to_numpy(dtype=np.float32)[:, None]).to(device)
        return coords_t, seqs_t, targets_t
    return coords_t, seqs_t, None

def train_pytorch(train_df, device, epochs=8):
    print(f"--- Training PyTorch Feature Extractor ({epochs} epochs) ---")
    coords_t, seqs_t, targets_t = get_tensors(train_df, device)
    
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
        print(f"Epoch {epoch} | Loss: {total_loss/(num_samples/batch_size):.6f}")
    return model

def main():
    torch.manual_seed(42)
    np.random.seed(42)
    device = torch.device("cpu")
    
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    full_df = prepare_data(train, test)
    train_clean = full_df[(full_df["is_test"] == 0)].dropna(subset=[f"{TARGET}_lag96", f"{TARGET}_lag4"]).copy()
    
    model = train_pytorch(train_clean, device, epochs=8)
    
    print("Extracting full global embeddings...")
    # Extract just the 64-dim spatial embeddings (independent of time)
    unique_geos = full_df[['geohash', 'lat', 'lon']].drop_duplicates('geohash')
    coords_t = torch.tensor(unique_geos[['lat', 'lon']].to_numpy(dtype=np.float32)).to(device)
    
    model.eval()
    with torch.no_grad():
        s_emb = model.spatial(coords_t).cpu().numpy()
        
    emb_df = pd.DataFrame(s_emb, columns=[f'fourier_trained_{i}' for i in range(64)])
    emb_df['geohash'] = unique_geos['geohash'].values
    
    output_path = "dataset/trained_fourier.csv"
    emb_df.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
