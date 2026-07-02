import pandas as pd
import numpy as np
import geohash
import math
import torch
import torch.nn as nn

# Must NOT import lightgbm here to prevent macOS segfault

FOURIER_DIM = 64
DEVICE = torch.device('cpu') # CPU is perfectly fine for this small mapping

class FourierFeatures(nn.Module):
    def __init__(self, in_features=2, out_features=FOURIER_DIM, scale=10.0):
        super().__init__()
        # Random but fixed projection matrix
        torch.manual_seed(42)
        self.B = nn.Parameter(torch.randn(in_features, out_features // 2) * scale, requires_grad=False)
        
    def forward(self, x: torch.Tensor) -> torch.Tensor:
        x_proj = (2.0 * math.pi * x) @ self.B
        return torch.cat([torch.sin(x_proj), torch.cos(x_proj)], dim=-1)

def main():
    print("--- STEP 1: PyTorch Fourier Spatial Extraction ---")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    unique_geos = pd.Series(pd.concat([train['geohash'], test['geohash']]).unique(), name='geohash')
    
    lats = unique_geos.apply(lambda x: geohash.decode(x)[0]).values
    lons = unique_geos.apply(lambda x: geohash.decode(x)[1]).values
    
    # Normalize
    lat_norm = (lats - lats.mean()) / (lats.std() + 1e-6)
    lon_norm = (lons - lons.mean()) / (lons.std() + 1e-6)
    
    coords = np.stack([lat_norm, lon_norm], axis=1)
    coords_t = torch.tensor(coords, dtype=torch.float32)
    
    fourier = FourierFeatures()
    with torch.no_grad():
        embeddings = fourier(coords_t).numpy()
        
    cols = [f'fourier_{i}' for i in range(FOURIER_DIM)]
    emb_df = pd.DataFrame(embeddings, columns=cols)
    emb_df['geohash'] = unique_geos.values
    
    output_path = "dataset/fourier_features.csv"
    emb_df.to_csv(output_path, index=False)
    print(f"Successfully saved {FOURIER_DIM} spatial dimensions to {output_path}")

if __name__ == "__main__":
    main()
