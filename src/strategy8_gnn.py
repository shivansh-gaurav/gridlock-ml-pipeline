import os
import pandas as pd
import numpy as np
import geohash
from math import radians, cos, sin, asin, sqrt

import torch
import torch.nn as nn
import torch.optim as optim
import torch.nn.functional as F

def haversine(lat1, lon1, lat2, lon2):
    R = 6372.8 
    dLat = radians(lat2 - lat1)
    dLon = radians(lon2 - lon1)
    lat1 = radians(lat1)
    lat2 = radians(lat2)
    a = sin(dLat/2)**2 + cos(lat1)*cos(lat2)*sin(dLon/2)**2
    c = 2*asin(sqrt(a))
    return R * c

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    # 1. Map all data
    all_df = pd.concat([train, test])
    all_df['lat'] = all_df['geohash'].apply(lambda x: geohash.decode(x)[0])
    all_df['lon'] = all_df['geohash'].apply(lambda x: geohash.decode(x)[1])
    
    train['time_minutes'] = train['timestamp'].apply(parse_minutes)
    train['absolute_time'] = train['day'] * 24 * 60 + train['time_minutes']
    test['time_minutes'] = test['timestamp'].apply(parse_minutes)
    test['absolute_time'] = test['day'] * 24 * 60 + test['time_minutes']
    
    unique_geos = sorted(all_df['geohash'].unique())
    geo_to_idx = {g: i for i, g in enumerate(unique_geos)}
    N_NODES = len(unique_geos)
    
    # 2. Build the Adjacency Matrix (Graph Edges)
    # We connect nodes if they are within 2.5 km of each other
    geo_coords = all_df.drop_duplicates('geohash')[['geohash', 'lat', 'lon']].set_index('geohash')
    adj_matrix = np.zeros((N_NODES, N_NODES), dtype=np.float32)
    
    print("Building Spatial Graph Adjacency Matrix...")
    for i, g1 in enumerate(unique_geos):
        lat1, lon1 = geo_coords.loc[g1, 'lat'], geo_coords.loc[g1, 'lon']
        for j, g2 in enumerate(unique_geos):
            if i != j:
                lat2, lon2 = geo_coords.loc[g2, 'lat'], geo_coords.loc[g2, 'lon']
                dist = haversine(lat1, lon1, lat2, lon2)
                if dist < 2.5: 
                    # Weighted edge: closer nodes have stronger connections
                    adj_matrix[i, j] = 1.0 / (dist + 0.1) 
                    
    # Normalize adjacency matrix (random walk normalization)
    row_sums = adj_matrix.sum(axis=1, keepdims=True)
    row_sums[row_sums == 0] = 1.0
    adj_matrix = adj_matrix / row_sums
    A_t = torch.tensor(adj_matrix, dtype=torch.float32)
    
    # 3. Build Node Features (Time Series Data)
    train_times = np.sort(train['absolute_time'].unique())
    time_to_idx = {t: i for i, t in enumerate(train_times)}
    
    node_data = np.zeros((len(train_times), N_NODES, 1), dtype=np.float32)
    global_mean = train['demand'].mean()
    node_data += global_mean 
    
    for _, row in train.iterrows():
        t_idx = time_to_idx[row['absolute_time']]
        g_idx = geo_to_idx[row['geohash']]
        node_data[t_idx, g_idx, 0] = row['demand']
        
    # Sequences (Look back 4 steps to predict 1)
    SEQ_LEN = 4
    X, y = [], []
    for i in range(len(train_times) - SEQ_LEN):
        X.append(node_data[i : i + SEQ_LEN]) # [SEQ_LEN, N_NODES, 1]
        y.append(node_data[i + SEQ_LEN])     # [N_NODES, 1]
        
    X_t = torch.tensor(np.array(X)).permute(0, 2, 1, 3).squeeze(3) # [Batch, N_NODES, SEQ_LEN]
    y_t = torch.tensor(np.array(y)).squeeze(2)                     # [Batch, N_NODES]
    
    # 4. Spatio-Temporal Graph Convolutional Network (ST-GCN)
    class STGCN(nn.Module):
        def __init__(self, seq_len):
            super().__init__()
            # Temporal convolution (extracting trends per node)
            self.temp_conv = nn.Conv1d(in_channels=N_NODES, out_channels=N_NODES, kernel_size=3, padding=1, groups=N_NODES)
            # Graph spatial message passing (exchanging data across roads)
            self.graph_weight = nn.Linear(seq_len, 16)
            self.output_layer = nn.Linear(16, 1)
            
        def forward(self, x, A):
            # x shape: [Batch, N_NODES, SEQ_LEN]
            x_temp = F.relu(self.temp_conv(x))
            
            # Graph Convolution: AXW
            # 1. Feature transformation (W)
            h = F.relu(self.graph_weight(x_temp)) # [Batch, N_NODES, 16]
            
            # 2. Neighborhood aggregation (AX)
            # A is [N_NODES, N_NODES], h is [Batch, N_NODES, 16]
            h_graph = torch.einsum('vw, bwc -> bvc', A, h) 
            
            # 3. Output prediction
            out = self.output_layer(h_graph).squeeze(2) # [Batch, N_NODES]
            return out

    model = STGCN(SEQ_LEN)
    optimizer = optim.Adam(model.parameters(), lr=0.01)
    criterion = nn.MSELoss()
    
    print(f"Training GNN with {N_NODES} graph nodes...")
    model.train()
    for epoch in range(300):
        optimizer.zero_grad()
        pred = model(X_t, A_t)
        loss = criterion(pred, y_t)
        loss.backward()
        optimizer.step()
        if epoch % 50 == 0:
            print(f"Epoch {epoch:03d} | Graph MSE Loss: {loss.item():.6f}")
            
    # 5. Inference with Teacher Anchor
    print("\nRunning autoregressive GNN inference on Test Set...")
    model.eval()
    
    test_times = np.sort(test['absolute_time'].unique())
    current_seq = torch.tensor(node_data[-SEQ_LEN:], dtype=torch.float32).permute(1, 0, 2).squeeze(2).unsqueeze(0) # [1, N_NODES, SEQ_LEN]
    
    anchor_df = pd.read_csv("outputs/final_master_blend.csv")
    anchor_df = anchor_df.rename(columns={'demand': 'demand_anchor'})
    test_anchored = test.merge(anchor_df[['Index', 'demand_anchor']], on="Index")
    
    anchor_nodes = np.zeros((len(test_times), N_NODES), dtype=np.float32) + global_mean
    test_time_to_idx = {t: i for i, t in enumerate(test_times)}
    for _, row in test_anchored.iterrows():
        t_idx = test_time_to_idx[row['absolute_time']]
        g_idx = geo_to_idx[row['geohash']]
        anchor_nodes[t_idx, g_idx] = row['demand_anchor']
        
    preds = []
    for t_idx, t in enumerate(test_times):
        with torch.no_grad():
            pred_step = model(current_seq, A_t).squeeze(0).numpy() # [N_NODES]
            
        anchor_step = anchor_nodes[t_idx]
        
        # Blend: 30% GNN Graph Flow + 70% Stable Teacher
        blended_step = 0.3 * pred_step + 0.7 * anchor_step
        blended_step = np.clip(blended_step, 0.0, 1.0)
        
        current_test_rows = test[test['absolute_time'] == t]
        for _, row in current_test_rows.iterrows():
            g_idx = geo_to_idx[row['geohash']]
            preds.append({'Index': int(row['Index']), 'demand': blended_step[g_idx]})
            
        blended_t = torch.tensor(blended_step, dtype=torch.float32).unsqueeze(0).unsqueeze(2) # [1, N_NODES, 1]
        current_seq = torch.cat([current_seq[:, :, 1:], blended_t], dim=2)
        
    sub = pd.DataFrame(preds).sort_values('Index')
    output_path = "outputs/strategy8_gnn_anchored.csv"
    sub.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

    # Analyze Variance Shift
    try:
        best_file = "outputs/final_master_blend.csv"
        best = pd.read_csv(best_file)["demand"].to_numpy(dtype=float)
        new = sub['demand'].to_numpy(dtype=float)
        
        ESTIMATED_SST = 1280.9313328482735
        CURRENT_BEST_SCORE = 91.80212  
        E_BEST = 1.0 - (CURRENT_BEST_SCORE / 100.0)
        
        d = new - best
        sum_d2 = np.sum(d * d)
        q = sum_d2 / ESTIMATED_SST
        worst_case_score = max(0.0, 100.0 * (1.0 - (E_BEST + q)))
        
        print("\n--- Geometric Variance Analysis ---")
        print(f"Distance squared (Sum D^2): {sum_d2:.4f}")
        print(f"Variance shift (q): {q * 100:.4f}%")
        print(f"If 100% wrong, score drops to: {worst_case_score:.4f}")
        
    except FileNotFoundError:
        pass

if __name__ == "__main__":
    main()
