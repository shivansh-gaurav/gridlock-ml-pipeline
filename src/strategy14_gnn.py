import os
import pandas as pd
import numpy as np
import geohash
import torch
import torch.nn as nn
import torch.nn.functional as F
from torch_geometric.nn import GCNConv
from torch_geometric.data import Data

# SOTA Constants
SEQ_LEN = 4
HIDDEN_DIM = 64
DEVICE = torch.device('cpu')

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 1: Building the Traffic Graph Adjacency ---")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    unique_geos = sorted(list(set(train['geohash'].unique()) | set(test['geohash'].unique())))
    geo_to_idx = {g: i for i, g in enumerate(unique_geos)}
    N_NODES = len(unique_geos)
    
    # Build Edges (Physical Adjacency)
    edge_index_list = []
    for geo in unique_geos:
        u = geo_to_idx[geo]
        neighbors = geohash.neighbors(geo)
        for n in neighbors:
            if n in geo_to_idx:
                v = geo_to_idx[n]
                edge_index_list.append([u, v])
                
    edge_index = torch.tensor(edge_index_list, dtype=torch.long).t().contiguous()
    
    print(f"Graph constructed with {N_NODES} nodes and {edge_index.shape[1]} edges.")
    
    # 2. Build the Time-Series Cube
    # Day 48: 0 to 95. Day 49 Morning: 96 to 103.
    train['time_minutes'] = train['timestamp'].apply(parse_minutes)
    train['absolute_step'] = (train['day'] - 48) * 96 + train['time_minutes'] // 15
    
    # Dense matrix: [Nodes, TimeSteps]
    MAX_STEP = 103
    data_matrix = np.zeros((N_NODES, MAX_STEP + 1), dtype=np.float32)
    
    # Fill with global mean for stability
    global_mean = train['demand'].mean()
    data_matrix += global_mean
    
    for _, row in train.iterrows():
        g_idx = geo_to_idx[row['geohash']]
        step = row['absolute_step']
        if step <= MAX_STEP:
            data_matrix[g_idx, step] = row['demand']
            
    # 3. Define the Spatio-Temporal GCN
    class STGNN(nn.Module):
        def __init__(self, in_channels, hidden_channels, out_channels):
            super().__init__()
            self.conv1 = GCNConv(in_channels, hidden_channels)
            self.conv2 = GCNConv(hidden_channels, hidden_channels)
            self.fc = nn.Linear(hidden_channels, out_channels)
            
        def forward(self, x, edge_index):
            # x: [N_NODES, SEQ_LEN]
            x = F.relu(self.conv1(x, edge_index))
            x = F.relu(self.conv2(x, edge_index))
            return self.fc(x)

    model = STGNN(in_channels=SEQ_LEN, hidden_channels=HIDDEN_DIM, out_channels=1).to(DEVICE)
    optimizer = torch.optim.Adam(model.parameters(), lr=0.01)
    
    # 4. Training (on Day 48 sequences)
    print("Training GNN on Day 48 flow patterns...")
    model.train()
    for epoch in range(100):
        total_loss = 0
        # Iterate through time steps
        for t in range(MAX_STEP - SEQ_LEN):
            # Input: [N_NODES, 4] -> Target: [N_NODES, 1]
            x_t = torch.tensor(data_matrix[:, t : t+SEQ_LEN], dtype=torch.float32).to(DEVICE)
            y_t = torch.tensor(data_matrix[:, t+SEQ_LEN : t+SEQ_LEN+1], dtype=torch.float32).to(DEVICE)
            
            optimizer.zero_grad()
            out = model(x_t, edge_index)
            loss = F.mse_loss(out, y_t)
            loss.backward()
            optimizer.step()
            total_loss += loss.item()
            
        if epoch % 20 == 0:
            print(f"Epoch {epoch} | Loss: {total_loss/MAX_STEP:.6f}")
            
    # 5. Recursive Rollout for Test Set
    print("\n--- Phase 2: Recursive GNN Inference ---")
    model.eval()
    
    test_steps = sorted((test['day'] - 48) * 96 + test['timestamp'].apply(parse_minutes) // 15)
    test_min_step = min(test_steps)
    test_max_step = max(test_steps)
    
    # Extend data matrix for future
    future_matrix = np.zeros((N_NODES, test_max_step + 1), dtype=np.float32)
    future_matrix[:, :MAX_STEP+1] = data_matrix
    
    predictions = []
    
    # We predict step-by-step
    for t in range(test_min_step, test_max_step + 1):
        with torch.no_grad():
            x_input = torch.tensor(future_matrix[:, t-SEQ_LEN : t], dtype=torch.float32).to(DEVICE)
            out = model(x_input, edge_index)
            pred_vals = out.squeeze().cpu().numpy()
            pred_vals = np.clip(pred_vals, 0.0, 1.0)
            
            # Update future matrix for next recursive step
            future_matrix[:, t] = pred_vals
            
    # Map back to test Index
    test['abs_step'] = (test['day'] - 48) * 96 + test['timestamp'].apply(parse_minutes) // 15
    for _, row in test.iterrows():
        g_idx = geo_to_idx[row['geohash']]
        step = row['abs_step']
        predictions.append({'Index': int(row['Index']), 'demand': float(future_matrix[g_idx, step])})
        
    sub = pd.DataFrame(predictions).sort_values('Index')
    
    # SOTA Peak Optimizer
    print("Applying SOTA Peak Optimizer...")
    peak_mask = sub['demand'] > 0.40
    sub.loc[peak_mask, 'demand'] = sub.loc[peak_mask, 'demand'] ** 0.93
    
    output_path = "outputs/st_gnn_flow_model.csv"
    sub.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
