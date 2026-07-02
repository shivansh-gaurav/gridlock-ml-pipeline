import pandas as pd
import numpy as np
import geohash
from lightgbm import LGBMRegressor

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("Loading Continuous Dataset...")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    train['time_minutes'] = train['timestamp'].apply(parse_minutes)
    test['time_minutes'] = test['timestamp'].apply(parse_minutes)
    
    train['absolute_time'] = train['day'] * 24 * 60 + train['time_minutes']
    test['absolute_time'] = test['day'] * 24 * 60 + test['time_minutes']
    
    train['prefix5'] = train['geohash'].str[:5]
    test['prefix5'] = test['geohash'].str[:5]
    
    # Pre-calculate Highway connections
    highways = train[train.RoadType == 'Highway']['geohash'].unique()
    highway_set = set(highways)
    highway_neighbors = {}
    for h in highways:
        ns = geohash.neighbors(h)
        highway_neighbors[h] = [n for n in ns if n in highway_set]
        
    demand_dict = {}
    for _, row in train.iterrows():
        geo = row['geohash']
        t = row['absolute_time']
        if geo not in demand_dict:
            demand_dict[geo] = {}
        demand_dict[geo][t] = row['demand']
        
    print("Engineering Pure Momentum Features...")
    train_records = []
    
    for geo, group in train.groupby('geohash'):
        group = group.sort_values('absolute_time')
        times = group['absolute_time'].values
        demands = group['demand'].values
        is_highway = group['RoadType'].iloc[0] == 'Highway'
        
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
            
            # Momentum Velocity
            row_data['velocity'] = l1 - l2 if not pd.isna(l1) and not pd.isna(l2) else 0.0
            
            # Highway Topology Feature
            hw_momentum = 0.0
            if is_highway and geo in highway_neighbors and len(highway_neighbors[geo]) > 0:
                hw_vals = []
                for n in highway_neighbors[geo]:
                    v1 = demand_dict.get(n, {}).get(t - 15, np.nan)
                    v2 = demand_dict.get(n, {}).get(t - 30, np.nan)
                    if not pd.isna(v1) and not pd.isna(v2):
                        hw_vals.append(v1 - v2)
                if hw_vals:
                    hw_momentum = np.mean(hw_vals)
            row_data['highway_velocity'] = hw_momentum
            
            train_records.append(row_data)
            
    train_df = pd.DataFrame(train_records)
    train_df = train_df.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'lag_4'])
    
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'velocity', 'highway_velocity', 'time_minutes', 'Temperature', 'NumberofLanes']
    categorical = ['RoadType', 'Weather', 'LargeVehicles', 'prefix5']
    
    train_encoded = pd.get_dummies(train_df, columns=categorical)
    model_cols = [c for c in train_encoded.columns if c in features or c.startswith(tuple([f"{cat}_" for cat in categorical]))]
    
    X_train = train_encoded[model_cols]
    y_train = train_encoded['demand']
    
    print(f"Training Continuous Model on {len(X_train)} rows...")
    model = LGBMRegressor(n_estimators=750, learning_rate=0.035, num_leaves=63, max_depth=7, subsample=0.85, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    print("Executing Recursive Inference with Neighborhood Smoothing...")
    test = test.sort_values(['absolute_time', 'geohash']).reset_index(drop=True)
    test_encoded = pd.get_dummies(test, columns=categorical)
    
    # Retain original columns needed for inference logic
    test_encoded['RoadType'] = test['RoadType']
    test_encoded['prefix5'] = test['prefix5']
    
    for c in model_cols:
        if c not in test_encoded.columns:
            test_encoded[c] = 0.0
            
    for c in ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'velocity', 'highway_velocity']:
        if c not in test_encoded.columns:
            test_encoded[c] = 0.0
        test_encoded[c] = test_encoded[c].astype(float)
            
    test_encoded = test_encoded[model_cols + ['Index', 'geohash', 'prefix5', 'absolute_time', 'RoadType']]
    
    unique_times = sorted(test['absolute_time'].unique())
    global_mean = train['demand'].mean()
    geo_means = train.groupby('geohash')['demand'].mean().to_dict()
    
    predictions = []
    
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
            l1 = l1 if not pd.isna(l1) else geo_mean
            l2 = l2 if not pd.isna(l2) else geo_mean
            
            current_rows.at[i, 'lag_1'] = l1 
            current_rows.at[i, 'lag_2'] = l2 
            current_rows.at[i, 'lag_3'] = l3 if not pd.isna(l3) else geo_mean
            current_rows.at[i, 'lag_4'] = l4 if not pd.isna(l4) else geo_mean
            current_rows.at[i, 'velocity'] = l1 - l2
            
            # Highway Topology
            hw_momentum = 0.0
            if row['RoadType'] == 'Highway' and geo in highway_neighbors:
                hw_vals = []
                for n in highway_neighbors[geo]:
                    v1 = demand_dict.get(n, {}).get(t - 15, np.nan)
                    v2 = demand_dict.get(n, {}).get(t - 30, np.nan)
                    if not pd.isna(v1) and not pd.isna(v2):
                        hw_vals.append(v1 - v2)
                if hw_vals:
                    hw_momentum = np.mean(hw_vals)
            current_rows.at[i, 'highway_velocity'] = hw_momentum
            
        X_test = current_rows[model_cols]
        preds = model.predict(X_test)
        
        # NEIGHBORHOOD SMOOTHING (Preventing AR drift)
        current_rows['pred'] = preds
        prefix_means = current_rows.groupby('prefix5')['pred'].transform('mean')
        
        # If a prediction is more than 3x its neighborhood average, pull it back 50%
        # This allows for local spikes but prevents exponential hallucinations
        smoothed_preds = np.where(
            current_rows['pred'] > (prefix_means * 3.0),
            (current_rows['pred'] + prefix_means) / 2.0,
            current_rows['pred']
        )
        smoothed_preds = np.clip(smoothed_preds, 0.0, 1.0)
        
        for i, (idx, row) in enumerate(current_rows.iterrows()):
            geo = row['geohash']
            val = smoothed_preds[i]
            if geo not in demand_dict:
                demand_dict[geo] = {}
            demand_dict[geo][t] = val
            predictions.append({'Index': int(row['Index']), 'demand': val})
            
    sub = pd.DataFrame(predictions).sort_values('Index')
    output_path = "outputs/pure_momentum_model.csv"
    sub.to_csv(output_path, index=False)
    
    print(f"Saved {output_path}")
    print(f"Final Model Std: {sub['demand'].std():.6f}")

if __name__ == "__main__":
    main()
