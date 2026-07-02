import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
import warnings
warnings.filterwarnings('ignore')

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- STEP 2: SOTA LightGBM & Rollout ---")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    fourier = pd.read_csv("dataset/fourier_features.csv")
    
    # Absolute time mapping
    train['time_minutes'] = train['timestamp'].apply(parse_minutes)
    test['time_minutes'] = test['timestamp'].apply(parse_minutes)
    train['absolute_time'] = train['day'] * 24 * 60 + train['time_minutes']
    test['absolute_time'] = test['day'] * 24 * 60 + test['time_minutes']
    
    # The Oracle (24h anchor) requires knowing exactly what happened yesterday
    demand_dict = {}
    for _, row in train.iterrows():
        geo = row['geohash']
        t = row['absolute_time']
        if geo not in demand_dict:
            demand_dict[geo] = {}
        demand_dict[geo][t] = row['demand']
        
    print("Engineering SOTA Features (Lags + Oracle)...")
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
            l24 = demand_dict[geo].get(t - 1440, np.nan) # THE ORACLE
            
            row_data = group.iloc[i].to_dict()
            row_data['lag_1'] = l1
            row_data['lag_2'] = l2
            row_data['lag_3'] = l3
            row_data['lag_4'] = l4
            row_data['lag_24h'] = l24
            row_data['velocity'] = l1 - l2 if pd.notna(l1) and pd.notna(l2) else 0.0
            row_data['acceleration'] = (l1 - l2) - (l2 - l3) if pd.notna(l1) and pd.notna(l2) and pd.notna(l3) else 0.0
            
            train_records.append(row_data)
            
    train_df = pd.DataFrame(train_records)
    # Require Oracle for training to ensure stability
    train_df = train_df.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_24h'])
    
    # Merge PyTorch Fourier Embeddings safely
    train_df = train_df.merge(fourier, on='geohash', how='left')
    
    # Encoding
    categorical = ['RoadType', 'Weather', 'LargeVehicles']
    all_raw = pd.concat([train_df[categorical], test[categorical]])
    all_encoded_cols = pd.get_dummies(all_raw, columns=categorical).columns
    
    train_encoded = pd.get_dummies(train_df, columns=categorical)
    for c in all_encoded_cols:
        if c not in train_encoded.columns:
            train_encoded[c] = 0.0
            
    fourier_cols = [c for c in fourier.columns if c != 'geohash']
    numeric_features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_24h', 'velocity', 'acceleration', 'time_minutes', 'Temperature', 'NumberofLanes']
    
    meta_cols = ['Index', 'geohash', 'day', 'timestamp', 'demand', 'absolute_time', 'Landmarks'] + categorical
    X_train = train_encoded.drop(columns=[c for c in meta_cols if c in train_encoded.columns])
    
    # Store exact columns
    final_model_cols = X_train.columns.tolist()
    y_train = train_encoded['demand']
    
    print(f"Training SOTA LightGBM on {len(X_train)} rows with {len(final_model_cols)} features...")
    model = LGBMRegressor(
        n_estimators=1000, 
        learning_rate=0.03, 
        num_leaves=127,      
        max_depth=8, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        verbose=-1
    )
    model.fit(X_train, y_train)
    
    print("Executing Stable SOTA Rollout Inference...")
    test = test.sort_values(['absolute_time', 'geohash']).reset_index(drop=True)
    test = test.merge(fourier, on='geohash', how='left')
    
    test_encoded = pd.get_dummies(test, columns=categorical)
    test_encoded['RoadType'] = test['RoadType']
    test_encoded['Temperature'] = test['Temperature']
    
    for c in final_model_cols:
        if c not in test_encoded.columns:
            test_encoded[c] = 0.0
            
    unique_times = sorted(test['absolute_time'].unique())
    global_mean = train['demand'].mean()
    geo_means = train.groupby('geohash')['demand'].mean().to_dict()
    
    predictions = []
    
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
            l24 = geo_dict.get(t - 1440, np.nan)
            
            geo_mean = geo_means.get(geo, global_mean)
            
            current_rows.at[i, 'lag_1'] = l1 if pd.notna(l1) else geo_mean
            current_rows.at[i, 'lag_2'] = l2 if pd.notna(l2) else geo_mean
            current_rows.at[i, 'lag_3'] = l3 if pd.notna(l3) else geo_mean
            current_rows.at[i, 'lag_4'] = l4 if pd.notna(l4) else geo_mean
            current_rows.at[i, 'lag_24h'] = l24 if pd.notna(l24) else geo_mean
            
            current_rows.at[i, 'velocity'] = current_rows.at[i, 'lag_1'] - current_rows.at[i, 'lag_2']
            current_rows.at[i, 'acceleration'] = current_rows.at[i, 'velocity'] - (current_rows.at[i, 'lag_2'] - current_rows.at[i, 'lag_3'])
            
        X_test = current_rows[final_model_cols]
        preds = model.predict(X_test)
        
        # SOTA Component: Hourly Thermostat
        hour = (t // 60) % 24
        if hour in day48_hourly_mean.index:
            current_mean = np.mean(preds)
            baseline_mean = day48_hourly_mean[hour]
            if current_mean > (baseline_mean * 1.15):
                correction_factor = (baseline_mean * 1.15) / current_mean
                preds *= correction_factor
                
        preds = np.clip(preds, 0.0, 1.0)
        
        for i, (idx, row) in enumerate(current_rows.iterrows()):
            geo = row['geohash']
            val = preds[i]
            if geo not in demand_dict:
                demand_dict[geo] = {}
            demand_dict[geo][t] = val
            
            predictions.append({
                'Index': int(row['Index']), 
                'demand': val,
                'Temperature': row['Temperature'],
                'RoadType': row['RoadType']
            })
            
    sub = pd.DataFrame(predictions).sort_values('Index')
    
    print("Applying SOTA Thermo-Spike Non-Linear Optimizer...")
    sub['Temperature'] = sub['Temperature'].fillna(25.0)
    extreme_heat_mask = sub["Temperature"] > 40.0
    artery_mask = sub["RoadType"].isin(["Highway", "Street"]) 
    target_mask = extreme_heat_mask & artery_mask
    if target_mask.sum() > 0:
        sub.loc[target_mask, "demand"] = sub.loc[target_mask, "demand"] ** 0.70
        
    peak_mask = sub["demand"] > 0.40
    if peak_mask.sum() > 0:
        sub.loc[peak_mask, "demand"] = sub.loc[peak_mask, "demand"] ** 0.93
        
    sub["demand"] = sub["demand"].clip(0.0, 1.0)
    
    output_path = "outputs/sota_master_fourier.csv"
    sub[['Index', 'demand']].to_csv(output_path, index=False)
    
    print(f"Saved {output_path}")
    print(f"Final Model Std: {sub['demand'].std():.6f}")

if __name__ == "__main__":
    main()
