import pandas as pd
import numpy as np
import geohash
from lightgbm import LGBMRegressor

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Advanced Knowledge Distillation (Pseudo-Labeling) ---")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    # Load the highest quality target we have
    pseudo_labels = pd.read_csv("outputs/advanced_variance_scaled_95.csv")
    test['demand'] = pseudo_labels['demand']
    
    # Combine
    full_df = pd.concat([train, test], ignore_index=True)
    full_df['time_minutes'] = full_df['timestamp'].apply(parse_minutes)
    full_df['absolute_time'] = full_df['day'] * 24 * 60 + full_df['time_minutes']
    
    full_df['lat'] = full_df['geohash'].apply(lambda x: geohash.decode(x)[0])
    full_df['lon'] = full_df['geohash'].apply(lambda x: geohash.decode(x)[1])
    
    # Features
    demand_dict = {}
    for _, row in full_df.iterrows():
        geo = row['geohash']
        t = row['absolute_time']
        if geo not in demand_dict:
            demand_dict[geo] = {}
        demand_dict[geo][t] = row['demand']
        
    print("Engineering Distillation Features...")
    records = []
    
    for geo, group in full_df.groupby('geohash'):
        group = group.sort_values('absolute_time')
        times = group['absolute_time'].values
        
        for i in range(len(times)):
            t = times[i]
            l1 = demand_dict[geo].get(t - 15, np.nan)
            l2 = demand_dict[geo].get(t - 30, np.nan)
            l3 = demand_dict[geo].get(t - 45, np.nan)
            l4 = demand_dict[geo].get(t - 60, np.nan)
            l24 = demand_dict[geo].get(t - 1440, np.nan)
            
            row_data = group.iloc[i].to_dict()
            row_data['lag_1'] = l1
            row_data['lag_2'] = l2
            row_data['lag_3'] = l3
            row_data['lag_4'] = l4
            row_data['lag_24h'] = l24
            row_data['velocity'] = l1 - l2 if pd.notna(l1) and pd.notna(l2) else 0.0
            
            records.append(row_data)
            
    df = pd.DataFrame(records)
    
    categorical = ['RoadType', 'Weather', 'LargeVehicles']
    numeric_features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'lag_24h', 'velocity', 'time_minutes', 'Temperature', 'NumberofLanes', 'lat', 'lon']
    
    df_encoded = pd.get_dummies(df, columns=categorical)
    model_cols = [c for c in df_encoded.columns if c in numeric_features or c.startswith(tuple([f"{cat}_" for cat in categorical]))]
    
    # Train only on rows that have enough history
    df_clean = df_encoded.dropna(subset=['lag_1', 'lag_24h'])
    
    # We train on BOTH Train and Test (Pseudo-labels)
    # This distills the ensemble logic into a single robust model boundary
    X = df_clean[model_cols]
    y = df_clean['demand']
    
    print(f"Training Distillation Model on {len(X)} rows...")
    model = LGBMRegressor(
        n_estimators=1200, 
        learning_rate=0.02, 
        num_leaves=127, 
        max_depth=8, 
        subsample=0.8, 
        colsample_bytree=0.8, 
        random_state=42, 
        verbose=-1
    )
    model.fit(X, y)
    
    print("Predicting test set...")
    # For prediction, we use the actual test rows
    test_rows = df_encoded[df_encoded['day'] == 49].copy()
    # Filter to only the actual test timestamps (from 10:00 to 23:45? No, the ones in test.csv)
    test_indices = test['Index'].values
    test_final = test_rows[test_rows['Index'].isin(test_indices)].copy()
    
    # We must drop NaNs? No, we impute for test set if any
    test_final = test_final.fillna(test_final.mean(numeric_only=True))
    
    X_test = test_final[model_cols]
    preds = model.predict(X_test)
    preds = np.clip(preds, 0.0, 1.0)
    
    sub = pd.DataFrame({'Index': test_final['Index'], 'demand': preds})
    sub = sub.sort_values('Index')
    
    output_path = "outputs/strategy11_distilled_95.csv"
    sub.to_csv(output_path, index=False)
    
    print(f"Saved {output_path}")
    print(f"Final Model Std: {sub['demand'].std():.6f}")

if __name__ == "__main__":
    main()
