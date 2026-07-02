import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor

def parse_minutes(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)

def main():
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    # Preprocess time
    train['time_minutes'] = train['timestamp'].map(parse_minutes)
    test['time_minutes'] = test['timestamp'].map(parse_minutes)
    
    train['absolute_time'] = train['day'] * 24 * 60 + train['time_minutes']
    test['absolute_time'] = test['day'] * 24 * 60 + test['time_minutes']
    
    # Build a dictionary to store the sequence of demand per geohash
    demand_dict = {}
    for _, row in train.iterrows():
        geo = row['geohash']
        t = row['absolute_time']
        if geo not in demand_dict:
            demand_dict[geo] = {}
        demand_dict[geo][t] = row['demand']
        
    # Build training data
    train_records = []
    for geo, group in train.groupby('geohash'):
        group = group.sort_values('absolute_time')
        times = group['absolute_time'].values
        demands = group['demand'].values
        
        for i in range(len(times)):
            t = times[i]
            lag_1 = demand_dict[geo].get(t - 15, np.nan)
            lag_2 = demand_dict[geo].get(t - 30, np.nan)
            lag_24h = demand_dict[geo].get(t - 1440, np.nan)
            
            row_data = group.iloc[i].to_dict()
            row_data['lag_1'] = lag_1
            row_data['lag_2'] = lag_2
            row_data['lag_24h'] = lag_24h
            train_records.append(row_data)
            
    train_df = pd.DataFrame(train_records)
    train_df = train_df.dropna(subset=['lag_1', 'lag_2', 'lag_24h'])
    
    features = ['lag_1', 'lag_2', 'lag_24h', 'time_minutes', 'Temperature', 'NumberofLanes']
    categorical = ['RoadType', 'Weather']
    
    train_encoded = pd.get_dummies(train_df, columns=categorical)
    model_cols = [c for c in train_encoded.columns if c in features or c.startswith('RoadType_') or c.startswith('Weather_')]
    
    X_train = train_encoded[model_cols]
    y_train = train_encoded['demand']
    
    model = LGBMRegressor(n_estimators=300, learning_rate=0.04, num_leaves=31, random_state=42)
    model.fit(X_train, y_train)
    
    # Load anchor predictions
    anchor = pd.read_csv("outputs/kartik_geometry_optimal_blend.csv")
    anchor_dict = dict(zip(anchor['Index'], anchor['demand']))
    
    test = test.sort_values(['absolute_time', 'geohash']).reset_index(drop=True)
    test_encoded = pd.get_dummies(test, columns=categorical)
    for c in model_cols:
        if c not in test_encoded.columns:
            test_encoded[c] = 0.0
            
    test_encoded = test_encoded[model_cols + ['Index', 'geohash', 'absolute_time']]
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
            l24 = geo_dict.get(t - 1440, np.nan)
            
            geo_mean = geo_means.get(geo, global_mean)
            current_rows.at[i, 'lag_1'] = l1 if not pd.isna(l1) else geo_mean
            current_rows.at[i, 'lag_2'] = l2 if not pd.isna(l2) else geo_mean
            current_rows.at[i, 'lag_24h'] = l24 if not pd.isna(l24) else geo_mean
            
        X_test = current_rows[model_cols]
        preds = model.predict(X_test)
        preds = np.clip(preds, 0.0, 1.0)
        
        for i, (idx, row) in enumerate(current_rows.iterrows()):
            geo = row['geohash']
            index_val = int(row['Index'])
            
            # TEACHER FORCING: Blend the AR prediction with the anchor prediction
            # This stops the error from exploding into the next step
            ar_pred = preds[i]
            anchor_pred = anchor_dict[index_val]
            
            # 30% AR, 70% Anchor
            blended_pred = 0.3 * ar_pred + 0.7 * anchor_pred
            
            if geo not in demand_dict:
                demand_dict[geo] = {}
            demand_dict[geo][t] = blended_pred
            
            predictions.append({
                'Index': index_val,
                'demand': blended_pred
            })
            
    sub = pd.DataFrame(predictions)
    sub = sub.sort_values('Index')
    sub.to_csv("outputs/teacher_forced_ar.csv", index=False)

if __name__ == "__main__":
    main()
