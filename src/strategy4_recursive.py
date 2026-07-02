import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.preprocessing import StandardScaler

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
    
    # Combine to engineer lags
    all_data = pd.concat([train, test], ignore_index=True)
    all_data = all_data.sort_values(['geohash', 'absolute_time']).reset_index(drop=True)
    
    # We will build a dictionary to store the sequence of demand per geohash
    # to do recursive forecasting
    demand_dict = {}
    for _, row in train.iterrows():
        geo = row['geohash']
        t = row['absolute_time']
        if geo not in demand_dict:
            demand_dict[geo] = {}
        demand_dict[geo][t] = row['demand']
        
    # Feature engineering for training
    # We want to predict demand using lag_1, lag_2, lag_3, lag_4 (15, 30, 45, 60 mins)
    train_records = []
    
    for geo, group in train.groupby('geohash'):
        group = group.sort_values('absolute_time')
        times = group['absolute_time'].values
        demands = group['demand'].values
        
        for i in range(len(times)):
            t = times[i]
            d = demands[i]
            
            # Look up lags in dictionary
            lag_1 = demand_dict[geo].get(t - 15, np.nan)
            lag_2 = demand_dict[geo].get(t - 30, np.nan)
            lag_3 = demand_dict[geo].get(t - 45, np.nan)
            lag_4 = demand_dict[geo].get(t - 60, np.nan)
            lag_24h = demand_dict[geo].get(t - 1440, np.nan)
            
            row_data = group.iloc[i].to_dict()
            row_data['lag_1'] = lag_1
            row_data['lag_2'] = lag_2
            row_data['lag_3'] = lag_3
            row_data['lag_4'] = lag_4
            row_data['lag_24h'] = lag_24h
            train_records.append(row_data)
            
    train_df = pd.DataFrame(train_records)
    
    # We need to drop rows where lag_1 is NaN for training
    train_df = train_df.dropna(subset=['lag_1', 'lag_2', 'lag_3', 'lag_4'])
    
    features = ['lag_1', 'lag_2', 'lag_3', 'lag_4', 'time_minutes', 'Temperature', 'NumberofLanes']
    categorical = ['RoadType', 'Weather']
    
    # Dummy encode
    train_encoded = pd.get_dummies(train_df, columns=categorical)
    model_cols = [c for c in train_encoded.columns if c in features or c.startswith('RoadType_') or c.startswith('Weather_')]
    
    # We should add any missing dummies
    # Let's train the model
    X_train = train_encoded[model_cols]
    y_train = train_encoded['demand']
    
    model = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=63, random_state=42)
    model.fit(X_train, y_train)
    
    print("Model trained. Starting recursive prediction on test set...")
    
    # Sort test set chronologically
    test = test.sort_values(['absolute_time', 'geohash']).reset_index(drop=True)
    
    test_encoded = pd.get_dummies(test, columns=categorical)
    # Ensure columns match
    for c in model_cols:
        if c not in test_encoded.columns:
            test_encoded[c] = 0.0
            
    # Explicitly convert lag columns to float
    for c in ['lag_1', 'lag_2', 'lag_3', 'lag_4']:
        if c not in test_encoded.columns:
            test_encoded[c] = 0.0
        test_encoded[c] = test_encoded[c].astype(float)
            
    test_encoded = test_encoded[model_cols + ['Index', 'geohash', 'absolute_time']]
    
    predictions = []
    
    # Get unique timestamps in test
    unique_times = sorted(test['absolute_time'].unique())
    
    # Fallback means
    global_mean = train['demand'].mean()
    geo_means = train.groupby('geohash')['demand'].mean().to_dict()
    
    for t in unique_times:
        current_rows = test_encoded[test_encoded['absolute_time'] == t].copy()
        
        # Build features for these rows
        for i, row in current_rows.iterrows():
            geo = row['geohash']
            
            # Get historical lags from our dictionary
            geo_dict = demand_dict.get(geo, {})
            l1 = geo_dict.get(t - 15, np.nan)
            l2 = geo_dict.get(t - 30, np.nan)
            l3 = geo_dict.get(t - 45, np.nan)
            l4 = geo_dict.get(t - 60, np.nan)
            l24 = geo_dict.get(t - 1440, np.nan)
            
            # Impute missing lags
            geo_mean = geo_means.get(geo, global_mean)
            
            current_rows.at[i, 'lag_1'] = l1 if not pd.isna(l1) else geo_mean
            current_rows.at[i, 'lag_2'] = l2 if not pd.isna(l2) else geo_mean
            current_rows.at[i, 'lag_3'] = l3 if not pd.isna(l3) else geo_mean
            current_rows.at[i, 'lag_4'] = l4 if not pd.isna(l4) else geo_mean
            
        X_test = current_rows[model_cols]
        preds = model.predict(X_test)
        preds = np.clip(preds, 0.0, 1.0)
        
        # Save predictions back into dictionary for future steps
        for i, (idx, row) in enumerate(current_rows.iterrows()):
            geo = row['geohash']
            if geo not in demand_dict:
                demand_dict[geo] = {}
            demand_dict[geo][t] = preds[i]
            
            predictions.append({
                'Index': int(row['Index']),
                'demand': preds[i]
            })
            
    sub = pd.DataFrame(predictions)
    sub = sub.sort_values('Index')
    sub.to_csv("outputs/recursive_ar_submission.csv", index=False)
    print("Saved outputs/recursive_ar_submission.csv")

if __name__ == "__main__":
    main()
