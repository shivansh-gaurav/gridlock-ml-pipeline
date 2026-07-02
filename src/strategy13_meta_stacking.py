import pandas as pd
import numpy as np
import geohash
from lightgbm import LGBMRegressor
from sklearn.linear_model import Ridge
from sklearn.preprocessing import StandardScaler
from pathlib import Path

# SOTA Experts from the provided Sub folder
SCRIPTS_DIR = Path("Sub/final_sota_submission_93.065/scripts")
DATA_DIR = Path("dataset")

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 1: Training Contextual Experts (Proxies) ---")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    # Preprocess
    for df in [train, test]:
        df['time_minutes'] = df['timestamp'].apply(parse_minutes)
        df['hour'] = df['time_minutes'] // 60
        df['lat'] = df['geohash'].apply(lambda x: geohash.decode(x)[0])
        df['lon'] = df['geohash'].apply(lambda x: geohash.decode(x)[1])
        
    d48 = train[train.day == 48].copy()
    d49_morning = train[train.day == 49].copy()
    
    # Feature engineering for the experts
    def engineer_expert_features(df):
        df = df.copy()
        categorical = ['RoadType', 'Weather', 'LargeVehicles']
        df_encoded = pd.get_dummies(df, columns=categorical)
        return df_encoded
        
    X_d48 = engineer_expert_features(d48)
    X_d49_m = engineer_expert_features(d49_morning)
    X_test = engineer_expert_features(test)
    
    # Align columns
    model_cols = [c for c in X_d48.columns if c not in ['Index', 'geohash', 'day', 'timestamp', 'demand', 'time_minutes', 'Landmarks']]
    for c in model_cols:
        if c not in X_d49_m.columns: X_d49_m[c] = 0
        if c not in X_test.columns: X_test[c] = 0
    
    X_d48 = X_d48[model_cols]
    X_d49_m = X_d49_m[model_cols]
    X_test = X_test[model_cols]
    
    # 1. Train 3 Orthogonal Experts on Day 48
    print("Training Expert V1 (Signal)...")
    v1 = LGBMRegressor(n_estimators=500, learning_rate=0.05, num_leaves=31, random_state=42, verbose=-1)
    v1.fit(X_d48, d48['demand'])
    
    print("Training Expert V5 (Spatial/Complex)...")
    v5 = LGBMRegressor(n_estimators=500, learning_rate=0.03, num_leaves=127, max_depth=8, random_state=42, verbose=-1)
    v5.fit(X_d48, d48['demand'])
    
    print("Training Expert V6 (Shock/Huber)...")
    v6 = LGBMRegressor(objective='huber', n_estimators=500, learning_rate=0.05, num_leaves=63, random_state=42, verbose=-1)
    v6.fit(X_d48, d48['demand'])
    
    # Oracle Expert V4 (Static Day 48)
    d48_map = d48.set_index(['geohash', 'timestamp'])['demand'].to_dict()
    def get_v4(df):
        return df.apply(lambda r: d48_map.get((r.geohash, r.timestamp), d48['demand'].mean()), axis=1)

    # 2. Collect Predictions on Day 49 Morning (The Stacking Set)
    print("\n--- Phase 2: Building the Meta-Learner ---")
    stack_train = pd.DataFrame({
        'v1': v1.predict(X_d49_m),
        'v5': v5.predict(X_d49_m),
        'v6': v6.predict(X_d49_m),
        'v4': get_v4(d49_morning),
        'hour': d49_morning['hour'],
        'lat': d49_morning['lat'],
        'lon': d49_morning['lon'],
        'lanes': d49_morning['NumberofLanes']
    })
    
    # The Meta-Selector learns which model works in which context
    # We use a Ridge model to find the dynamic weights
    meta_features = ['v1', 'v5', 'v6', 'v4', 'hour', 'lat', 'lon', 'lanes']
    meta_selector = Ridge(alpha=1.0)
    meta_selector.fit(stack_train[meta_features], d49_morning['demand'])
    
    print("Meta-Selection Coefficients (Stability check):")
    for feat, coef in zip(meta_features[:4], meta_selector.coef_[:4]):
        print(f"  {feat}: {coef:.4f}")
        
    # 3. Final Prediction on Test Set
    print("\n--- Phase 3: Final Inference ---")
    stack_test = pd.DataFrame({
        'v1': v1.predict(X_test),
        'v5': v5.predict(X_test),
        'v6': v6.predict(X_test),
        'v4': get_v4(test),
        'hour': test['hour'],
        'lat': test['lat'],
        'lon': test['lon'],
        'lanes': test['NumberofLanes']
    })
    
    final_preds = meta_selector.predict(stack_test[meta_features])
    final_preds = np.clip(final_preds, 0.0, 1.0)
    
    # 4. Final Peak Optimization (from SOTA)
    print("Applying SOTA Peak Optimizer...")
    test['demand'] = final_preds
    peak_mask = test['demand'] > 0.40
    test.loc[peak_mask, 'demand'] = test.loc[peak_mask, 'demand'] ** 0.93
    
    output_path = "outputs/meta_expert_stacking.csv"
    test[['Index', 'demand']].to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
