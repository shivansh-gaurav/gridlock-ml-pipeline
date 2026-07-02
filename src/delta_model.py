import pandas as pd
import numpy as np
import geohash
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42

def parse_minutes(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    # Preprocess
    for df in [train, test]:
        df["time_minutes"] = df["timestamp"].map(parse_minutes)
        df["hour"] = df["time_minutes"] // 60
        df["prefix5"] = df["geohash"].str[:5]
        df["prefix4"] = df["geohash"].str[:4]
    
    day48 = train[train["day"] == 48].copy()
    known_day49 = train[train["day"] == 49].copy()
    
    print("Preparing fallback lags...")
    # Map for lag_d48 fallback
    p5_map = day48.groupby(["prefix5", "timestamp"])["demand"].mean().to_dict()
    p4_map = day48.groupby(["prefix4", "timestamp"])["demand"].mean().to_dict()
    exact_map = day48.set_index(["geohash", "timestamp"])["demand"].to_dict()
    
    def get_lag(row):
        val = exact_map.get((row.geohash, row.timestamp), np.nan)
        if pd.isna(val):
            val = p5_map.get((row.prefix5, row.timestamp), np.nan)
        if pd.isna(val):
            val = p4_map.get((row.prefix4, row.timestamp), np.nan)
        return val

    print("Adding lags to known day 49...")
    known_day49["lag_d48"] = known_day49.apply(get_lag, axis=1)
    known_day49["delta"] = known_day49["demand"] - known_day49["lag_d48"]
    
    print("Adding lags to test set...")
    test["lag_d48"] = test.apply(get_lag, axis=1)
    
    # Train delta model
    # We drop rows where we couldn't find a lag (should be rare)
    train_delta = known_day49.dropna(subset=["delta"])
    
    features = ["time_minutes", "hour", "NumberofLanes", "Temperature", "lag_d48"]
    categorical = ["RoadType", "Weather", "LargeVehicles", "Landmarks"]
    
    X_train = pd.get_dummies(train_delta[features + categorical], columns=categorical)
    y_train = train_delta["delta"]
    
    X_test = pd.get_dummies(test[features + categorical], columns=categorical)
    
    # Align
    for col in X_train.columns:
        if col not in X_test.columns: X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    print(f"Training Delta Model on {len(X_train)} rows...")
    
    # 5-fold CV on the Day 49 morning data
    kf = KFold(n_splits=5, shuffle=True, random_state=RANDOM_STATE)
    cv_scores = []
    for train_idx, val_idx in kf.split(X_train):
        X_tr, X_val = X_train.iloc[train_idx], X_train.iloc[val_idx]
        y_tr, y_val = y_train.iloc[train_idx], y_train.iloc[val_idx]
        
        m = LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=31, random_state=RANDOM_STATE, verbose=-1)
        m.fit(X_tr, y_tr)
        
        # We evaluate on demand, not delta
        lag_val = train_delta.iloc[val_idx]["lag_d48"]
        y_true = train_delta.iloc[val_idx]["demand"]
        y_pred = lag_val + m.predict(X_val)
        y_pred = np.clip(y_pred, 0.0, 1.0)
        
        cv_scores.append(r2_score(y_true, y_pred))
        
    print(f"CV R2 Scores: {cv_scores}")
    print(f"Mean CV R2: {np.mean(cv_scores):.6f}")
    
    model = LGBMRegressor(n_estimators=500, learning_rate=0.02, num_leaves=31, random_state=RANDOM_STATE, verbose=-1)
    model.fit(X_train, y_train)
    
    delta_pred = model.predict(X_test)
    
    # Final prediction: lag + delta
    test["demand"] = test["lag_d48"] + delta_pred
    test["demand"] = np.clip(test["demand"], 0.0, 1.0)
    
    # Fallback for any NaN (unlikely but safe)
    global_mean = train["demand"].mean()
    test["demand"] = test["demand"].fillna(global_mean)
    
    output_path = OUTPUT_DIR / "delta_model_v1.csv"
    test[["Index", "demand"]].to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
