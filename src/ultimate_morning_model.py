import pandas as pd
import numpy as np
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from pathlib import Path
import os

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs")

def parse_minutes(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)

def prefix_slope(group: pd.DataFrame) -> float:
    clean = group.dropna(subset=["demand"])
    if len(clean) < 2: return 0.0
    x = clean["time_minutes"].to_numpy(dtype=float)
    y = clean["demand"].to_numpy(dtype=float)
    x = x - x.mean()
    denominator = float(np.dot(x, x))
    if denominator == 0: return 0.0
    return float(np.dot(x, y - y.mean()) / denominator)

def get_prefix_stats(df: pd.DataFrame, cutoff_minutes: int) -> pd.DataFrame:
    prefix = df[df["time_minutes"] <= cutoff_minutes].copy()
    prefix = prefix.sort_values(["geohash", "time_minutes"])
    
    stats = prefix.groupby("geohash").agg(
        prefix_mean=("demand", "mean"),
        prefix_max=("demand", "max"),
        prefix_last=("demand", "last"),
        prefix_count=("demand", "size")
    )
    stats["prefix_slope"] = prefix.groupby("geohash").apply(prefix_slope, include_groups=False)
    
    # Add geohash static features (RoadType, etc) - just take the first one found
    static = prefix.groupby("geohash")[["RoadType", "NumberofLanes", "LargeVehicles", "Landmarks"]].first()
    return stats.join(static).reset_index()

def main():
    print("Phase 1: Loading and Preprocessing...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    for df in [train, test]:
        df["time_minutes"] = df["timestamp"].map(parse_minutes)
        df["hour"] = df["time_minutes"] // 60
        df["slot_15m"] = df["time_minutes"] // 15

    # 1. Prepare Day 48 Training Set (Simulate 00:00-02:00 morning to predict afternoon)
    print("Phase 2: Simulating Day 48 Training Data...")
    d48 = train[train["day"] == 48]
    d48_morning_stats = get_prefix_stats(d48, 120)
    
    d48_afternoon = d48[d48["time_minutes"] > 120].copy()
    train_df = d48_afternoon.merge(d48_morning_stats, on="geohash", suffixes=('', '_morning'))
    
    # 2. Prepare Day 49 Test Set (Actual 00:00-02:00 morning to predict Day 49 test)
    print("Phase 3: Preparing Day 49 Test Data...")
    d49_morning = train[train["day"] == 49]
    d49_morning_stats = get_prefix_stats(d49_morning, 120)
    
    test_df = test.merge(d49_morning_stats, on="geohash", how="left", suffixes=('', '_morning'))
    
    # Fill missing morning stats for geohashes not seen in morning
    # Fallback to global means
    for col in ["prefix_mean", "prefix_max", "prefix_last", "prefix_slope", "prefix_count"]:
        test_df[col] = test_df[col].fillna(train_df[col].mean())
        
    # 3. Features
    numeric_features = [
        "time_minutes", "hour", "slot_15m", "NumberofLanes", "Temperature",
        "prefix_mean", "prefix_max", "prefix_last", "prefix_slope", "prefix_count"
    ]
    categorical_features = ["RoadType", "LargeVehicles", "Landmarks", "Weather"]
    
    print(f"Phase 4: Training LightGBM on {len(train_df)} rows...")
    
    # Encoding
    X_train = pd.get_dummies(train_df[numeric_features + categorical_features], columns=categorical_features)
    y_train = train_df["demand"]
    
    X_test = pd.get_dummies(test_df[numeric_features + categorical_features], columns=categorical_features)
    
    # Align columns
    for col in X_train.columns:
        if col not in X_test.columns: X_test[col] = 0
    X_test = X_test[X_train.columns]
    
    model = LGBMRegressor(n_estimators=1000, learning_rate=0.03, num_leaves=63, random_state=42, verbose=-1)
    model.fit(X_train, y_train)
    
    print("Phase 5: Predicting and Calibrating...")
    preds = model.predict(X_test)
    
    # Variance Calibration (Targeting 0.1751 std as per report)
    current_std = preds.std()
    current_mean = preds.mean()
    target_std = 0.1751
    scale = target_std / current_std
    
    print(f"Current Pred Std: {current_std:.4f}, Target Std: {target_std:.4f}, Scale: {scale:.4f}")
    calibrated_preds = (preds - current_mean) * scale + current_mean
    calibrated_preds = np.clip(calibrated_preds, 0.0, 1.0)
    
    # 4. Save
    submission = pd.DataFrame({"Index": test["Index"], "demand": calibrated_preds})
    submission.to_csv(OUTPUT_DIR / "ultimate_morning_model.csv", index=False)
    print(f"Saved ultimate_morning_model.csv to {OUTPUT_DIR}")

if __name__ == "__main__":
    main()
