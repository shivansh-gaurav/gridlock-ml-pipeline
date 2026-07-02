import pandas as pd
import numpy as np
import geohash
from pathlib import Path
from lightgbm import LGBMRegressor
from sklearn.metrics import r2_score
from sklearn.model_selection import KFold
import joblib

DATA_DIR = Path("dataset")
OUTPUT_DIR = Path("outputs")
RANDOM_STATE = 42

def parse_minutes(value: str) -> int:
    hour, minute = str(value).split(":")
    return int(hour) * 60 + int(minute)

def add_time_features(df: pd.DataFrame) -> pd.DataFrame:
    df = df.copy()
    df["time_minutes"] = df["timestamp"].map(parse_minutes)
    df["hour"] = df["time_minutes"] // 60
    df["slot_15m"] = df["time_minutes"] // 15
    df["time_sin"] = np.sin(2 * np.pi * df["time_minutes"] / 1440)
    df["time_cos"] = np.cos(2 * np.pi * df["time_minutes"] / 1440)
    return df

def get_neighbors(geo: str):
    try:
        return geohash.neighbors(geo)
    except:
        return []

def main():
    print("Loading data...")
    train = pd.read_csv(DATA_DIR / "train.csv")
    test = pd.read_csv(DATA_DIR / "test.csv")
    
    train = add_time_features(train)
    test = add_time_features(test)
    
    day48 = train[train["day"] == 48].copy()
    known_day49 = train[train["day"] == 49].copy()
    
    print("Preparing Day 48 lag maps...")
    # Exact lag
    exact_lag_map = day48.set_index(["geohash", "timestamp"])["demand"].to_dict()
    
    # Prefix lags
    day48["prefix5"] = day48["geohash"].str[:5]
    day48["prefix4"] = day48["geohash"].str[:4]
    p5_lag_map = day48.groupby(["prefix5", "timestamp"])["demand"].mean().to_dict()
    p4_lag_map = day48.groupby(["prefix4", "timestamp"])["demand"].mean().to_dict()
    
    # Spatial lag
    # Pre-calculate neighbors for all unique geohashes
    unique_geos = pd.concat([train["geohash"], test["geohash"]]).unique()
    geo_to_neighbors = {g: get_neighbors(g) for g in unique_geos}
    
    print("Preparing Day 49 morning stats...")
    morning_stats = known_day49.groupby("geohash").agg(
        morning_mean=("demand", "mean"),
        morning_last=("demand", "last"),
        morning_count=("demand", "size")
    )
    
    def enrich(df: pd.DataFrame) -> pd.DataFrame:
        df = df.copy()
        df["prefix5"] = df["geohash"].str[:5]
        df["prefix4"] = df["geohash"].str[:4]
        
        print("  Adding lag features...")
        # Exact lag
        exact_lag_df = day48[["geohash", "timestamp", "demand"]].rename(columns={"demand": "lag_d48_exact"})
        df = df.merge(exact_lag_df, on=["geohash", "timestamp"], how="left")
        
        # Prefix 5 lag
        p5_lag_df = pd.DataFrame(p5_lag_map.items(), columns=["key", "lag_d48_p5"])
        p5_lag_df[["prefix5", "timestamp"]] = pd.DataFrame(p5_lag_df["key"].tolist(), index=p5_lag_df.index)
        df = df.merge(p5_lag_df.drop(columns="key"), on=["prefix5", "timestamp"], how="left")
        
        # Prefix 4 lag
        p4_lag_df = pd.DataFrame(p4_lag_map.items(), columns=["key", "lag_d48_p4"])
        p4_lag_df[["prefix4", "timestamp"]] = pd.DataFrame(p4_lag_df["key"].tolist(), index=p4_lag_df.index)
        df = df.merge(p4_lag_df.drop(columns="key"), on=["prefix4", "timestamp"], how="left")
        
        # Best lag (fallback order)
        df["lag_d48_best"] = df["lag_d48_exact"].fillna(df["lag_d48_p5"]).fillna(df["lag_d48_p4"])
        
        print("  Adding spatial lag...")
        # To optimize spatial lag, we need a neighbors dataframe
        neighbor_data = []
        for g, ns in geo_to_neighbors.items():
            for n in ns:
                neighbor_data.append({"geohash": g, "neighbor": n})
        neighbor_df = pd.DataFrame(neighbor_data)
        
        # Join neighbors with day48 demand
        neighbor_demand = neighbor_df.merge(exact_lag_df.rename(columns={"geohash": "neighbor", "lag_d48_exact": "neighbor_demand"}), on="neighbor")
        spatial_lag_df = neighbor_demand.groupby(["geohash", "timestamp"])["neighbor_demand"].mean().reset_index().rename(columns={"neighbor_demand": "lag_d48_spatial"})
        
        df = df.merge(spatial_lag_df, on=["geohash", "timestamp"], how="left")
        
        print("  Adding morning stats...")
        df = df.merge(morning_stats, on="geohash", how="left")
        
        return df

    print("Enriching test set...")
    test_enriched = enrich(test)
    print("Enriching known day 49...")
    known_day49_enriched = enrich(known_day49)
    
    # We will train on Day 49 morning to predict the rest of Day 49?
    # No, that's too little data. We train on Day 48 AND Day 49 morning.
    # But Day 48 doesn't have "morning stats" from its own morning.
    # We can calculate them for Day 48 too.
    
    print("Preparing Day 48 morning stats...")
    morning_48 = day48[day48["time_minutes"] <= 120].groupby("geohash").agg(
        morning_mean=("demand", "mean"),
        morning_last=("demand", "last"),
        morning_count=("demand", "size")
    )
    
    print("Enriching day 48...")
    # For day 48, lag features would come from day 47 (which we don't have).
    # This is a problem. 
    # Let's use Day 48 as the ONLY source of lag features for Day 49.
    # To train a model, we can use Day 49 morning as the target, and Day 48 as the features.
    
    # Let's try to train on the 0-2 AM window of Day 49.
    X_train = known_day49_enriched.dropna(subset=["lag_d48_best"])
    y_train = X_train["demand"]
    
    features = [
        "time_minutes", "hour", "slot_15m", "time_sin", "time_cos",
        "NumberofLanes", "Temperature",
        "lag_d48_exact", "lag_d48_p5", "lag_d48_p4", "lag_d48_best", "lag_d48_spatial",
        "morning_mean", "morning_last", "morning_count"
    ]
    # Add dummies for categorical
    X_train_f = pd.get_dummies(X_train[features + ["RoadType", "Weather", "LargeVehicles"]], drop_first=True)
    X_test_f = pd.get_dummies(test_enriched[features + ["RoadType", "Weather", "LargeVehicles"]], drop_first=True)
    
    # Align
    for col in X_train_f.columns:
        if col not in X_test_f.columns: X_test_f[col] = 0
    X_test_f = X_test_f[X_train_f.columns]
    
    print(f"Training on {len(X_train_f)} rows...")
    model = LGBMRegressor(n_estimators=1000, learning_rate=0.01, num_leaves=31, random_state=RANDOM_STATE, verbose=-1)
    model.fit(X_train_f, y_train)
    
    print("\nFeature Importances:")
    importances = pd.Series(model.feature_importances_, index=X_train_f.columns).sort_values(ascending=False)
    print(importances.head(20))
    
    preds = model.predict(X_test_f)
    preds = np.clip(preds, 0.0, 1.0)
    
    submission = pd.DataFrame({"Index": test["Index"], "demand": preds})
    submission.to_csv(OUTPUT_DIR / "mega_model_v1.csv", index=False)
    print("Saved mega_model_v1.csv")

if __name__ == "__main__":
    main()
