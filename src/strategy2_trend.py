import pandas as pd
import numpy as np
from pathlib import Path

def main():
    train = pd.read_csv("dataset/train.csv")
    
    # Convert timestamp to minutes
    train["time_minutes"] = train["timestamp"].apply(lambda x: int(x.split(":")[0]) * 60 + int(x.split(":")[1]))
    
    # Filter 00:00 to 02:00 (which is <= 120 minutes)
    day48_early = train[(train["day"] == 48) & (train["time_minutes"] <= 120)]
    day49_early = train[(train["day"] == 49) & (train["time_minutes"] <= 120)]
    
    mean_48 = day48_early["demand"].mean()
    mean_49 = day49_early["demand"].mean()
    multiplier = mean_49 / mean_48
    
    print(f"Day 48 early mean: {mean_48:.6f}")
    print(f"Day 49 early mean: {mean_49:.6f}")
    print(f"Global Shift Multiplier: {multiplier:.6f}")
    
    # Apply to best submission
    best_sub = pd.read_csv("outputs/kartik_geometry_optimal_blend.csv")
    best_sub["demand"] = np.clip(best_sub["demand"] * multiplier, 0.0, 1.0)
    
    output_path = "outputs/strategy2_trend_submission.csv"
    best_sub.to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
