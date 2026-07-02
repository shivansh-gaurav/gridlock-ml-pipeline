import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 3: The Super Refiner (Mean + Variance Optimization) ---")
    sota = pd.read_csv('outputs/winning_submission_93.065.csv')
    test = pd.read_csv('dataset/test.csv')
    archetypes = pd.read_csv('dataset/geohash_archetypes.csv')
    
    merged = pd.merge(sota, test[['Index', 'geohash', 'timestamp']], on='Index')
    merged = pd.merge(merged, archetypes, on='geohash', how='left')
    
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    t_norm = (merged['time_minutes'] - 135) / (825 - 135)
    
    # 1. Apply Differentiated Decay (Fixing the Mean)
    # Core/Busy = 1.5% decay, Suburbs = 5.0% decay
    core_mask = merged['area_type'].isin(['Core', 'Busy'])
    suburb_mask = merged['area_type'] == 'Suburbs'
    
    merged.loc[core_mask, 'demand'] *= (1.0 - 0.015 * t_norm[core_mask])
    merged.loc[suburb_mask, 'demand'] *= (1.0 - 0.050 * t_norm[suburb_mask])
    
    # 2. Apply Variance Stretching (Fixing the Peaks)
    curr_mean = merged['demand'].mean()
    curr_std = merged['demand'].std()
    TARGET_STD = 0.175101
    scale = TARGET_STD / curr_std
    
    print(f"Pre-Stretch Std: {curr_std:.6f} | Stretch Scale: {scale:.6f}")
    
    # Stretch around the mean
    merged['demand'] = (merged['demand'] - curr_mean) * scale + curr_mean
    merged['demand'] = merged['demand'].clip(0.0, 1.0)
    
    output_path = "outputs/SUPER_REFINER_95_V1.csv"
    merged[['Index', 'demand']].sort_values('Index').to_csv(output_path, index=False)
    print(f"Saved {output_path} | Final Std: {merged['demand'].std():.6f}")

if __name__ == "__main__":
    main()
