import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 2: Area-Aware SOTA Boosting ---")
    sota = pd.read_csv('outputs/winning_submission_93.065.csv')
    test = pd.read_csv('dataset/test.csv')
    archetypes = pd.read_csv('dataset/geohash_archetypes.csv')
    
    # Merge everything
    merged = pd.merge(sota, test[['Index', 'geohash', 'timestamp']], on='Index')
    merged = pd.merge(merged, archetypes, on='geohash', how='left')
    
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    t_start = 135 # 02:15
    t_end = 825   # 13:45
    duration = t_end - t_start
    t_norm = (merged['time_minutes'] - t_start) / duration
    
    # Archetype Boosters
    # Core/Busy = City Center (Needs more volume as day progresses)
    # Suburbs = Residential (Needs decay/mean reversion)
    
    # Trial 1: Subtle Area-Aware (Recommended)
    df1 = merged.copy()
    
    # Core/Busy: Apply a 2% GROWTH trend
    core_mask = df1['area_type'].isin(['Core', 'Busy'])
    df1.loc[core_mask, 'demand'] *= (1.0 + 0.02 * t_norm[core_mask])
    
    # Suburbs: Apply a 4% DECAY trend
    suburb_mask = df1['area_type'] == 'Suburbs'
    df1.loc[suburb_mask, 'demand'] *= (1.0 - 0.04 * t_norm[suburb_mask])
    
    df1['demand'] = df1['demand'].clip(0.0, 1.0)
    df1[['Index', 'demand']].to_csv("outputs/booster_area_aware_v1.csv", index=False)
    
    # Trial 2: Aggressive Artery Boost
    df2 = merged.copy()
    # Boost Core by 5%, Decay Suburbs by 6%
    df2.loc[core_mask, 'demand'] *= (1.0 + 0.05 * t_norm[core_mask])
    df2.loc[suburb_mask, 'demand'] *= (1.0 - 0.06 * t_norm[suburb_mask])
    
    df2['demand'] = df2['demand'].clip(0.0, 1.0)
    df2[['Index', 'demand']].to_csv("outputs/booster_area_aware_v2.csv", index=False)
    
    print("Generated 2 Area-Aware Boosters:")
    print("- outputs/booster_area_aware_v1.csv")
    print("- outputs/booster_area_aware_v2.csv")

if __name__ == "__main__":
    main()
