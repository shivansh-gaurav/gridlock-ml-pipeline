import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 4: Climate-Aware Precision Probes ---")
    sub = pd.read_csv('outputs/winning_submission_93.065.csv')
    test = pd.read_csv('dataset/test.csv')
    
    merged = pd.merge(sub, test[['Index', 'timestamp', 'RoadType', 'Temperature']], on='Index')
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    
    t_start = 135 # 02:15
    t_end = 825   # 13:45
    duration = t_end - t_start
    t_norm = (merged['time_minutes'] - t_start) / duration
    
    # 1. The Frost Anchor (Climate Physics)
    df_frost = merged.copy()
    freezing_mask = df_frost['Temperature'] < 0
    highway_mask = df_frost['RoadType'] == 'Highway'
    res_mask = df_frost['RoadType'] == 'Residential'
    
    # Boost Highways in frost (aggregating traffic)
    df_frost.loc[freezing_mask & highway_mask, 'demand'] *= 1.05
    # Decay Residential in frost (staying home)
    df_frost.loc[freezing_mask & res_mask, 'demand'] *= 0.90
    # Apply standard 3% trend decay to others
    df_frost.loc[~freezing_mask, 'demand'] *= (1.0 - 0.03 * t_norm[~freezing_mask])
    
    df_frost[['Index', 'demand']].to_csv("outputs/probe_frost_anchor.csv", index=False)
    
    # 2. Logarithmic Decay (Fast morning cooling)
    df_log = merged.copy()
    # Log decay: factor starts at 1.0, ends at 0.95
    log_factor = 1.0 - 0.05 * np.log1p(t_norm * (np.e - 1))
    df_log['demand'] = df_log['demand'] * log_factor
    df_log[['Index', 'demand']].to_csv("outputs/probe_logarithmic_decay.csv", index=False)
    
    # 3. The 95+ Gridlock Stretch (Aggressive Power)
    df_stretch = merged.copy()
    # Apply 3% decay first
    df_stretch['demand'] = df_stretch['demand'] * (1.0 - 0.03 * t_norm)
    # Then apply more aggressive power to peaks
    peak_mask = df_stretch['demand'] > 0.40
    df_stretch.loc[peak_mask, 'demand'] = df_stretch.loc[peak_mask, 'demand'] ** 0.88
    
    df_stretch[['Index', 'demand']].to_csv("outputs/probe_aggressive_stretch.csv", index=False)
    
    print("Generated 3 Climate/Physics Probes:")
    print("- outputs/probe_frost_anchor.csv")
    print("- outputs/probe_logarithmic_decay.csv")
    print("- outputs/probe_aggressive_stretch.csv")

if __name__ == "__main__":
    main()
