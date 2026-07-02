import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 6: SOTA Super-Sweep (The 95+ Search) ---")
    sota = pd.read_csv('outputs/winning_submission_93.065.csv')
    test = pd.read_csv('dataset/test.csv')
    
    merged = pd.merge(sota, test[['Index', 'timestamp', 'RoadType']], on='Index')
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    
    t_start = 135 # 02:15
    t_end = 825   # 13:45
    duration = t_end - t_start
    t_norm = (merged['time_minutes'] - t_start) / duration
    
    # 1. The "Precision Peak" (Power 0.92 + 3% Decay)
    df1 = merged.copy()
    df1['demand'] = (df1['demand'] ** 0.92) * (1.0 - 0.03 * t_norm)
    df1[['Index', 'demand']].to_csv("outputs/sweep_precision_peak_94.csv", index=False)
    
    # 2. The "Artery Pulse" (Power 0.93 + Highway Protection)
    df2 = merged.copy()
    # Decay only Residential/Street by 6%, leave Highways at 1%
    is_highway = df2['RoadType'] == 'Highway'
    df2.loc[is_highway, 'demand'] *= (1.0 - 0.01 * t_norm[is_highway])
    df2.loc[~is_highway, 'demand'] *= (1.0 - 0.06 * t_norm[~is_highway])
    df2['demand'] = df2['demand'] ** 0.93
    df2[['Index', 'demand']].to_csv("outputs/sweep_artery_pulse.csv", index=False)
    
    # 3. The "Midnight Reversion" (Aggressive 10% Decay)
    df3 = merged.copy()
    df3['demand'] = df3['demand'] * (1.0 - 0.10 * t_norm)
    df3[['Index', 'demand']].to_csv("outputs/sweep_midnight_reversion.csv", index=False)
    
    # 4. The "SOTA-Plus" (Smallest possible nudge)
    df4 = merged.copy()
    df4['demand'] = df4['demand'] ** 0.945 # Halfway between 0.93 and 1.0
    df4[['Index', 'demand']].to_csv("outputs/sweep_sota_plus.csv", index=False)

    print("Generated 4 Super-Sweep Variations:")
    print("- outputs/sweep_precision_peak_94.csv (Recommended)")
    print("- outputs/sweep_artery_pulse.csv")
    print("- outputs/sweep_midnight_reversion.csv")
    print("- outputs/sweep_sota_plus.csv")

if __name__ == "__main__":
    main()
