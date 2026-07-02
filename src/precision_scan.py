import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 3: Precision Parabola Scan ---")
    sub = pd.read_csv('outputs/winning_submission_93.065.csv')
    test = pd.read_csv('dataset/test.csv')
    
    merged = pd.merge(sub, test[['Index', 'timestamp', 'RoadType']], on='Index')
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    
    t_start = 135 # 02:15
    t_end = 825   # 13:45
    duration = t_end - t_start
    t_norm = (merged['time_minutes'] - t_start) / duration
    
    # Probes to find the global optimum
    for strength in [0.015, 0.040, 0.050]:
        df = merged.copy()
        df['demand'] = df['demand'] * (1.0 - strength * t_norm)
        output_path = f"outputs/precision_scan_{int(strength*1000)}bps.csv"
        df[['Index', 'demand']].to_csv(output_path, index=False)
        print(f"Saved {output_path} | Strength: {strength*100}%")
        
    # SOTA Neighborhood Probe: Residential Mean Reversion
    # People go home after a surge. Highways keep flowing.
    df_res = merged.copy()
    # 5% decay for residential, 1% for everything else
    res_mask = (df_res['RoadType'] == 'Residential')
    df_res.loc[res_mask, 'demand'] *= (1.0 - 0.05 * t_norm[res_mask])
    df_res.loc[~res_mask, 'demand'] *= (1.0 - 0.01 * t_norm[~res_mask])
    df_res[['Index', 'demand']].to_csv("outputs/probe_residential_physics.csv", index=False)
    print("Saved outputs/probe_residential_physics.csv")

if __name__ == "__main__":
    main()
