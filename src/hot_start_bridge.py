import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 7: The Hot-Start Persistence Bridge ---")
    sota = pd.read_csv('outputs/winning_submission_93.065.csv')
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    # 1. Get the "Launch Pad" (Real 02:00 AM demand)
    d49_latest = train[(train.day == 49) & (train.timestamp == "2:0")][['geohash', 'demand']].rename(columns={'demand': 'real_0200'})
    
    # 2. Merge with Test Set
    test['time_minutes'] = test['timestamp'].apply(parse_minutes)
    merged = pd.merge(test[['Index', 'geohash', 'time_minutes']], sota, on='Index')
    merged = pd.merge(merged, d49_latest, on='geohash', how='left')
    
    # Fill missing 02:00 values with the global 02:00 mean
    global_0200 = d49_latest['real_0200'].mean()
    merged['real_0200'] = merged['real_0200'].fillna(global_0200)
    
    # 3. Apply the Bridge
    # Test starts at 02:15 AM (135 mins). 
    # We apply the bridge for 60 minutes (until 03:15 AM / 195 mins).
    t_start = 135
    t_bridge_end = 195
    
    def apply_bridge(row):
        t = row['time_minutes']
        sota_val = row['demand']
        real_val = row['real_0200']
        
        if t >= t_bridge_end:
            return sota_val
        
        # Calculate fade factor (1.0 at 02:00, 0.0 at 03:15)
        fade_factor = 1.0 - ((t - 120) / (t_bridge_end - 120))
        
        return (real_val * fade_factor) + (sota_val * (1.0 - fade_factor))

    merged['demand'] = merged.apply(apply_bridge, axis=1)
    merged['demand'] = merged['demand'].clip(0.0, 1.0)
    
    # 4. Final SOTA Peak Stretch (preserving the SOTA's internal logic)
    peak_mask = merged['demand'] > 0.40
    merged.loc[peak_mask, 'demand'] = merged.loc[peak_mask, 'demand'] ** 0.93
    
    output_path = "outputs/SOTA_HOT_START_BRIDGE.csv"
    merged[['Index', 'demand']].sort_values('Index').to_csv(output_path, index=False)
    print(f"Saved {output_path}")

if __name__ == "__main__":
    main()
