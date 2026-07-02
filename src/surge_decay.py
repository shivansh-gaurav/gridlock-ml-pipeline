import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 5: Surge Component Decay ---")
    sub = pd.read_csv('outputs/winning_submission_93.065.csv')
    train = pd.read_csv('dataset/train.csv')
    test = pd.read_csv('dataset/test.csv')
    
    # 1. Map Day 48 Baseline
    d48 = train[train.day == 48][['geohash', 'timestamp', 'demand']].rename(columns={'demand': 'd48'})
    
    # 2. Merge with SOTA
    merged = pd.merge(test[['Index', 'geohash', 'timestamp']], sub, on='Index')
    merged = pd.merge(merged, d48, on=['geohash', 'timestamp'], how='left')
    
    # For geohashes missing in Day 48, assume d48 = global mean of d48
    d48_global_mean = train[train.day == 48]['demand'].mean()
    merged['d48'] = merged['d48'].fillna(d48_global_mean)
    
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    t_start = 135 # 02:15
    t_end = 825   # 13:45
    duration = t_end - t_start
    t_norm = (merged['time_minutes'] - t_start) / duration
    
    # 3. Calculate the Surge
    merged['surge'] = merged['demand'] - merged['d48']
    
    # 4. Apply Decay only to the SURGE
    # We'll try 3 strengths: 20%, 40%, 60% decay of the deviation
    for strength in [0.20, 0.40, 0.60]:
        df = merged.copy()
        
        # Factor: 1.0 at start, (1.0 - strength) at end
        decay_factor = 1.0 - (strength * t_norm)
        
        df['demand'] = df['d48'] + (df['surge'] * decay_factor)
        df['demand'] = df['demand'].clip(0.0, 1.0)
        
        output_path = f"outputs/surge_decay_{int(strength*100)}pct.csv"
        df[['Index', 'demand']].sort_values('Index').to_csv(output_path, index=False)
        print(f"Saved {output_path} | Surge Pullback: {strength*100}%")

if __name__ == "__main__":
    main()
