import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Trend Decay Optimization on 93.065 SOTA ---")
    sub = pd.read_csv('outputs/winning_submission_93.065.csv')
    test = pd.read_csv('dataset/test.csv')
    
    merged = pd.merge(sub, test[['Index', 'timestamp']], on='Index')
    merged['time_minutes'] = merged['timestamp'].apply(parse_minutes)
    
    # We want to apply a decay that starts at 02:15 and reaches its peak at 13:45
    t_start = 135 # 02:15
    t_end = 825   # 13:45
    duration = t_end - t_start
    
    # Let's try 3 different decay strengths: 3%, 7%, 12%
    for strength in [0.03, 0.07, 0.12]:
        df = merged.copy()
        
        # Calculate linear factor: 1.0 at start, (1.0 - strength) at end
        df['decay_factor'] = 1.0 - (strength * (df['time_minutes'] - t_start) / duration)
        
        df['demand'] = df['demand'] * df['decay_factor']
        df['demand'] = df['demand'].clip(0.0, 1.0)
        
        output_path = f"outputs/sota_decay_{int(strength*100)}pct.csv"
        df[['Index', 'demand']].sort_values('Index').to_csv(output_path, index=False)
        print(f"Saved {output_path} | Strength: {strength*100}% | Final Mean: {df['demand'].mean():.4f}")

if __name__ == "__main__":
    main()
