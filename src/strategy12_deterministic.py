import pandas as pd
import numpy as np

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Pure Deterministic Projection ---")
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    train['time_minutes'] = train['timestamp'].apply(parse_minutes)
    
    # Split train into Day 48 and Day 49
    d48 = train[train['day'] == 48].copy()
    d49 = train[train['day'] == 49].copy()
    
    # The morning is from 00:00 to 09:45 (0 to 585 minutes)
    morning_cutoff = 585
    
    d48_morning = d48[d48['time_minutes'] <= morning_cutoff]
    d49_morning = d49[d49['time_minutes'] <= morning_cutoff]
    
    # Calculate Mean Morning Demand per Geohash
    d48_mean = d48_morning.groupby('geohash')['demand'].mean()
    d49_mean = d49_morning.groupby('geohash')['demand'].mean()
    
    # Calculate the Shifts
    # Multiplicative Shift: (D49 / D48)
    # Additive Shift: (D49 - D48)
    
    # To be safe from division by zero, we add a tiny epsilon
    eps = 1e-6
    ratio_shift = (d49_mean + eps) / (d48_mean + eps)
    delta_shift = d49_mean - d48_mean
    
    # Fallback to global means if a geohash wasn't seen in the morning
    global_ratio = (d49_mean.mean() + eps) / (d48_mean.mean() + eps)
    global_delta = d49_mean.mean() - d48_mean.mean()
    
    # Prepare Test Set
    test['time_minutes'] = test['timestamp'].apply(parse_minutes)
    predictions_ratio = []
    predictions_delta = []
    predictions_blend = []
    
    # We need the exact Day 48 demand for the corresponding test timestamps
    d48_dict = d48.set_index(['geohash', 'timestamp'])['demand'].to_dict()
    global_d48_mean = d48['demand'].mean()
    global_d49_morning_mean = d49_morning['demand'].mean()
    
    for _, row in test.iterrows():
        geo = row['geohash']
        ts = row['timestamp']
        idx = row['Index']
        
        # Get the exact value from yesterday
        yesterday_val = d48_dict.get((geo, ts), np.nan)
        
        # Get the shifts
        g_ratio = ratio_shift.get(geo, global_ratio)
        g_delta = delta_shift.get(geo, global_delta)
        
        # If we have yesterday's data, use the shift
        if not pd.isna(yesterday_val):
            pred_r = yesterday_val * g_ratio
            pred_d = yesterday_val + g_delta
            pred_b = (pred_r + pred_d) / 2.0
        else:
            # Fallback for geohashes completely missing from Day 48
            # We use the global morning mean as a baseline proxy
            pred_b = global_d49_morning_mean
        
        predictions_blend.append({'Index': idx, 'demand': np.clip(pred_b, 0.0, 1.0)})
        
    sub_b = pd.DataFrame(predictions_blend).sort_values('Index')
    sub_b.to_csv("outputs/deterministic_blend.csv", index=False)
    
    print(f"Saved Deterministic Blend:")
    print(f"Mean: {sub_b['demand'].mean():.6f} | Std: {sub_b['demand'].std():.6f}")

if __name__ == "__main__":
    main()
