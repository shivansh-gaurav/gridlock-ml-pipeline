import pandas as pd
import numpy as np

def main():
    df = pd.read_csv('GridLock/dataset/train.csv')
    
    # 1. Calculate the 'Surge Ratio' for each geohash (D49 Morning / D48 Morning)
    d48_morning = df[(df.day == 48) & (df.timestamp.str.startswith(('0:', '1:', '2:')))].groupby('geohash').demand.mean()
    d49_morning = df[(df.day == 49) & (df.timestamp.str.startswith(('0:', '1:', '2:')))].groupby('geohash').demand.mean()
    
    ratio_signal = (d49_morning / (d48_morning + 1e-6)).to_dict()
    
    # 2. Track how this signal correlates with Day 48 demand throughout the day
    d48_full = df[df.day == 48].copy()
    d48_full['morning_ratio'] = d48_full['geohash'].map(ratio_signal)
    d48_full['hour'] = d48_full.timestamp.apply(lambda x: int(x.split(':')[0]))
    
    print("Correlation between Morning Surge Ratio and Demand by Hour (Day 48):")
    results = []
    for h in sorted(d48_full.hour.unique()):
        hour_data = d48_full[d48_full.hour == h].dropna()
        if len(hour_data) > 0:
            corr = hour_data['demand'].corr(hour_data['morning_ratio'])
            results.append({'hour': h, 'correlation': corr})
            print(f"  Hour {h:02d}: {corr:.4f}")

if __name__ == "__main__":
    main()
