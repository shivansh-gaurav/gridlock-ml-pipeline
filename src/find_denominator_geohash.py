import pandas as pd
import numpy as np

def find_denominator(series, max_den=2000, tol=1e-5):
    s = series[(series > 0) & (series < 1)].copy()
    if len(s) == 0:
        return 1
        
    for den in range(1, max_den + 1):
        vals = s * den
        diffs = np.abs(vals - np.round(vals))
        if diffs.max() < tol:
            return den
    return None

def main():
    train = pd.read_csv("dataset/train.csv")
    
    print("Searching per Geohash...")
    grouped = train.groupby('geohash')
    found_count = 0
    total_count = len(grouped)
    
    capacities = []
    
    for geo, group in grouped:
        den = find_denominator(group['demand'], max_den=200000, tol=1e-6)
        if den is not None:
            found_count += 1
            capacities.append(den)
            
    print(f"Found denominators for {found_count} out of {total_count} geohashes.")
    
    if found_count > 0:
        print("Summary of capacities:")
        print(pd.Series(capacities).value_counts().head(10))

if __name__ == "__main__":
    main()
