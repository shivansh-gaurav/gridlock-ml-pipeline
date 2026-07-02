import pandas as pd

def main():
    train = pd.read_csv("dataset/train.csv")
    
    # Get Day 48 and Day 49
    d48 = train[train['day'] == 48].copy()
    d49 = train[train['day'] == 49].copy()
    
    # Merge them to compare the exact same geohash and timestamp
    merged = d49.merge(d48[['geohash', 'timestamp', 'demand']], 
                       on=['geohash', 'timestamp'], 
                       suffixes=('_d49', '_d48'))
    
    geos = merged['geohash'].unique()[:5]
    
    for geo in geos:
        print(f"\n--- Geohash: {geo} ---")
        sub = merged[merged['geohash'] == geo]
        for _, row in sub.iterrows():
            print(f"Time: {row['timestamp']:>5} | D48: {row['demand_d48']:.6f} | D49: {row['demand_d49']:.6f} | Diff: {row['demand_d49'] - row['demand_d48']:+.6f} | Ratio: {row['demand_d49'] / (row['demand_d48'] + 1e-8):.6f}")

if __name__ == "__main__":
    main()
