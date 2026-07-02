import pandas as pd
import numpy as np

def main():
    train = pd.read_csv("dataset/train.csv")
    test = pd.read_csv("dataset/test.csv")
    
    # 1. Is Index just the row number?
    print("Train Index == row number?", (train['Index'] == train.index).all())
    print("Test Index == row number?", (test['Index'] == test.index).all())
    
    # 2. Let's see how the data is actually sorted.
    # We can check if it's sorted by time, geohash, etc.
    train_time_sorted = train.sort_values(['day', 'timestamp', 'geohash'])
    train_geo_sorted = train.sort_values(['geohash', 'day', 'timestamp'])
    
    print("Is train sorted by Time -> Geohash?", (train.index == train_time_sorted.index).all())
    print("Is train sorted by Geohash -> Time?", (train.index == train_geo_sorted.index).all())
    
    # Let's look at the first few rows of train to see the ordering
    print("\nFirst 10 rows of train:")
    print(train[['Index', 'geohash', 'day', 'timestamp']].head(10))
    
    # 3. Correlation of Index with Target
    print("\nCorrelation of Index with Demand (Train):", train['Index'].corr(train['demand']))
    
    # 4. Check if geohashes have a fixed number of rows
    print("\nRow counts per geohash in Train:")
    print(train.groupby('geohash').size().describe())
    
    print("\nRow counts per geohash in Test:")
    print(test.groupby('geohash').size().describe())

    # 5. Let's check for any duplicate rows
    print("\nExact duplicate rows in Train:", train.drop(columns=['Index', 'demand'], errors='ignore').duplicated().sum())
    print("Exact duplicate rows in Test:", test.drop(columns=['Index'], errors='ignore').duplicated().sum())

if __name__ == "__main__":
    main()
