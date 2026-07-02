import pandas as pd
import numpy as np

train = pd.read_csv("dataset/train.csv")
test = pd.read_csv("dataset/test.csv")

print(f"Test size: {len(test)}")

# Does test exactly match Day 48 in train, just shifted by time?
day48 = train[train['day'] == 48].copy()
day48['day'] = 49

# Join test and day48 on geohash and timestamp
test_merged = test.merge(day48[['geohash', 'timestamp', 'demand']], on=['geohash', 'timestamp'], how='left')

# Print how many matched
print(f"Matched with exact Day 48: {test_merged['demand'].notna().sum()} / {len(test)}")

