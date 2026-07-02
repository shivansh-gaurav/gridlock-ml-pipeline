import pandas as pd
import numpy as np
from sklearn.cluster import KMeans
from sklearn.preprocessing import StandardScaler

def parse_minutes(value: str) -> int:
    hour, minute = value.split(":")
    return int(hour) * 60 + int(minute)

def main():
    print("--- Phase 1: Behavioral Archetype Clustering ---")
    train = pd.read_csv("dataset/train.csv")
    
    # 1. Feature Extraction per Geohash (Day 48 is our baseline)
    d48 = train[train.day == 48].copy()
    d48['hour'] = d48.timestamp.apply(lambda x: int(x.split(":")[0]))
    
    # Volume and Volatility
    stats = d48.groupby('geohash')['demand'].agg(['mean', 'std', 'max']).rename(columns={'mean': 'vol', 'std': 'vola', 'max': 'peak'})
    
    # Hourly Profile (Simple morning vs afternoon vs evening)
    morning = d48[d48.hour.between(6, 10)].groupby('geohash')['demand'].mean()
    afternoon = d48[d48.hour.between(12, 16)].groupby('geohash')['demand'].mean()
    evening = d48[d48.hour.between(19, 23)].groupby('geohash')['demand'].mean()
    
    # Day 49 Morning Surge (The 'Mood' of the geohash today)
    d49_m = train[(train.day == 49) & (train.timestamp.str.startswith(('0:', '1:', '2:')))].groupby('geohash')['demand'].mean()
    d48_m = d48[d48.timestamp.str.startswith(('0:', '1:', '2:'))].groupby('geohash')['demand'].mean()
    surge = (d49_m / (d48_m + 1e-6)).rename('surge_factor')
    
    # Combine
    features = pd.concat([stats, morning, afternoon, evening, surge], axis=1).fillna(0)
    features.columns = ['vol', 'vola', 'peak', 'morning_vol', 'afternoon_vol', 'evening_vol', 'surge']
    
    # 2. Clustering into 4 Archetypes
    scaler = StandardScaler()
    X = scaler.fit_transform(features)
    
    kmeans = KMeans(n_clusters=4, random_state=42, n_init=10)
    features['archetype'] = kmeans.fit_predict(X)
    
    # 3. Labeling Archetypes based on Volume
    # Cluster with highest 'vol' is likely "City Center/Main Arteries"
    # Cluster with lowest 'vol' is "Outer/Quiet"
    archetype_map = features.groupby('archetype')['vol'].mean().sort_values().index
    # Map them to names: 0: Quiet, 1: Suburbs, 2: Busy, 3: Core
    rename_dict = {archetype_map[0]: 'Quiet', archetype_map[1]: 'Suburbs', archetype_map[2]: 'Busy', archetype_map[3]: 'Core'}
    features['area_type'] = features['archetype'].map(rename_dict)
    
    print("\nArchetype Summary:")
    summary = features.groupby('area_type').agg({'vol': 'mean', 'surge': 'mean'})
    summary['count'] = features.groupby('area_type').size()
    print(summary)
    
    # Save mapping
    features.index.name = 'geohash'
    print("\nSaved geohash archetypes to dataset/geohash_archetypes.csv")

if __name__ == "__main__":
    main()
