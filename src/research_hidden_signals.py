import pandas as pd
import geohash
import numpy as np

def main():
    df = pd.read_csv('dataset/train.csv')
    
    # 1. Are Highways connected?
    print("--- Highway Topology Check ---")
    highways = df[df.RoadType == 'Highway']['geohash'].unique()
    print(f"Total Highway Geohashes: {len(highways)}")
    
    # Check how many highway geohashes have another highway geohash as a neighbor
    highway_set = set(highways)
    connected_highways = 0
    for h in highways:
        neighbors = geohash.neighbors(h)
        if any(n in highway_set for n in neighbors):
            connected_highways += 1
            
    print(f"Highways connected to other Highways: {connected_highways} ({connected_highways/len(highways)*100:.1f}%)")
    
    # 2. Does Weather move?
    print("\n--- Weather Front Check ---")
    # Let's track 'Rainy' weather
    rainy = df[df.Weather == 'Rainy'].copy()
    rainy['time_minutes'] = rainy['timestamp'].apply(lambda x: int(x.split(":")[0])*60 + int(x.split(":")[1]))
    
    # Get the center of mass of the rain for each timestamp
    def get_center_of_mass(group):
        lats = group['geohash'].apply(lambda x: geohash.decode(x)[0])
        lons = group['geohash'].apply(lambda x: geohash.decode(x)[1])
        return pd.Series({'lat': lats.mean(), 'lon': lons.mean(), 'count': len(group)})
        
    rain_centers = rainy.groupby(['day', 'time_minutes']).apply(get_center_of_mass, include_groups=False).reset_index()
    
    # Check if the center moves continuously
    d48_rain = rain_centers[rain_centers.day == 48].sort_values('time_minutes')
    
    print("Rain Center of Mass (First 10 intervals of Day 48):")
    print(d48_rain.head(10).to_string(index=False))
    
    # Calculate movement
    d48_rain['lat_diff'] = d48_rain['lat'].diff()
    d48_rain['lon_diff'] = d48_rain['lon'].diff()
    
    print("\nAverage Rain Movement per 15 mins:")
    print(f"Lat: {d48_rain['lat_diff'].mean():.6f}, Lon: {d48_rain['lon_diff'].mean():.6f}")

if __name__ == "__main__":
    main()
