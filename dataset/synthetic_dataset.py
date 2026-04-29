import numpy as np
import pandas as pd
import holidays
import osmnx as ox
import networkx as nx
from osmnx.distance import add_edge_lengths
import yaml

G = ox.graph_from_place("Warsaw, Poland", network_type="drive")
G = add_edge_lengths(G)

nodes = list(G.nodes)

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def load_timestamps(path="timestamps.csv"):
    df = pd.read_csv(path)
    timestamps = pd.to_datetime(df.iloc[:, 0])

    return timestamps.tolist()

def sample_node():
    return np.random.choice(nodes)

def get_latlon(node):
    return G.nodes[node]["y"], G.nodes[node]["x"]
        
def sample_trip():
    orig = sample_node()
    dest = sample_node()

    lat1, lon1 = get_latlon(orig)
    lat2, lon2 = get_latlon(dest)

    return orig, dest, lat1, lon1, lat2, lon2

def compute_distance_km(orig, dest, G):
    try:
        distance_m = nx.shortest_path_length(
            G,
            orig,
            dest,
            weight="length"
        )
        return distance_m / 1000.0
    except nx.NetworkXNoPath:
        return None

def generate_data(timestamps, G):
    rows = []

    for t in timestamps:
        orig, dest, lat1, lon1, lat2, lon2 = sample_trip()

        dist = compute_distance_km(orig, dest, G)

        if dist is None:
            continue

        dist *= np.random.uniform(1.0, 1.15)

        base_speed = np.random.uniform(20, 50)
        travel_time = (dist / base_speed) * 60

        rows.append([
            t,
            lat1, lon1,
            lat2, lon2,
            dist,
            travel_time
        ])

    return pd.DataFrame(rows, columns=[
        "timestamp",
        "start_lat", "start_lon",
        "end_lat", "end_lon",
        "distance_km",
        "travel_time_min"
    ])

def add_time_features(df):
    df["hour"] = df["timestamp"].dt.hour
    df["day_of_week"] = df["timestamp"].dt.dayofweek
    df["is_weekend"] = df["day_of_week"] >= 5
    df["rush_hour"] = df["hour"].isin([7, 8, 9, 16, 17, 18])

    df.loc[df["rush_hour"], "travel_time_min"] *= 1.5
    df.loc[df["is_weekend"], "travel_time_min"] *= 0.85

    return df

def add_weather(df, cfg):
    weather_types = cfg["types"]
    impact = cfg["impact"]
    seasons = cfg["seasons"]

    month_to_season = {}
    for season_name, season_cfg in seasons.items():
        for m in season_cfg["months"]:
            month_to_season[m] = season_name

    weather_col = []
    impact_col = []

    for ts in df["timestamp"]:
        month = ts.month
        season = month_to_season.get(month, "spring")

        probs_dict = seasons[season]["probs"]

        probs = np.array([probs_dict[w] for w in weather_types])
        probs = probs / probs.sum()

        w = np.random.choice(weather_types, p=probs)

        weather_col.append(w)
        impact_col.append(impact[w])

    df["weather"] = weather_col
    df["travel_time_min"] *= impact_col

    return df

def add_holidays(df):
    pl_holidays = holidays.Poland()

    df["is_holiday"] = df["timestamp"].dt.date.apply(lambda d: d in pl_holidays)

    df.loc[df["is_holiday"], "travel_time_min"] *= 0.9
    return df


if __name__ == "__main__":
    # generate data coordinates
    # timestamps = load_timestamps("timestamps.csv")
    # df = generate_data(timestamps, G)
    output_file = "warsaw_osm_network_stream.csv"
    # df.to_csv(output_file, index=False)

    # add features
    df = pd.read_csv(output_file)
    df['timestamp'] = pd.to_datetime(df['timestamp'])
    weather_cfg = load_config('weather_impact.yaml')
    # df = add_time_features(df)
    df = add_weather(df, weather_cfg['weather'])
    df = add_holidays(df)

    df.to_csv(output_file, index=False)

    print(df.sort_values("distance_km", ascending=True).head(20))

