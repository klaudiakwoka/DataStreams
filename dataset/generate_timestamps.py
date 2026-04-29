import yaml
import random
from datetime import datetime, timedelta
import pandas as pd

def load_config(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)

def daterange(start_date, end_date):
    current = start_date
    while current <= end_date:
        yield current
        current += timedelta(days=1)

def normalize(weights):
    total = sum(weights)
    return [w / total for w in weights]

def weighted_choice(items, weights):
    return random.choices(items, weights=weights, k=1)[0]

def build_day_distribution(cfg):
    start = datetime.strptime(cfg["date_range"]["start"], "%Y-%m-%d").date()
    end = datetime.strptime(cfg["date_range"]["end"], "%Y-%m-%d").date()

    dates = []
    weights = []

    for d in daterange(start, end):
        iso_day = d.isoweekday() 
        month = d.month

        day_weight = cfg["day_weights"][iso_day]
        month_weight = cfg["month_weights"][month]

        weight = day_weight * month_weight

        dates.append(d)
        weights.append(weight)

    return dates, normalize(weights)


def generate_timestamps(cfg, n):
    random.seed(cfg.get("seed", None))

    dates, date_probs = build_day_distribution(cfg)

    weekday_profile = cfg["hourly_profiles"]["weekday"]
    weekend_profile = cfg["hourly_profiles"]["weekend"]

    timestamps = []

    for _ in range(n):
        d = weighted_choice(dates, date_probs)

        if d.isoweekday() in (6, 7):
            hour_probs = weekend_profile
        else:
            hour_probs = weekday_profile

        hours = list(range(24))
        hour = weighted_choice(hours, hour_probs)

        minute = random.randint(0, 59)
        second = random.randint(0, 59)

        ts = datetime(d.year, d.month, d.day, hour, minute, second)
        timestamps.append(ts.strftime("%Y-%m-%d %H:%M:%S"))

    return timestamps

if __name__ == "__main__":
    cfg = load_config("synth_dataset_config.yaml")

    n = 300_000
    timestamps = generate_timestamps(cfg, n=n)

    #sort
    timestamps = sorted(timestamps)
    timestamps = pd.DataFrame(timestamps)
    timestamps.to_csv('timestamps.csv', index=False)
