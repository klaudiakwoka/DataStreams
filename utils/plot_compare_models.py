import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

MARKER_MAP = {
    "MEAN": None,
    "HTR": "x",
    "ARF": "s",
    "ARF_drift": "D",
    "SRP": "^",
    "SRP_drift": "v",
    "Ensemble": "P",
}

def plot_compare_models(df, start_date=None, end_date=None, window=1, resample_rule=None, save_path=None):
    df = df.copy()
    df["timestamp"] = pd.to_datetime(df["timestamp"])
    df = df.sort_values("timestamp")

    if start_date is not None:
        start = pd.to_datetime(start_date)
        start = start.normalize()  # 00:00:00
    
    if end_date is not None:
        end = pd.to_datetime(end_date)
        end = end.normalize() + pd.Timedelta(days=1) - pd.Timedelta(microseconds=1)
    
    if start_date is not None and end_date is not None:
        df = df[(df["timestamp"] >= start) & (df["timestamp"] <= end)]
    elif start_date is not None:
        df = df[df["timestamp"] >= start]
    elif end_date is not None:
        df = df[df["timestamp"] <= end]

    # result columns
    model_cols = [c for c in df.columns if c.startswith("y_") and c != "y_true"]

    # resample
    if resample_rule:
        df = df.set_index("timestamp")
        df = df[["y_true"] + model_cols].resample(resample_rule).mean().dropna().reset_index()

    # rolling rmse
    for col in model_cols:
        sq_err = (df["y_true"] - df[col]) ** 2
        df[f"roll_rmse_{col}"] = np.sqrt(
            sq_err.rolling(window=window, min_periods=1).mean()
        )

    # plot
    legend_labels = {
        c: c.split("_", 1)[1] if "_" in c else c
        for c in model_cols
    }
    plt.figure(figsize=(14, 6))

    for col in model_cols:
        suffix = legend_labels[col]
        marker = MARKER_MAP.get(
            suffix
        )
        plt.plot(
            df["timestamp"],
            df[f"roll_rmse_{col}"],
            label=suffix,
            marker=marker,
            markevery=max(len(df)//40,1),
            linewidth=2           
        )

    plt.title(f"Rolling RMSE (window={window})")
    plt.xlabel("Timestamp")
    plt.ylabel("RMSE")
    plt.legend()
    plt.tight_layout()

    if save_path:
        plt.savefig(save_path, bbox_inches="tight")
        
    plt.show()

    return df
    