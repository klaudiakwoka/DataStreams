import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D
from matplotlib.patches import Patch

SYNTHETIC_DRIFT_RANGES = {
    "South":     [("2008-02-08", "2008-02-09")],
    "Northeast": [("2008-02-15", "2008-02-19")],
}

def plot_centroid(
    df_predictions, df_drift, districts_settings, dataset,
    date_from=None, date_to=None, save_path=None, y_max=80,
):
    df_predictions = df_predictions.copy()
    df_drift = df_drift.copy()
    df_predictions["timestamp"] = pd.to_datetime(df_predictions["timestamp"])
    df_drift["timestamp"] = pd.to_datetime(df_drift["timestamp"])

    t_from = pd.to_datetime(date_from) if date_from else None
    t_to   = pd.to_datetime(date_to)   if date_to   else None

    districts = [d for d in districts_settings if d != "Global"]

    def rolling_rmse(dist):
        window = districts_settings[dist]["window"]
        sub = df_predictions[df_predictions["district"] == dist].sort_values("timestamp").reset_index(drop=True)
        sub["rmse"] = (sub["y_hat"] - sub["y"]).pow(2).rolling(window).mean().apply(np.sqrt)
        sub = sub.dropna(subset=["rmse"])
        if t_from is not None: sub = sub[sub["timestamp"] >= t_from]
        if t_to   is not None: sub = sub[sub["timestamp"] <= t_to]
        return sub["timestamp"].tolist(), sub["rmse"].tolist()

    fig, axes = plt.subplots(len(districts), 1, figsize=(20, 6 * len(districts)), sharex=False)
    if len(districts) == 1:
        axes = [axes]

    for ax, dist in zip(axes, districts):
        dates_plot, rmses_plot = rolling_rmse(dist)
        ax.plot(list(dates_plot), list(rmses_plot), color="#16A34A", linewidth=1.5, zorder=3)


        for start_str, end_str in SYNTHETIC_DRIFT_RANGES.get(dist, []):
            ax.axvspan(pd.to_datetime(start_str),
                       pd.to_datetime(end_str) + pd.Timedelta(days=1),
                       color="#A855F7", alpha=0.2, zorder=1)


        sub_drift = df_drift[(df_drift["district"] == dist) & (df_drift["event_type"] != "warning")
            & (t_from is None or df_drift["timestamp"] >= t_from)
            & (t_to   is None or df_drift["timestamp"] <= t_to)
        ]
        present_types = set()
        for i, row in sub_drift.iterrows():
            present_types.add(row["event_type"])
            if row["event_type"] == "centroid":

                ax.axvline(row["timestamp"], color="#DC2626", linestyle=":", linewidth=4.0, zorder=2)
            elif row["event_type"] == "drift":
                ax.axvline(row["timestamp"], color="#2563EB", linestyle="--", linewidth=2.0, zorder=2)


        legend = [Line2D([0], [0], color="#16A34A", linewidth=1.8, label="RMSE")]
        if "drift"    in present_types:
            legend.append(Line2D([0], [0], color="#2563EB", linestyle="--", linewidth=2.0, label="Concept drift (ADWIN)"))
        if "centroid" in present_types:
            legend.append(Line2D([0], [0], color="#DC2626", linestyle=":", linewidth=2.5, label="Virtual drift (Centroid PH)"))
        if dist in SYNTHETIC_DRIFT_RANGES:
            legend.append(Patch(facecolor="#A855F7", alpha=0.4, label="Synthetic drift period"))

        samples = int((df_predictions[df_predictions["district"] == dist]["timestamp"] <= t_to).sum()) \
                  if t_to is not None else len(df_predictions[df_predictions["district"] == dist])

        ax.text(0.97, 0.93, f"Samples: {samples:,}", transform=ax.transAxes, va="top", ha="right",
                bbox=dict(boxstyle="round,pad=0.3", facecolor="#FEF08A", alpha=0.85, edgecolor="#2563EB"),
                fontsize=13, fontweight="bold")
        ax.set_title(dist, fontsize=22, fontweight="bold", pad=8)
        ax.set_ylabel("RMSE", fontsize=12)
        ax.set_ylim(0, y_max)
        ax.grid(True, linestyle="--", alpha=0.45)
        ax.legend(handles=legend, loc="upper left", frameon=True, facecolor="white", fontsize=11)
        ax.xaxis.set_major_formatter(mdates.DateFormatter("%y-%m-%d"))
        ax.tick_params(axis="x", rotation=45)
        if t_from or t_to:
            ax.set_xlim(left=t_from if t_from else None, right=t_to if t_to else None)

    fig.suptitle(f"Centroid Drift Analysis – {dataset}", fontsize=26, fontweight="bold", y=1.005)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)