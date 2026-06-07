import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.dates as mdates
from matplotlib.lines import Line2D


DRIFT_STYLE = {
    "warning":  dict(color="#F59E0B", linestyle="-.", linewidth=1.8, alpha=0.9),
    "drift":    dict(color="#2563EB", linestyle="--", linewidth=2.0, alpha=0.9),
    "centroid": dict(color="#DC2626", linestyle=":",  linewidth=2.5, alpha=0.9),
}
DRIFT_LABEL = {
    "warning":  "Warning (ADWIN)",
    "drift":    "Error drift (ADWIN)",
    "centroid": "Feature drift (Centroid PH)",
}


def rolling_rmse(df_predictions, dist, window, t_from, t_to):
    sub = df_predictions[df_predictions["district"] == dist].sort_values("timestamp").reset_index(drop=True)
    sub["rmse"] = (sub["y_hat"] - sub["y"]).pow(2).rolling(window).mean().apply(np.sqrt)
    sub = sub.dropna(subset=["rmse"])
    if t_from is not None: sub = sub[sub["timestamp"] >= t_from]
    if t_to   is not None: sub = sub[sub["timestamp"] <= t_to]
    return sub["timestamp"].tolist(), sub["rmse"].tolist()


def plot_drift(
    df_predictions, df_drift, districts_settings, dataset,
    date_from=None, date_to=None, save_path=None, y_max=80,
):
    df_predictions = df_predictions.copy()
    df_drift = df_drift.copy()
    df_predictions["timestamp"] = pd.to_datetime(df_predictions["timestamp"])
    df_drift["timestamp"] = pd.to_datetime(df_drift["timestamp"])

    t_from = pd.to_datetime(date_from) if date_from else None
    t_to   = pd.to_datetime(date_to)   if date_to   else None

    districts = list(districts_settings.keys())

    fig, axes = plt.subplots(len(districts), 1, figsize=(20, 6 * len(districts)), sharex=False)
    if len(districts) == 1:
        axes = [axes]

    for ax, dist in zip(axes, districts):
        window = districts_settings[dist]["window"]
        dates_plot, rmses_plot = rolling_rmse(df_predictions, dist, window, t_from, t_to)
        ax.plot(dates_plot, rmses_plot, color="#16A34A", linewidth=1.5, zorder=3)

        sub_drift = df_drift[
            (df_drift["district"] == dist) &
            (t_from is None or df_drift["timestamp"] >= t_from) &
            (t_to   is None or df_drift["timestamp"] <= t_to)
        ]

        present_types = set()
        for _, row in sub_drift.iterrows():
            etype = row["event_type"]
            style = DRIFT_STYLE.get(etype)
            if style is None:
                continue
            ax.axvline(row["timestamp"], **style, zorder=2)
            present_types.add(etype)

        legend = [Line2D([0], [0], color="#16A34A", linewidth=1.8, label="RMSE")]
        for etype in ["warning", "drift", "centroid"]:
            if etype in present_types:
                s = DRIFT_STYLE[etype]
                legend.append(Line2D([0], [0], color=s["color"], linestyle=s["linestyle"],
                                     linewidth=s["linewidth"], label=DRIFT_LABEL[etype]))

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

    fig.suptitle(f"Drift Analysis – {dataset}", fontsize=26, fontweight="bold", y=1.005)
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=300, bbox_inches="tight")
    plt.show()
    plt.close(fig)
