from __future__ import annotations
import numpy as np
import pandas as pd
from river.drift import PageHinkley


class CentroidDriftDetector:

    def __init__(self, warmup_days: int = 10, ph_threshold: float = 30.0, ph_min_instances: int = 10, ph_alpha: float = 0.999, ph_delta: float = 0.01) -> None:

        self.ph = PageHinkley(
            threshold=ph_threshold,
            min_instances=ph_min_instances,
            alpha=ph_alpha,
            delta=ph_delta
        )
        self.drift_detected = False
        self.drift_log = []

        self.warmup_days = warmup_days
        self.start_date = None
        self.reference_centroid = None
        self.warmup_vectors = []

        self._ref_mean = None
        self._ref_std = None

        self.current_day = None
        self.current_vectors = []

    def update(self, x: np.ndarray, timestamp: pd.Timestamp, instance_count: int = 0) -> CentroidDriftDetector:

        self.drift_detected = False
        day = pd.Timestamp(timestamp).normalize()

        if self.start_date is None:
            self.start_date = day
            self.current_day = day

        days_from_start = (day - self.start_date).days

        if days_from_start < self.warmup_days:
            self.warmup_vectors.append(x)
            return self

        if self.reference_centroid is None:
            vectors = np.array(self.warmup_vectors)
            self._ref_mean = vectors.mean(axis=0)
            self._ref_std  = vectors.std(axis=0) + 1e-6
            self.reference_centroid = np.zeros(len(self._ref_mean))

        if day != self.current_day:
            self.analyze_day(self.current_day, instance_count)
            self.current_day = day
            self.current_vectors = []

        self.current_vectors.append(x)

        return self

    def analyze_day(self, day: pd.Timestamp, instance_count: int) -> None:

        if not self.current_vectors:
            return

        day_centroid = np.mean(self.current_vectors, axis=0)

        day_centroid_norm = (day_centroid - self._ref_mean) / self._ref_std

        distance = np.linalg.norm(day_centroid_norm)

        self.ph.update(distance)
        self.drift_detected = self.ph.drift_detected

        if self.drift_detected:
            self.drift_log.append({
                "date": day,
                "instance": instance_count,
                "distance": float(distance),
                "daily_centroid": day_centroid.copy(),
            })