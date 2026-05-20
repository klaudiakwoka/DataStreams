from __future__ import annotations
import numpy as np
import pandas as pd
from river.drift import PageHinkley


class CentroidDriftDetector:

    def __init__(self, warmup_days: int = 10, ph_threshold: float = 30.0, ph_min_instances: int = 10, ph_alpha: float = 0.999) -> None:

        self.ph = PageHinkley(
            threshold=ph_threshold,
            min_instances=ph_min_instances,
            alpha=ph_alpha,
        )
        self.drift_detected = False
        self.drift_log = []

        self.warmup_days = warmup_days
        self.start_date = None
        self.reference_centroid = None
        self.warmup_vectors = []

        self.current_day = None
        self.current_vectors = []



    def update(self, x: np.ndarray, timestamp: pd.Timestamp, instance_count: int = 0) -> CentroidDriftDetector:

        self.drift_detected = False
        day = timestamp.normalize()


        if self.start_date is None:
            self.start_date = day
            self.current_day = day

        days_from_start = (day - self.start_date).days


        if days_from_start < self.warmup_days:
            self.warmup_vectors.append(x)
            return self
        if self.reference_centroid is None:
            self.reference_centroid = np.mean(
                self.warmup_vectors,
                axis=0,
            )
        #new day - analyzing previous
        if day != self.current_day:
            self.analyze_day(self.current_day, instance_count )

            self.current_day = day
            self.current_vectors = []

        self.current_vectors.append(x)

        return self

    def analyze_day(self, day: pd.Timestamp, instance_count: int ) -> None:

        if not self.current_vectors:
            return

        day_centroid = np.mean(self.current_vectors,axis=0)

        distance = np.linalg.norm(day_centroid - self.reference_centroid)

        self.ph.update(distance)

        self.drift_detected = self.ph.drift_detected

        if self.drift_detected:
            self.drift_log.append({
                "date": day,
                "instance": instance_count,
                "distance": float(distance),
                "daily_centroid": day_centroid.copy(),
            })