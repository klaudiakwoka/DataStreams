from __future__ import annotations

import copy
import math
from typing import Literal
import pandas as pd
from collections import deque

from river import base, metrics as river_metrics

def _clone(estimator):
    """Untrained copy of estimator"""
    return copy.deepcopy(estimator)

def _metric_better_than(a_score, b_score, metric):
    """Comparing river metric values"""
    if metric.bigger_is_better:
        return a_score > b_score
    return a_score < b_score


def _worst_index(scores, metric):
    """Return the index of the worst score"""
    if metric.bigger_is_better:
        return scores.index(min(scores))
    return scores.index(max(scores))


class _EnsembleMember:
    """River regressor with its metric"""

    def __init__(self, model, metric, member_id=0):
        self.model: base.Regressor = model
        self.metric: base.Metric = _clone(metric)
        self.member_id = member_id
        self.created_at = 0
        self.promoted_at = None

    @property
    def score(self):
        """Metric value"""
        return self.metric.get()

    def learn_one(self, x, y):
        pred = self.model.predict_one(x)
        if pred is not None:
            self.metric.update(y, pred)
        self.model.learn_one(x, y)

    def predict_one(self, x):
        pred = self.model.predict_one(x)
        return pred if pred is not None else 0.0

    def __repr__(self): 
        return (
            f"<_EnsembleMember id={self.member_id} "
            f"model={self.model.__class__.__name__} "
            f"metric={self.metric.__class__.__name__}={self.score:.4f} "
            f"created_at={self.created_at} promoted_at={self.promoted_at}>"
        )

class DriftAdaptiveEnsemble(base.Regressor):
    def __init__(
        self,
        base_estimator: base.Regressor,
        max_ensemble_size: int = 5,
        retain_initial_model: bool = True,
        drift_detector: base.DriftDetector,
        warning_detector: base.DriftDetector,
        metric: base.Metric,
        prediction_strategy: Literal["mean", "weighted_mean"] = "mean",
        warmup: int = 1000,
        window_size: int = 500
    ):
        
        if max_ensemble_size < 1:
            raise ValueError("max_ensemble_size must be >= 1.")
        if prediction_strategy not in {"mean", "weighted_mean"}:
            raise ValueError("prediction_strategy must be 'mean' or 'weighted_mean'.")

        self.base_estimator = base_estimator
        self.max_ensemble_size = max_ensemble_size
        self.retain_initial_model = retain_initial_model
        self.drift_detector = drift_detector
        self.warning_detector = warning_detector
        self.metric = metric
        self.prediction_strategy = prediction_strategy
        self._next_member_id = 0

        # initial ensemble state
        initial_member = _EnsembleMember(_clone(base_estimator), self.metric, member_id=0)
        self._next_member_id = 1
        self._members: list[_EnsembleMember] = [initial_member]
        self._initial_member: _EnsembleMember = initial_member

        self._shadow_member = None

        self._detector = _clone(drift_detector)
        self._warning_detector = _clone(warning_detector)

        self._n_seen = 0
        self._drift_events = []
        self._warmup = warmup
        self._error_window = deque(maxlen=window_size)

    # learn one method
    def learn_one(self, x, y, timestamp):
        self._n_seen += 1

        # predict
        ensemble_pred = self._raw_predict(x)

         # train
        for member in self._members:
            member.learn_one(x, y)

        # train shadow
        if self._shadow_member is not None:
            self._shadow_member.learn_one(x, y)

        # update drift detector
        if self._detector is not None and self._n_seen >= self._warmup:
            error = abs(y - ensemble_pred)
            self._error_window.append(error)
            
            if len(self._error_window) == self._error_window.maxlen:
                smoothed_error = sum(self._error_window) / len(self._error_window)
                self._detector.update(smoothed_error)
                if self._warning_detector is not None:
                    self._warning_detector.update(smoothed_error)
                self._handle_drift_signal(timestamp)

    # predict method
    def predict_one(self, x):
        return self._aggregate(x)

    def _handle_drift_signal(self, timestamp):
        """Handle warning and drift signals from the detector"""

        # drift detected
        if self._detector.drift_detected:
            self._drift_events.append({
                "timestamp": timestamp,
                "event_type": "drift",
                "n_seen": self._n_seen,
                "ensemble_ids": [m.member_id for m in self._members]
            })
            if self._shadow_member is not None:
                self._promote_shadow(timestamp)
            # reset
            self._reset_detector()
            return 

        # warning detected
        if self._warning_detector is not None:
            warning = self._warning_detector.drift_detected
        else:
            warning = getattr(detector, "warning_detected", False)
        # create shadow model
        if warning and self._shadow_member is None:
            new_id = self._next_member_id
            self._next_member_id += 1

            shadow = _EnsembleMember(_clone(self.base_estimator), self.metric, member_id=new_id)
            shadow.created_at = self._n_seen
            self._shadow_member = shadow
        
            self._drift_events.append({
                "timestamp": timestamp,
                "event_type": "warning",
                "n_seen": self._n_seen,
                "shadow_id": new_id,
                "ensemble_ids": [m.member_id for m in self._members]
            })

    def _promote_shadow(self, timestamp):
        """Add the shadow model to the ensemble, removing the worst if there is already max model capacity"""
        if self._shadow_member is None:
            return

        evicted_id = None
        if len(self._members) >= self.max_ensemble_size:
            evicted_id = self._evict_worst()

        self._shadow_member.promoted_at = self._n_seen
        self._members.append(self._shadow_member)
        self._drift_events.append({
            "timestamp": timestamp,
            "event_type": "promotion",
            "n_seen": self._n_seen,
            "promoted_id": self._shadow_member.member_id,
            "evicted_id": evicted_id,
            "ensemble_ids": [m.member_id for m in self._members],
        })
        self._shadow_member = None

    def _evict_worst(self):
        """Remove the worst model"""
        if not self._members:
            return

        scores = [m.score for m in self._members]

        # remove worst model besides the first one
        if self.retain_initial_model and self._initial_member in self._members:
            candidates = [
                (i, s) for i, (m, s) in enumerate(zip(self._members, scores))
                if m is not self._initial_member
            ]
            if not candidates:
                return
            candidate_scores = [s for _, s in candidates]
            worst_local = _worst_index(candidate_scores, self.metric)
            worst_idx = candidates[worst_local][0]
        # remove worst model
        else:
            worst_idx = _worst_index(scores, self.metric)

        evicted_id = self._members[worst_idx].member_id
        del self._members[worst_idx]
        return evicted_id

    def _reset_detector(self):
        """Drift detector reset"""
        self._detector = _clone(self.drift_detector)
        if self.warning_detector is not None:
            self._warning_detector = _clone(self.warning_detector)

    def _raw_predict(self, x):
        """Aggregate predictions"""
        return self._aggregate(x)

    def _aggregate(self, x):
        if not self._members:
            return 0.0

        predictions = [m.predict_one(x) for m in self._members]

        if self.prediction_strategy == "mean" or len(self._members) == 1:
            return sum(predictions) / len(predictions)

        # weighted mean
        eps = 1e-10
        weights: list[float] = []
        for member in self._members:
            s = member.score
            if math.isnan(s) or math.isinf(s):
                s = eps
            if self.metric.bigger_is_better:
                weights.append(max(s, eps))
            else:
                weights.append(1.0 / max(s, eps))

        total_weight = sum(weights)
        if total_weight == 0:
            return sum(predictions) / len(predictions)

        return sum(p * w for p, w in zip(predictions, weights)) / total_weight

    @property
    def n_members(self):
        """Number of active ensemble members."""
        return len(self._members)

    @property
    def has_shadow(self):
        """Whether a shadow model is currently being trained."""
        return self._shadow_member is not None

    @property
    def drift_log(self):
        return pd.DataFrame(
            self._drift_events,
            columns=["timestamp", "n_seen", "event_type", "ensemble_ids", "shadow_id", "promoted_id", "evicted_id"]
        )

    def member_scores(self):
        """Return the current metric score for each ensemble member."""
        weights = self._compute_weights()
        return {
            m.member_id: {"score": s, "weight": w}
            for m, s, w in zip(self._members, scores, weights)
        }

    def __repr__(self):
        return (
            f"DriftEnsembleRegressor("
            f"n_members={self.n_members}, "
            f"max_size={self.max_ensemble_size}, "
            f"strategy={self.prediction_strategy!r}, "
            f"has_shadow={self.has_shadow}, "
            f"n_seen={self._n_seen})"
        )
