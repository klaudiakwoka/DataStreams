from __future__ import annotations

import copy
from collections import deque
from dataclasses import dataclass, field
from typing import Any, Deque, List, Optional

from river import base, metrics, tree
from river.drift import ADWIN
import math

@dataclass
class DriftEvent:
    sample_id: int
    timestamp: Any
    event_type: str   # "warning" or "drift"
    model: str

@dataclass
class _Slot:
    """Wraps a single estimator together with its drift detector and bookkeeping."""

    model: base.Regressor
    detector: ADWIN = field(default_factory=lambda: ADWIN(delta=0.002))
    error_window: Deque[float] = field(default_factory=deque)
    weight: float = 1.0
    is_shadow: bool = False
    label: str = "model"

    def update_error(self, y_true: float, y_pred: float) -> None:
        err = abs(y_true - y_pred)
        self.error_window.append(err)
        self.detector.update(err)

    # rmse
    @property
    def rmse(self) -> float:
        if not self.error_window:
            return 1.0
    
        mse = sum(self.error_window) / len(self.error_window)
        return math.sqrt(mse)


class DriftAdaptiveEnsemble(base.Regressor):
    """Weighted streaming ensemble that adapts to concept drift."""

    def __init__(
        self,
        n: int = 5,
        base_estimator: Optional[base.Regressor] = None,
        keep_first_model: bool = False,
        warning_delta: float = 0.005,
        drift_delta: float = 0.002,
        weight_alpha: float = 5.0,
        error_window_size: int = 200,
    ) -> None:
        if n < 1:
            raise ValueError("`n` must be at least 1.")

        self.n = n
        self.base_estimator = base_estimator or tree.HoeffdingTreeRegressor()
        self.keep_first_model = keep_first_model
        self.warning_delta = warning_delta
        self.drift_delta = drift_delta
        self.weight_alpha = weight_alpha

        # first model
        self._genesis: Optional[_Slot] = None
        self._slots: List[_Slot] = []
        # separate per-slot warning detectors
        self._warning_detectors: List[ADWIN] = []
        self._model_counter: int = 0
        self._has_shadow: bool = False
        
        self.event_log: List[DriftEvent] = []
        self._n_samples_seen: int = 0
        self.error_window_size = error_window_size

    def _new_slot(self, label: str, is_shadow: bool = False) -> _Slot:
        model = copy.deepcopy(self.base_estimator)
        return _Slot(
            model=model,
            detector=ADWIN(delta=self.drift_delta),
            error_window=deque(maxlen=self.error_window_size),
            weight=0.0 if is_shadow else 1.0,
            is_shadow=is_shadow,
            label=label,
        )

    def _new_warning_detector(self) -> ADWIN:
        return ADWIN(delta=self.warning_delta)

    @property
    def _active_slots(self) -> List[_Slot]:
        """All slots that contribute to predictions (non-shadow, non-genesis)."""
        return [s for s in self._slots if not s.is_shadow]

    @property
    def _all_slots(self) -> List[_Slot]:
        """Genesis (if any) + sliding window slots."""
        if self._genesis is not None and self.keep_first_model:
            return [self._genesis] + self._slots
        return self._slots

    def _recalculate_weights(self) -> None:
        """Inverse-MAE softmax over all non-shadow slots (including genesis)."""
        slots = [s for s in self._all_slots if not s.is_shadow]
        if not slots:
            return

        inv_errors = [1.0 / max(s.rmse, 1e-9) for s in slots]
        # softmax with temperature
        scaled = [self.weight_alpha * ie for ie in inv_errors]
        max_s = max(scaled)
        exps = [math.exp(v - max_s) for v in scaled]
        total = sum(exps)
        weights = [e / total for e in exps]

        for slot, w in zip(slots, weights):
            slot.weight = w

    def _evict_oldest_if_needed(self) -> None:
        """Remove oldest non-genesis slot when sliding window exceeds `n`."""
        non_shadow = [s for s in self._slots if not s.is_shadow]
        # keep at most self.n models in the window
        while len(non_shadow) > self.n:
            # find and remove the oldest non-shadow slot
            for i, s in enumerate(self._slots):
                if not s.is_shadow:
                    self._slots.pop(i)
                    self._warning_detectors.pop(i)
                    break
            non_shadow = [s for s in self._slots if not s.is_shadow]


    def learn_one(self, x: dict, y: float, timestamp=None) -> "DriftAdaptiveEnsemble":

        self._n_samples_seen += 1
    
        # -------------------------------------------------
        # initialize first model
        # -------------------------------------------------
        if not self._slots and self._genesis is None:
            self._model_counter += 1
    
            first_slot = self._new_slot(label=f"m_{self._model_counter}")
            first_slot.weight = 1.0
    
            self._slots.append(first_slot)
            self._warning_detectors.append(self._new_warning_detector())
    
            if self.keep_first_model:
                self._genesis = self._new_slot(label="m_genesis")
    
        # -------------------------------------------------
        # prediction + error + training
        # -------------------------------------------------
        for i, slot in enumerate(self._slots):
    
            try:
                y_hat = slot.model.predict_one(x)
            except Exception:
                y_hat = 0.0
    
            err = (y - y_hat) ** 2
    
            slot.error_window.append(err)
    
            slot.detector.update(err)
    
            if hasattr(self, "event_log"):
                pass
    
            # train model
            slot.model.learn_one(x, y)
    
        # -------------------------------------------------
        # genesis model (if used)
        # -------------------------------------------------
        if self._genesis is not None and self.keep_first_model:
    
            try:
                y_hat_g = self._genesis.model.predict_one(x)
            except Exception:
                y_hat_g = 0.0
    
            err_g = (y - y_hat_g) ** 2
    
            self._genesis.error_window.append(err_g)
            self._genesis.detector.update(err_g)
    
            self._genesis.model.learn_one(x, y)
    
        # -------------------------------------------------
        # WARNING + SHADOW detection
        # -------------------------------------------------
        for i, slot in enumerate(self._slots):
    
            if slot.is_shadow:
                continue
    
            warn_det = self._warning_detectors[i]
    
            # use SAME signal (no second window)
            current_err = slot.error_window[-1] if slot.error_window else 0.0
            warn_det.update(current_err)
    
            # -------------------------
            # WARNING EVENT
            # -------------------------
            if warn_det.drift_detected and not self._has_shadow:
    
                self._model_counter += 1
    
                shadow = self._new_slot(
                    label=f"m_{self._model_counter}",
                    is_shadow=True,
                )
    
                self._slots.append(shadow)
                self._warning_detectors.append(self._new_warning_detector())
    
                self._has_shadow = True
    
                # immediate warm start
                shadow.model.learn_one(x, y)
    
                # reset warning detector that fired
                self._warning_detectors[i] = self._new_warning_detector()
    
                if hasattr(self, "event_log"):
                    self.event_log.append(
                        DriftEvent(
                            sample_id=self._n_samples_seen,
                            timestamp=timestamp,
                            event_type="warning",
                            model=slot.label,
                        )
                    )
    
            # -------------------------
            # DRIFT CONFIRMATION
            # -------------------------
            if slot.detector.drift_detected:
    
                promoted_any = False
    
                for s in self._slots:
                    if s.is_shadow:
                        s.is_shadow = False
                        promoted_any = True
    
                if promoted_any:
                    self._has_shadow = False
    
                    self._evict_oldest_if_needed()
                    self._recalculate_weights()
    
                    if hasattr(self, "event_log"):
                        self.event_log.append(
                            DriftEvent(
                                sample_id=self._n_samples_seen,
                                timestamp=timestamp,
                                event_type="drift",
                                model=slot.label,
                            )
                        )
    
                # reset detector after trigger
                slot.detector = ADWIN(delta=self.drift_delta)
    
        return self

    def predict_one(self, x: dict) -> float:
        if not self._all_slots:
            return 0.0

        total_weight = 0.0
        weighted_sum = 0.0
        for slot in self._all_slots:
            if slot.is_shadow:
                continue  # shadows don't vote
            try:
                pred = slot.model.predict_one(x)
            except Exception:
                pred = 0.0
            weighted_sum += slot.weight * pred
            total_weight += slot.weight

        if total_weight == 0.0:
            return 0.0
        return weighted_sum / total_weight

    # ------------------------------------------------------------------
    # introspection helpers
    # ------------------------------------------------------------------

    def summary(self) -> str:
        """Human-readable state of the ensemble."""
        lines = [
            f"DriftAdaptiveEnsemble  (n={self.n}, "
            f"keep_first_model={self.keep_first_model}, "
            f"samples_seen={self._n_samples_seen})",
            "-" * 60,
        ]
        for slot in self._all_slots:
            tag = "[SHADOW] " if slot.is_shadow else ""
            lines.append(
                f"  {tag}{slot.label:12s}  weight={slot.weight:.4f}  "
                f"mean_err={slot.rmse:.5f}"
            )
        lines.append(
            f"\nTotal active models (excl. shadow): "
            f"{len([s for s in self._all_slots if not s.is_shadow])}"
        )
        return "\n".join(lines)

    def recent_events(self, n: int = 10) -> str:
        rows = self.event_log[-n:]
    
        lines = []
        for e in rows:
            lines.append(
                f"[{e.sample_id}] "
                f"{e.event_type.upper():7s} "
                f"model={e.source_model:10s} "
                f"mean_err={e.rmse:.4f} "
                f"inst_err={e.squared_error:.4f} "
                f"action={e.action}"
            )
    
        return "\n".join(lines)

    @property
    def n_models(self) -> int:
        """Number of voting (non-shadow) models currently in the ensemble."""
        return len([s for s in self._all_slots if not s.is_shadow])

    @property
    def models(self) -> List[base.Regressor]:
        """Ordered list of voting model objects."""
        return [s.model for s in self._all_slots if not s.is_shadow]

    @property
    def weights(self) -> List[float]:
        """Weights corresponding to ``self.models``."""
        return [s.weight for s in self._all_slots if not s.is_shadow]
