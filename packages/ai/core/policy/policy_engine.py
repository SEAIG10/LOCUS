"""
Policy module for LOCUS.

This module centralizes the rules that translate GRU pollution predictions
into actionable cleaning plans.  It can be used from realtime scripts as well
as backend workers so that every surface (robot, simulator, backend) shares the
same decision logic.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from datetime import datetime
from typing import Dict, Iterable, List, Mapping, Optional, Sequence, Tuple

import numpy as np


@dataclass
class PolicyConfig:
    """User/environment-specific knobs for the policy engine."""

    zone_names: List[str]
    pollution_threshold: float = 0.5
    time_per_zone_min: int = 10
    quiet_hours: Sequence[Tuple[int, int]] = ((22, 7),)
    min_battery_level: float = 0.25
    max_zones_per_run: int = 4
    occupant_sensitive_zones: Sequence[str] = ()


@dataclass
class PolicyDecision:
    """Structured decision returned by the policy engine."""

    zones_to_clean: List[str]
    priority_order: List[float]
    path: List[str]
    estimated_time: int
    threshold_used: float
    action: str
    reason: str
    deferred_zones: List[str] = field(default_factory=list)

    def __str__(self) -> str:
        header = f"\n{'=' * 60}\n POLICY DECISION ({self.action.upper()})\n{'=' * 60}\n"
        body = f"Reason: {self.reason}\nThreshold: {self.threshold_used}\n"

        if not self.zones_to_clean:
            if self.deferred_zones:
                body += f"Deferred zones: {', '.join(self.deferred_zones)}\n"
            return header + body + "=" * 60

        body += f"Estimated cleaning time: {self.estimated_time} min\n"
        body += f"Path:\n"
        for idx, (zone, score) in enumerate(zip(self.path, self.priority_order), 1):
            body += f"  {idx}. {zone:15s} (pollution: {score:.2%})\n"

        if self.deferred_zones:
            body += f"\nDeferred (occupied/quiet hour): {', '.join(self.deferred_zones)}\n"

        return header + body + "=" * 60


class PolicyEngine:
    """Rule-based policy engine."""

    def __init__(self, config: PolicyConfig):
        self.config = config

    def decide(
        self,
        prediction: Mapping[str, float] | Sequence[float] | np.ndarray,
        *,
        occupancy: Optional[Mapping[str, bool] | Iterable[str]] = None,
        battery_level: float = 1.0,
        timestamp: Optional[datetime] = None,
        current_position: Optional[Dict[str, float]] = None,
    ) -> PolicyDecision:
        zone_scores = self._normalize_prediction(prediction)

        ranked = sorted(zone_scores.items(), key=lambda x: x[1], reverse=True)
        selected = [
            (zone, score)
            for zone, score in ranked
            if score >= self.config.pollution_threshold
        ]

        occupied_zones = self._extract_occupied_zones(occupancy)
        sensitive_zones = set(self.config.occupant_sensitive_zones or [])

        filtered: List[Tuple[str, float]] = []
        deferred: List[str] = []
        for zone, score in selected:
            if zone in occupied_zones and (
                not sensitive_zones or zone in sensitive_zones
            ):
                deferred.append(zone)
            else:
                filtered.append((zone, score))

        selected = filtered

        if self.config.max_zones_per_run:
            selected = selected[: self.config.max_zones_per_run]

        path = self._plan_path(selected, current_position)
        priority_map = {zone: score for zone, score in selected}
        priority_order = [priority_map[zone] for zone in path]

        zones_to_clean = path.copy()
        estimated_time = len(zones_to_clean) * self.config.time_per_zone_min

        action = "clean_now"
        if not zones_to_clean:
            if deferred:
                action = "defer"
            else:
                action = "idle"

        reason = self._build_reason(action, zones_to_clean, deferred)

        if action == "clean_now":
            if timestamp and self._is_quiet_hours(timestamp):
                action = "defer"
                reason = "Quiet hours - postpone cleaning"
            elif battery_level < self.config.min_battery_level:
                action = "defer"
                reason = (
                    f"Battery {battery_level * 100:.0f}% "
                    f"< {self.config.min_battery_level * 100:.0f}% limit"
                )

        return PolicyDecision(
            zones_to_clean=zones_to_clean,
            priority_order=priority_order,
            path=zones_to_clean,
            estimated_time=estimated_time,
            threshold_used=self.config.pollution_threshold,
            action=action,
            reason=reason,
            deferred_zones=deferred,
        )

    def _normalize_prediction(
        self, prediction: Mapping[str, float] | Sequence[float] | np.ndarray
    ) -> Dict[str, float]:
        if isinstance(prediction, Mapping):
            return {
                zone: float(prediction.get(zone, 0.0))
                for zone in self.config.zone_names
            }

        array = np.asarray(prediction, dtype=np.float32).flatten()
        if array.size < len(self.config.zone_names):
            raise ValueError(
                f"Prediction contains {array.size} values "
                f"but {len(self.config.zone_names)} zones are configured."
            )

        return {
            zone: float(array[idx]) for idx, zone in enumerate(self.config.zone_names)
        }

    def _extract_occupied_zones(
        self, occupancy: Optional[Mapping[str, bool] | Iterable[str]]
    ) -> set[str]:
        if occupancy is None:
            return set()

        if isinstance(occupancy, Mapping):
            return {zone for zone, present in occupancy.items() if bool(present)}

        return {str(zone) for zone in occupancy}

    def _is_quiet_hours(self, timestamp: datetime) -> bool:
        if not self.config.quiet_hours:
            return False

        hour = timestamp.hour
        for start, end in self.config.quiet_hours:
            if start <= end:
                if start <= hour < end:
                    return True
            else:
                if hour >= start or hour < end:
                    return True
        return False

    def _plan_path(
        self,
        selected: Sequence[Tuple[str, float]],
        current_position: Optional[Dict[str, float]],
    ) -> List[str]:
        # Placeholder for more advanced path planning (A*, TSP, etc.)
        # For now we simply return in descending pollution order.
        return [zone for zone, _ in selected]

    def _build_reason(
        self, action: str, zones_to_clean: Sequence[str], deferred: Sequence[str]
    ) -> str:
        if action == "idle":
            return "All zones below threshold"
        if action == "defer" and deferred:
            return "Occupied zones will be retried later"
        if action == "clean_now":
            return f"{len(zones_to_clean)} zones exceed threshold"
        return "Policy condition triggered"
