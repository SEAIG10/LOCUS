"""
Decision Engine wrapper that delegates to the shared policy module.
"""

from __future__ import annotations

from datetime import datetime
from typing import Iterable, Mapping, Optional

import numpy as np

from core.policy import PolicyConfig, PolicyDecision, PolicyEngine

# Backwards compatibility for existing imports
CleaningDecision = PolicyDecision


class LocalDecisionEngine:
    """
    로컬 의사결정 엔진

    PolicyEngine을 감싸고 장치별 설정(임계값, 구역, 시간당 소요 시간 등)을 전달합니다.
    """

    def __init__(
        self,
        pollution_threshold: float = 0.5,
        zone_names: Optional[list[str]] = None,
        time_per_zone: int = 10,
        occupant_sensitive_zones: Optional[list[str]] = None,
    ):
        self.zone_names = zone_names or ["balcony", "bedroom", "kitchen", "living_room"]
        self.policy = PolicyEngine(
            PolicyConfig(
                zone_names=self.zone_names,
                pollution_threshold=pollution_threshold,
                time_per_zone_min=time_per_zone,
                occupant_sensitive_zones=occupant_sensitive_zones or self.zone_names,
            )
        )
        self.current_position: Optional[dict] = None

        print(f"\n{'='*60}")
        print("Policy-backed Decision Engine Initialized")
        print(f"{'='*60}")
        print(f"Zones: {self.zone_names}")
        print(f"Threshold: {pollution_threshold}")
        print(f"Time/zone: {time_per_zone} min")
        print(f"{'='*60}\n")

    def decide(
        self,
        prediction: np.ndarray,
        *,
        occupancy: Optional[Mapping[str, bool] | Iterable[str]] = None,
        battery_level: float = 1.0,
        timestamp: Optional[datetime] = None,
    ) -> PolicyDecision:
        """
        정책 엔진을 호출해 CleaningDecision(PolicyDecision)을 생성합니다.
        """
        return self.policy.decide(
            prediction,
            occupancy=occupancy,
            battery_level=battery_level,
            timestamp=timestamp,
            current_position=self.current_position,
        )

    def update_position(self, position: dict):
        """
        현재 위치 업데이트 (예: ARKit 좌표)
        """
        self.current_position = position


if __name__ == "__main__":
    engine = LocalDecisionEngine(
        pollution_threshold=0.55,
        zone_names=["balcony", "bedroom", "kitchen", "living_room"],
    )
    test_predictions = [
        np.array([0.85, 0.12, 0.65, 0.23]),
        np.array([0.30, 0.45, 0.20, 0.15]),
        np.array([0.95, 0.88, 0.72, 0.91]),
    ]

    for idx, pred in enumerate(test_predictions, 1):
        print(f"\n{'#'*60}")
        print(f"Test Case {idx}")
        print(f"{'#'*60}")
        print(f"Prediction: {pred}")
        decision = engine.decide(pred)
        print(decision)
