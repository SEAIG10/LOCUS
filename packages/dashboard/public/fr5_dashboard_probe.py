"""
FR5 Dashboard Probe

policy_bridge / policy_engine이 남긴 policy_events.log.jsonl 로그를 읽어서
웹 대시보드에서 사용할 fr5_live.json 스냅샷을 생성한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Any, Dict, List, Optional

ROOT = Path(__file__).resolve().parents[2]  # .../LOCUS
LOG_PATH = ROOT / "packages" / "gateway" / "logs" / "policy_events.log.jsonl"
DASH_PUBLIC = ROOT / "packages" / "dashboard" / "public"
SNAPSHOT_PATH = DASH_PUBLIC / "fr5_live.json"


@dataclass
class MapRow:
  zone: str
  score: float


@dataclass
class PathRow:
  zone: str
  eta: float
  prob: float


@dataclass
class Fr5Snapshot:
  action: str
  reason: str
  eta: float
  battery: float
  map: List[MapRow]
  path: List[PathRow]
  notes: List[str]


class FR5Probe:
  def __init__(self) -> None:
    DASH_PUBLIC.mkdir(parents=True, exist_ok=True)

  def _read_last_decision(self) -> Optional[Dict[str, Any]]:
    if not LOG_PATH.exists():
      return None

    last: Optional[Dict[str, Any]] = None
    with LOG_PATH.open("r", encoding="utf-8") as f:
      for line in f:
        line = line.strip()
        if not line:
          continue
        try:
          obj = json.loads(line)
        except json.JSONDecodeError:
          continue
        if obj.get("role") == "policy" and obj.get("type") == "decision":
          last = obj
    return last

  def build_snapshot(self) -> Fr5Snapshot:
    decision = self._read_last_decision()

    if not decision:
      # 결정이 아직 없을 때 기본값
      return Fr5Snapshot(
        action="IDLE",
        reason="아직 정책 결정 로그가 없습니다.",
        eta=0,
        battery=0,
        map=[],
        path=[],
        notes=["정책 엔진이 첫 결정을 내리면 이 화면이 자동으로 갱신됩니다."],
      )

    map_rows = [
      MapRow(
        zone=str(row.get("zone", "unknown")),
        score=float(row.get("score", 0.0)),
      )
      for row in decision.get("map", [])
    ]

    path_rows = [
      PathRow(
        zone=str(row.get("zone", "unknown")),
        eta=float(row.get("eta", 0.0)),
        prob=float(row.get("prob", 0.0)),
      )
      for row in decision.get("path", [])
    ]

    notes = [str(n) for n in decision.get("notes", [])]

    return Fr5Snapshot(
      action=str(decision.get("action", "UNKNOWN")),
      reason=str(decision.get("reason", "")),
      eta=float(decision.get("eta", 0.0)),
      battery=float(decision.get("battery", 0.0)),
      map=map_rows,
      path=path_rows,
      notes=notes,
    )

  def run(self, interval: float = 1.0) -> None:
    print(f"[FR5-Probe] writing snapshot to {SNAPSHOT_PATH}")
    while True:
      snap = self.build_snapshot()
      try:
        with SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
          json.dump(asdict(snap), f, ensure_ascii=False, indent=2)
      except Exception as e:
        print(f"[FR5-Probe] snapshot write error: {e}")
      time.sleep(interval)


def main() -> None:
  probe = FR5Probe()
  probe.run(interval=1.0)


if __name__ == "__main__":
  main()
