from __future__ import annotations

import json
import sqlite3
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import List, Dict, Any, Optional

from packages.config import settings  # 필요하면 사용
from packages.config import zones_config  # 있으면 사용, 없으면 json 직접 읽기

ROOT = Path(__file__).resolve().parents[3]
DASH_PUBLIC = ROOT / "packages" / "dashboard" / "public"
SNAPSHOT_PATH = DASH_PUBLIC / "fr1_live.json"

DB_PATH = ROOT / "packages" / "ai" / "data" / "context_vectors.db"  # 실제 DB에 맞게 조정


@dataclass
class ZoneSnapshot:
    name: str
    occupancy: str
    humidity: str = "-"
    lighting: str = "-"
    last_seen: str = "-"


@dataclass
class TrackerSnapshot:
    status: str
    device: str
    zone: Optional[str] = None
    x: Optional[float] = None
    y: Optional[float] = None
    last_seen: Optional[str] = None
    latency_ms: Optional[float] = None


@dataclass
class EventSnapshot:
    time: str
    label: str


class FR1Probe:
    def __init__(self) -> None:
        DASH_PUBLIC.mkdir(parents=True, exist_ok=True)

    # ---------- 존 정보 로드 ----------

    def load_zones_from_config(self) -> List[ZoneSnapshot]:
        """
        zones_config.json 에서 존 기본 정보 로딩
        """
        cfg_path = ROOT / "packages" / "config" / "zones_config.json"
        if not cfg_path.exists():
            return []

        with cfg_path.open("r", encoding="utf-8") as f:
            data = json.load(f)

        zones: List[ZoneSnapshot] = []
        for z in data.get("zones", []):
            name = z.get("name") or z.get("id") or "Unknown"
            # 여기 humidity/lighting은 아직 없을 수도 있으니 기본값 처리
            zones.append(
                ZoneSnapshot(
                    name=name,
                    occupancy="Unknown",
                    humidity=z.get("humidity", "-"),
                    lighting=z.get("lighting", "-"),
                )
            )
        return zones

    # ---------- DB에서 tracker / 존 활동 읽기 ----------

    def _connect_db(self) -> sqlite3.Connection:
        return sqlite3.connect(DB_PATH)

    def load_latest_tracker(self) -> TrackerSnapshot:
        """
        pose/log DB에서 가장 최근 위치 1개를 읽어와 tracker 정보 구성.
        실제 테이블/컬럼 이름에 맞게 SQL을 수정해야 한다.
        """
        if not DB_PATH.exists():
            return TrackerSnapshot(status="offline", device="N/A")

        conn = self._connect_db()
        try:
            cur = conn.cursor()
            # TODO: 실제 스키마에 맞게 수정
            # 예시: pose_log(zone_id TEXT, x REAL, y REAL, ts TEXT, source TEXT, latency_ms REAL)
            cur.execute(
                """
                SELECT zone_id, x, y, ts, source, latency_ms
                FROM pose_log
                ORDER BY ts DESC
                LIMIT 1
                """
            )
            row = cur.fetchone()
            if not row:
                return TrackerSnapshot(status="idle", device="mobile-tracker")

            zone_id, x, y, ts, source, latency = row
            return TrackerSnapshot(
                status="online",
                device=source or "mobile-tracker",
                zone=zone_id,
                x=x,
                y=y,
                last_seen=str(ts),
                latency_ms=float(latency) if latency is not None else None,
            )
        except Exception:
            return TrackerSnapshot(status="error", device="mobile-tracker")
        finally:
            conn.close()

    def load_zone_activity(self) -> Dict[str, Dict[str, Any]]:
        """
        각 존별 마지막 방문 시간/활동 정도를 계산.
        실제 스키마에 맞게 구현.

        리턴 예시:
          {
            "Living Room": {"last_seen": "21:32:10", "occupancy": "Active"},
            "Bedroom": {"last_seen": "21:20:03", "occupancy": "Idle"}
          }
        """
        if not DB_PATH.exists():
            return {}

        conn = self._connect_db()
        activity: Dict[str, Dict[str, Any]] = {}
        try:
            cur = conn.cursor()
            # TODO: 실제 region/zone 로그 테이블에 맞게 수정
            # 예시: region_log(zone_id TEXT, ts TEXT)
            cur.execute(
                """
                SELECT zone_id, MAX(ts) as last_ts
                FROM region_log
                GROUP BY zone_id
                """
            )
            rows = cur.fetchall()
            for zone_id, last_ts in rows:
                # 간단히 "얼마나 최근이냐"로 occupancy를 나누는 예시
                # 실제로는 now - last_ts 계산해서 Active/Idle/Empty 분기
                occupancy = "Active"
                activity[zone_id] = {
                    "last_seen": str(last_ts),
                    "occupancy": occupancy,
                }
        except Exception:
            return {}
        finally:
            conn.close()

        return activity

    def load_recent_events(self, limit: int = 10) -> List[EventSnapshot]:
        """
        최근 FR1 관련 이벤트 N개를 읽는다.
        - region_log 테이블에서 뽑거나
        - 별도 jsonl 로그 파일에서 읽어도 됨.
        여기서는 예시로 jsonl 사용.
        """
        log_path = ROOT / "packages" / "dashboard" / "data" / "fr1_events.log.jsonl"
        if not log_path.exists():
            return []

        events: List[EventSnapshot] = []
        with log_path.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                ts = obj.get("ts") or obj.get("time") or ""
                label = obj.get("label") or obj.get("event") or ""
                events.append(EventSnapshot(time=str(ts), label=label))

        return events[-limit:]

    # ---------- 전체 스냅샷 빌드 + 파일로 쓰기 ----------

    def build_snapshot(self) -> Dict[str, Any]:
        zones = self.load_zones_from_config()
        tracker = self.load_latest_tracker()
        activity_map = self.load_zone_activity()
        events = self.load_recent_events()

        # zones에 activity 정보 merge
        for z in zones:
            info = activity_map.get(z.name) or activity_map.get(z.name.lower())
            if info:
                z.occupancy = info.get("occupancy", z.occupancy)
                z.last_seen = info.get("last_seen", z.last_seen)

        return {
            "zones": [asdict(z) for z in zones],
            "tracker": asdict(tracker),
            "events": [asdict(e) for e in events],
        }

    def run(self, interval: float = 1.0) -> None:
        print(f"[FR1-Probe] writing snapshot to {SNAPSHOT_PATH}")
        while True:
            snapshot = self.build_snapshot()
            try:
                with SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[FR1-Probe] snapshot write error: {e}")
            time.sleep(interval)


def main() -> None:
    probe = FR1Probe()
    probe.run(interval=1.0)


if __name__ == "__main__":
    main()
