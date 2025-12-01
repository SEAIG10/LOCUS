# packages/dashboard/core/loaders.py

from __future__ import annotations

import json
import sqlite3
from pathlib import Path
from typing import Any, Dict, List


ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_DATA = ROOT / "packages" / "dashboard" / "data"
CONFIG_DIR = ROOT / "packages" / "config"
AI_DATA_DIR = ROOT / "packages" / "ai" / "data"


# ===== 이미 있는 함수라고 가정 =====
def load_sample() -> Dict[str, Any]:
    with open(DASHBOARD_DATA / "dashboard_samples.json", "r", encoding="utf-8") as f:
        return json.load(f)


# ===== FR1 전용 라이브 로더 =====

def _load_fr1_zones_from_config() -> List[Dict[str, Any]]:
    """zones_config.json에서 존 정보를 읽어와 대시보드용으로 정리."""
    zones_path = CONFIG_DIR / "zones_config.json"
    if not zones_path.exists():
        return []

    with open(zones_path, "r", encoding="utf-8") as f:
        cfg = json.load(f)

    zones: List[Dict[str, Any]] = []
    # zones_config.json 구조에 맞게 필드 매핑
    for z in cfg.get("zones", []):
        zones.append(
            {
                "name": z.get("name") or z.get("id") or "?",
                # 아래 값들은 실제 구조에 맞게 필요하면 수정
                "occupancy": z.get("occupancy", "-"),
                "humidity": z.get("humidity", "-"),
                "lighting": z.get("lighting", "-"),
            }
        )
    return zones


def _load_fr1_tracker_from_db() -> Dict[str, Any]:
    """
    예시: AI 쪽에서 쓰는 context DB나 pose 로그에서
    가장 최근 위치/존 정보를 한 건 읽어오는 형태.
    실제 테이블/컬럼 이름에 맞춰 쿼리는 수정 필요.
    """
    db_path = AI_DATA_DIR / "context_vectors.db"
    if not db_path.exists():
        return {
            "status": "offline",
            "note": "context_vectors.db not found",
        }

    conn = sqlite3.connect(db_path)
    try:
        cur = conn.cursor()
        # 여기는 네가 실제 사용하는 테이블/컬럼에 맞게 바꿔야 함
        # 예시:
        #   table: pose_log(zone_id TEXT, x REAL, y REAL, ts INTEGER)
        cur.execute(
            """
            SELECT zone_id, x, y, ts
            FROM pose_log
            ORDER BY ts DESC
            LIMIT 1
            """
        )
        row = cur.fetchone()
        if not row:
            return {
                "status": "no-data",
            }
        zone_id, x, y, ts = row
        return {
            "status": "online",
            "zone_id": zone_id,
            "x": x,
            "y": y,
            "last_seen": ts,
        }
    finally:
        conn.close()


def _load_fr1_events_from_log(limit: int = 20) -> List[Dict[str, Any]]:
    """
    예시: FR1 관련 이벤트 로그(예: JSON Lines 파일)에서 최근 이벤트를 읽어옴.
    아직 이런 로그를 안 쓰고 있다면, 일단 빈 리스트 반환만 해도 됨.
    """
    log_path = DASHBOARD_DATA / "fr1_events.log.jsonl"
    if not log_path.exists():
        return []

    events: List[Dict[str, Any]] = []
    with open(log_path, "r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue
            try:
                ev = json.loads(line)
            except json.JSONDecodeError:
                continue
            events.append(ev)

    events = events[-limit:]
    rows: List[Dict[str, Any]] = []
    for ev in events:
        rows.append(
            {
                "time": ev.get("ts") or ev.get("timestamp") or "",
                "label": ev.get("label") or ev.get("event") or "",
            }
        )
    return rows


def get_fr1_data(source: str = "sample") -> Dict[str, Any]:
    """
    FR1 대시보드에서 사용할 데이터 스냅샷을 반환.
    - source == "sample": 기존 샘플 JSON 사용
    - source == "live"  : 실제 모듈이 만든 config/DB/로그를 읽어서 구성
    """
    if source == "sample":
        return load_sample()["fr1"]

    # live
    zones = _load_fr1_zones_from_config()
    tracker = _load_fr1_tracker_from_db()
    events = _load_fr1_events_from_log()

    return {
        "zones": zones,
        "tracker": tracker,
        "events": events,
    }
