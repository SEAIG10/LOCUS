"""
FR4 Dashboard Probe

packages/federated/logs/fl_events.log.jsonl 로그를 읽어
웹 대시보드에서 사용할 fr4_live.json 스냅샷을 주기적으로 생성한다.
"""

from __future__ import annotations

import json
import time
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Dict, Any, List, Optional

ROOT = Path(__file__).resolve().parents[3]
LOG_PATH = ROOT / "packages" / "federated" / "logs" / "fl_events.log.jsonl"
DASH_PUBLIC = ROOT / "packages" / "dashboard" / "public"
SNAPSHOT_PATH = DASH_PUBLIC / "fr4_live.json"


@dataclass
class Summary:
    global_round: int
    total_clients: int
    online_clients: int
    avg_loss: Optional[float]
    last_updated: str


@dataclass
class ClientState:
    id: str
    name: str
    status: str
    latency_ms: Optional[float]
    loss: Optional[float]
    rounds: Optional[int]


@dataclass
class EventRow:
    time: str
    source: str
    event: str


class FR4Probe:
    def __init__(self) -> None:
        DASH_PUBLIC.mkdir(parents=True, exist_ok=True)

    def _read_events(self, limit: int = 200) -> List[Dict[str, Any]]:
        if not LOG_PATH.exists():
            return []

        events: List[Dict[str, Any]] = []
        with LOG_PATH.open("r", encoding="utf-8") as f:
            for line in f:
                line = line.strip()
                if not line:
                    continue
                try:
                    obj = json.loads(line)
                except json.JSONDecodeError:
                    continue
                events.append(obj)

        # 마지막 limit개만 사용
        return events[-limit:]

    def _build_summary_and_clients(
        self, events: List[Dict[str, Any]]
    ) -> (Summary, List[ClientState], List[str], List[EventRow]):

        latest_round = 0
        latest_avg_loss: Optional[float] = None
        last_ts = ""

        # client_id → state
        clients: Dict[str, ClientState] = {}
        flow_texts: List[str] = []
        timeline_events: List[EventRow] = []

        for ev in events:
            ts = str(ev.get("ts") or ev.get("time") or "")
            role = ev.get("role", "")
            etype = ev.get("type", "")
            rnd = int(ev.get("round") or 0)

            if rnd > latest_round:
                latest_round = rnd

            # server round aggregation에서 avg_loss 가져오기
            if role == "server" and etype in ("round_agg", "round_end"):
                avg_loss_val = ev.get("avg_loss")
                if avg_loss_val is not None:
                    latest_avg_loss = float(avg_loss_val)
                    last_ts = ts or last_ts

            # client 상태 업데이트
            if role == "client":
                cid = str(ev.get("client_id") or "unknown")
                name = str(ev.get("client_name") or cid)
                loss_val = ev.get("loss")
                latency = ev.get("latency_ms")
                status = ev.get("status") or "online"

                st = clients.get(
                    cid,
                    ClientState(
                        id=cid,
                        name=name,
                        status=status,
                        latency_ms=None,
                        loss=None,
                        rounds=None,
                    ),
                )
                st.name = name
                st.status = status
                if latency is not None:
                    st.latency_ms = float(latency)
                if loss_val is not None:
                    st.loss = float(loss_val)
                # round 정보가 있다면 업데이트
                if rnd:
                    st.rounds = max(st.rounds or 0, rnd)
                clients[cid] = st

            # flow 텍스트 (간단 요약)
            desc = ev.get("desc")
            if not desc:
                if role == "server" and etype == "round_start":
                    desc = f"Round {rnd} 시작"
                elif role == "server" and etype == "round_agg":
                    desc = f"Round {rnd} aggregation (avg_loss={ev.get('avg_loss')})"
                elif role == "client" and etype == "local_done":
                    desc = f"{ev.get('client_id')} local_done (round={rnd}, loss={ev.get('loss')})"
                elif role == "server" and etype == "round_end":
                    desc = f"Round {rnd} 종료"
            if desc:
                flow_texts.append(str(desc))

            # 타임라인 이벤트
            src_label = role
            if role == "client":
                src_label = str(ev.get("client_id") or "client")
            if ts and etype:
                timeline_events.append(
                    EventRow(time=ts, source=src_label, event=desc or etype)
                )

        total_clients = len(clients)
        online_clients = sum(1 for c in clients.values() if c.status != "offline")

        summary = Summary(
            global_round=latest_round,
            total_clients=total_clients,
            online_clients=online_clients,
            avg_loss=latest_avg_loss,
            last_updated=last_ts,
        )

        # flow는 너무 길면 마지막 10개만
        flow_texts = flow_texts[-10:]
        # events도 마지막 20개만
        timeline_events = timeline_events[-20:]

        return summary, list(clients.values()), flow_texts, timeline_events

    def build_snapshot(self) -> Dict[str, Any]:
        events = self._read_events(limit=200)
        summary, clients, flow_texts, timeline_events = self._build_summary_and_clients(
            events
        )

        return {
            "summary": asdict(summary),
            "clients": [asdict(c) for c in clients],
            "flow": flow_texts,
            "events": [asdict(e) for e in timeline_events],
        }

    def run(self, interval: float = 1.0) -> None:
        print(f"[FR4-Probe] writing snapshot to {SNAPSHOT_PATH}")
        while True:
            snapshot = self.build_snapshot()
            try:
                with SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[FR4-Probe] snapshot write error: {e}")
            time.sleep(interval)


def main() -> None:
    probe = FR4Probe()
    probe.run(interval=1.0)


if __name__ == "__main__":
    main()
