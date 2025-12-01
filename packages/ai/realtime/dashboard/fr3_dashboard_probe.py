"""
FR3 Dashboard Probe
-------------------

GRU Predictor / Attention Encoder / Context Summary의 ZMQ 스트림을 받아
웹 대시보드에서 읽을 수 있는 fr3_live.json 형태로 스냅샷을 지속적으로 기록한다.
"""

from __future__ import annotations
import json
import time
from pathlib import Path
from typing import Dict, Any, List
from dataclasses import dataclass, asdict

import zmq
from packages.config import zmq_endpoints

ROOT = Path(__file__).resolve().parents[3]
PUBLIC_DIR = ROOT / "packages" / "dashboard" / "public"
SNAPSHOT_PATH = PUBLIC_DIR / "fr3_live.json"


@dataclass
class TimelineEntry:
    time: str
    summary: str


@dataclass
class AttentionEntry:
    label: str
    weight: float


@dataclass
class PredictionEntry:
    zone: str
    value: float


class FR3Probe:

    def __init__(self):
        self.ctx = zmq.Context.instance()

        # ZMQ Endpoints (네 프로젝트에 맞게 수정)
        self.pred_ep = zmq_endpoints.GRU_PRED_PUB      # 예: tcp://127.0.0.1:7001
        self.context_ep = zmq_endpoints.CONTEXT_SUM_PUB  # 예: tcp://127.0.0.1:7002

        # Subscribers
        self.pred_sub = self._make_sub_socket(self.pred_ep)
        self.ctx_sub = self._make_sub_socket(self.context_ep)

        # 마지막으로 받은 값들
        self.latest_prediction: Dict[str, float] = {}
        self.latest_attention: List[float] = []
        self.latest_timeline: List[str] = []

        PUBLIC_DIR.mkdir(parents=True, exist_ok=True)

    def _make_sub_socket(self, endpoint: str):
        s = self.ctx.socket(zmq.SUB)
        s.connect(endpoint)
        s.setsockopt_string(zmq.SUBSCRIBE, "")
        return s

    def _parse_prediction_msg(self, raw: bytes):
        """GRU Predictor ZMQ 데이터 파싱"""
        try:
            msg = json.loads(raw.decode())
            pred = msg.get("prediction", {})
            attn = msg.get("attention", [])
            self.latest_prediction = pred
            self.latest_attention = attn
        except Exception:
            pass

    def _parse_context_msg(self, raw: bytes):
        """
        Context Summary 메시지 파싱

        TimeSyncBuffer가 압축된 인간 친화적 summary를 제공한다고 가정:
        예:
          { "timeline": ["주방 활동 증가", "거실 체류", ... ] }
        """
        try:
            msg = json.loads(raw.decode())
            timeline = msg.get("timeline", [])
            self.latest_timeline = timeline
        except Exception:
            pass

    def _build_snapshot(self) -> Dict[str, Any]:
        # timeline 구성이 문자열 리스트라고 가정 → 대시보드 형식으로 변환
        timeline: List[TimelineEntry] = []
        for i, text in enumerate(self.latest_timeline[-4:]):
            t_label = f"t-{(len(self.latest_timeline)-i-1)*10}s" if i < len(self.latest_timeline)-1 else "now"
            timeline.append(TimelineEntry(time=t_label, summary=text))

        # attention 요약
        attn: List[AttentionEntry] = []
        for i, w in enumerate(self.latest_attention):
            attn.append(AttentionEntry(label=f"Step {i}", weight=float(w)))

        # zone predictions → 리스트 형태 변환
        pred_list: List[PredictionEntry] = [
            PredictionEntry(zone=z, value=float(v))
            for z, v in self.latest_prediction.items()
        ]

        return {
            "timeline": [asdict(t) for t in timeline],
            "attention": [asdict(a) for a in attn],
            "prediction": [asdict(p) for p in pred_list]
        }

    def run(self, interval: float = 0.5):
        print(f"[FR3-Probe] Writing snapshot to: {SNAPSHOT_PATH}")

        poller = zmq.Poller()
        poller.register(self.pred_sub, zmq.POLLIN)
        poller.register(self.ctx_sub, zmq.POLLIN)

        while True:
            socks = dict(poller.poll(timeout=200))

            if self.pred_sub in socks:
                raw = self.pred_sub.recv()
                self._parse_prediction_msg(raw)

            if self.ctx_sub in socks:
                raw = self.ctx_sub.recv()
                self._parse_context_msg(raw)

            snapshot = self._build_snapshot()

            try:
                with SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
                    json.dump(snapshot, f, ensure_ascii=False, indent=2)
            except Exception as e:
                print(f"[FR3-Probe] write error: {e}")

            time.sleep(interval)


def main():
    probe = FR3Probe()
    probe.run(interval=0.5)


if __name__ == "__main__":
    main()
