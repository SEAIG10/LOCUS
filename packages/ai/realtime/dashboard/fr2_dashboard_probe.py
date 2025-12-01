"""
FR2 Dashboard Probe

YOLO / YAMNet / TimeSyncBuffer ZMQ 스트림을 구독해서,
웹 대시보드가 읽어갈 수 있는 fr2_live.json 스냅샷을 계속 갱신한다.
"""

from __future__ import annotations

import json
import time
from collections import deque
from dataclasses import dataclass, asdict
from pathlib import Path
from typing import Deque, List

import zmq

# 중앙 설정에서 ZMQ endpoint 가져오기
from packages.config import zmq_endpoints  # 실제 모듈 경로에 맞게 조정

ROOT = Path(__file__).resolve().parents[3]
DASHBOARD_PUBLIC = ROOT / "packages" / "dashboard" / "public"
SNAPSHOT_PATH = DASHBOARD_PUBLIC / "fr2_live.json"


@dataclass
class YoloEvent:
  label: str
  confidence: float


@dataclass
class YamnetEvent:
  label: str
  score: float


@dataclass
class SyncStep:
  step: str
  latency: float


class Fr2Probe:
  def __init__(self, max_items: int = 10) -> None:
    self.ctx = zmq.Context.instance()

    # 여기서 endpoint 이름/주소는 zmq_endpoints에 맞게 수정 필요
    visual_ep = zmq_endpoints.VISUAL_PUB  # 예: "tcp://127.0.0.1:6001"
    audio_ep = zmq_endpoints.AUDIO_PUB    # 예: "tcp://127.0.0.1:6002"
    sync_ep = zmq_endpoints.TIMESYNC_PUB  # 예: "tcp://127.0.0.1:6003"

    self.yolo_sub = self.ctx.socket(zmq.SUB)
    self.yolo_sub.connect(visual_ep)
    self.yolo_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    self.yamnet_sub = self.ctx.socket(zmq.SUB)
    self.yamnet_sub.connect(audio_ep)
    self.yamnet_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    self.sync_sub = self.ctx.socket(zmq.SUB)
    self.sync_sub.connect(sync_ep)
    self.sync_sub.setsockopt_string(zmq.SUBSCRIBE, "")

    self.poller = zmq.Poller()
    self.poller.register(self.yolo_sub, zmq.POLLIN)
    self.poller.register(self.yamnet_sub, zmq.POLLIN)
    self.poller.register(self.sync_sub, zmq.POLLIN)

    self.yolo_events: Deque[YoloEvent] = deque(maxlen=max_items)
    self.yamnet_events: Deque[YamnetEvent] = deque(maxlen=max_items)
    self.sync_events: Deque[SyncStep] = deque(maxlen=max_items)

  def _handle_yolo_msg(self, raw: bytes) -> None:
    """
    YOLO Publisher가 내보내는 메시지 포맷에 맞게 파싱.

    예시:
      b'{"label": "person_cooking", "confidence": 0.92}'
    """
    try:
      msg = json.loads(raw.decode("utf-8"))
      label = msg.get("label") or msg.get("event") or "unknown"
      conf = float(msg.get("confidence", 0.0))
      self.yolo_events.append(YoloEvent(label=label, confidence=conf))
    except Exception:
      return

  def _handle_yamnet_msg(self, raw: bytes) -> None:
    """
    YAMNet Publisher 메시지 포맷 예시:

      b'{"top_classes": [{"label": "Dishes", "score": 0.81}, ...]}'
    """
    try:
      msg = json.loads(raw.decode("utf-8"))
      top_classes = msg.get("top_classes") or msg.get("classes") or []
      for c in top_classes[:3]:
        label = c.get("label") or "unknown"
        score = float(c.get("score", 0.0))
        self.yamnet_events.append(YamnetEvent(label=label, score=score))
    except Exception:
      return

  def _handle_sync_msg(self, raw: bytes) -> None:
    """
    TimeSyncBuffer Publisher 메시지 포맷 예시:

      b'{"step": "t-3s", "latency_ms": 42}'
    """
    try:
      msg = json.loads(raw.decode("utf-8"))
      step = msg.get("step") or msg.get("ts_label") or "t"
      latency = float(msg.get("latency_ms", msg.get("latency", 0.0)))
      self.sync_events.append(SyncStep(step=step, latency=latency))
    except Exception:
      return

  def _build_snapshot(self) -> dict:
    # 최근 값 중 상위 몇 개만 사용
    yolo = [asdict(e) for e in list(self.yolo_events)[-5:]]
    yamnet = [asdict(e) for e in list(self.yamnet_events)[-5:]]
    sync = [asdict(e) for e in list(self.sync_events)[-4:]]
    return {"yolo": yolo, "yamnet": yamnet, "sync": sync}

  def run(self, interval_sec: float = 1.0) -> None:
    SNAPSHOT_PATH.parent.mkdir(parents=True, exist_ok=True)
    print(f"[FR2-Probe] Writing snapshot to: {SNAPSHOT_PATH}")

    while True:
      socks = dict(self.poller.poll(timeout=int(interval_sec * 1000)))
      if self.yolo_sub in socks and socks[self.yolo_sub] == zmq.POLLIN:
        raw = self.yolo_sub.recv()
        self._handle_yolo_msg(raw)

      if self.yamnet_sub in socks and socks[self.yamnet_sub] == zmq.POLLIN:
        raw = self.yamnet_sub.recv()
        self._handle_yamnet_msg(raw)

      if self.sync_sub in socks and socks[self.sync_sub] == zmq.POLLIN:
        raw = self.sync_sub.recv()
        self._handle_sync_msg(raw)

      snapshot = self._build_snapshot()
      try:
        with SNAPSHOT_PATH.open("w", encoding="utf-8") as f:
          json.dump(snapshot, f, ensure_ascii=False, indent=2)
      except Exception as e:
        print(f"[FR2-Probe] Failed to write snapshot: {e}")

      # 너무 빡세지 않도록 살짝 sleep
      time.sleep(interval_sec)


def main() -> None:
  probe = Fr2Probe(max_items=20)
  probe.run(interval_sec=1.0)


if __name__ == "__main__":
  main()
