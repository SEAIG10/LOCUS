"""
Policy Bridge
-------------
FR3(GRU Predictor) → FR5(Policy Engine & Cleaning Executor) 사이를 ZeroMQ로 연결합니다.

- GRU Predictor가 FR5 버스(`POLICY_STREAM`)에 예측 결과를 퍼블리시하면,
  본 브리지가 구독하여 CleaningExecutor에 전달합니다.
- 동시에, 정책/청소 결정을 JSONL 로그로 남겨
  FR5 대시보드 프로브가 `fr5_live.json`을 생성할 수 있게 합니다.
"""

from __future__ import annotations

import json
import os
import sys
import time
from dataclasses import asdict, dataclass
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
import zmq

# ---------- 프로젝트 경로 세팅 ----------
PROJECT_ROOT = Path(__file__).resolve().parents[2]  # .../LOCUS
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

from packages.ai.realtime.cleaning_executor import CleaningExecutor  # type: ignore
from packages.config.zmq_endpoints import POLICY_STREAM  # type: ignore

# ---------- FR5 대시보드용 정책 로그 경로 ----------
LOG_PATH = PROJECT_ROOT / "packages" / "gateway" / "logs" / "policy_events.log.jsonl"
LOG_PATH.parent.mkdir(parents=True, exist_ok=True)


def _log_policy_decision(event: Dict[str, Any]) -> None:
    """
    정책 결정을 JSONL 형태로 append.
    FR5 대시보드 프로브가 이 파일을 읽어 fr5_live.json으로 변환한다.
    """
    try:
        ev = dict(event)
        ev.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        ev.setdefault("role", "policy")
        ev.setdefault("type", "decision")
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(ev, ensure_ascii=False) + "\n")
    except Exception:
        # 대시보드용 로깅 실패는 정책 실행에 영향 주지 않도록 조용히 무시
        pass


@dataclass
class PolicyDecision:
    """FR5 대시보드와 공유할 정책 결정 스냅샷."""

    action: str
    reason: str
    eta: float
    battery: float
    map: List[Dict[str, Any]]
    path: List[Dict[str, Any]]
    notes: List[str]
    source: str
    prediction_index: Optional[int] = None
    raw_prediction: Optional[List[float]] = None


class PolicyBridge:
    def __init__(
        self,
        backend_url: str = "http://localhost:4000",
        device_id: str = "robot_001",
    ):
        # ZeroMQ SUB 초기화
        self.zmq_context = zmq.Context()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.connect(POLICY_STREAM)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[PolicyBridge] SUB connected to {POLICY_STREAM}")

        # CleaningExecutor 초기화
        self.cleaning_executor = CleaningExecutor(
            backend_url=backend_url,
            device_id=device_id,
            enable_backend=True,
        )

    # ------------------------------------------------------------------ helpers
    def _build_fallback_decision(
        self,
        prediction: np.ndarray,
        payload: Dict[str, Any],
    ) -> PolicyDecision:
        """
        CleaningExecutor가 dict를 반환하지 않는 경우를 대비한 fallback.
        단순히 prediction 값만 기반으로 최소한의 정보라도 남긴다.
        """
        scores = prediction.astype(float).tolist()
        map_rows = [
            {"zone": f"zone_{i}", "score": float(s)} for i, s in enumerate(scores)
        ]

        # 가장 점수가 높은 존들을 경로처럼 보여주기 (상위 3개)
        top_indices = np.argsort(prediction)[::-1][:3]
        path_rows: List[Dict[str, Any]] = []
        for idx in top_indices:
            path_rows.append(
                {
                    "zone": f"zone_{int(idx)}",
                    "eta": 0.0,
                    "prob": float(prediction[idx]),
                }
            )

        return PolicyDecision(
            action="UNKNOWN",
            reason="CleaningExecutor returned no structured decision; fallback based on raw prediction.",
            eta=0.0,
            battery=0.0,
            map=map_rows,
            path=path_rows,
            notes=[
                "This entry was generated from PolicyBridge fallback logic.",
                "Update CleaningExecutor.handle_prediction_sync to return a decision dict for richer FR5 dashboards.",
            ],
            source="policy_bridge_fallback",
            prediction_index=payload.get("prediction_index"),
            raw_prediction=scores,
        )

    def _normalize_decision(
        self,
        decision: Dict[str, Any],
        prediction: np.ndarray,
        payload: Dict[str, Any],
    ) -> PolicyDecision:
        """
        CleaningExecutor에서 dict를 반환했다고 가정하고,
        FR5 대시보드가 기대하는 포맷으로 정리한다.
        기대 키:
          - action (str)
          - reason (str)
          - eta (float, 분 단위)
          - battery (float, %)
          - map_scores: {zone_name: score} 또는 이미 list[{"zone","score"}]
          - path: list[{"zone","eta","prob"}]
          - notes: list[str]
        """
        action = str(decision.get("action", "UNKNOWN"))
        reason = str(decision.get("reason", ""))

        eta = float(decision.get("eta", 0.0))
        battery = float(decision.get("battery", 0.0))

        raw_pred_list = prediction.astype(float).tolist()

        # map_scores 형태 지원: dict 또는 list
        map_field = decision.get("map_scores") or decision.get("map")
        map_rows: List[Dict[str, Any]] = []
        if isinstance(map_field, dict):
            for z, s in map_field.items():
                map_rows.append({"zone": str(z), "score": float(s)})
        elif isinstance(map_field, list):
            # 이미 [{"zone":..., "score":...}] 형태라고 가정
            for row in map_field:
                zone = str(row.get("zone", "unknown"))
                score = float(row.get("score", 0.0))
                map_rows.append({"zone": zone, "score": score})

        path_field = decision.get("path") or []
        path_rows: List[Dict[str, Any]] = []
        if isinstance(path_field, list):
            for row in path_field:
                zone = str(row.get("zone", "unknown"))
                eta_row = float(row.get("eta", 0.0))
                prob = float(row.get("prob", row.get("score", 0.0)))
                path_rows.append({"zone": zone, "eta": eta_row, "prob": prob})

        notes_raw = decision.get("notes") or []
        notes = [str(n) for n in notes_raw]

        return PolicyDecision(
            action=action,
            reason=reason,
            eta=eta,
            battery=battery,
            map=map_rows,
            path=path_rows,
            notes=notes,
            source=str(decision.get("source", "policy_bridge")),
            prediction_index=payload.get("prediction_index"),
            raw_prediction=raw_pred_list,
        )

    def _log_decision_from_executor(
        self,
        prediction: np.ndarray,
        payload: Dict[str, Any],
        result: Any,
    ) -> None:
        """
        CleaningExecutor 결과를 해석해서 PolicyDecision으로 변환 후 로그에 기록.
        """
        if isinstance(result, dict):
            decision = self._normalize_decision(result, prediction, payload)
        else:
            decision = self._build_fallback_decision(prediction, payload)

        _log_policy_decision(asdict(decision))

    # ------------------------------------------------------------------ main io
    def handle_message(self, message: Dict[str, Any]) -> None:
        payload = message.get("payload", {}) or {}
        prediction = np.array(payload.get("prediction", []), dtype=np.float32)
        if prediction.size == 0:
            return

        print(f"[PolicyBridge] Received prediction #{payload.get('prediction_index')}")

        # CleaningExecutor에 전달 (동기)
        started = time.time()
        try:
            result = self.cleaning_executor.handle_prediction_sync(prediction)
        except Exception as exc:
            # 정책 실행 실패도 로그에 남겨서 디버깅 가능하게
            print(f"[PolicyBridge] CleaningExecutor error: {exc}")
            _log_policy_decision(
                {
                    "source": "policy_bridge",
                    "error": str(exc),
                    "prediction_index": payload.get("prediction_index"),
                    "raw_prediction": prediction.astype(float).tolist(),
                    "action": "ERROR",
                    "reason": f"CleaningExecutor raised exception: {exc}",
                    "eta": 0.0,
                    "battery": 0.0,
                    "map": [],
                    "path": [],
                    "notes": ["See gateway logs for full traceback."],
                }
            )
            return

        elapsed_ms = int((time.time() - started) * 1000)
        print(
            f"[PolicyBridge] Cleaning decision processed "
            f"in {elapsed_ms} ms (prediction #{payload.get('prediction_index')})"
        )

        # FR5 대시보드용 정책 이벤트 로깅
        self._log_decision_from_executor(prediction, payload, result)

    def run(self) -> None:
        print("[PolicyBridge] Waiting for GRU predictions...")
        try:
            while True:
                message = self.zmq_socket.recv_pyobj()
                if not isinstance(message, dict):
                    # 구버전 퍼블리셔 호환용: dict가 아니라면 감싸서 payload에 넣는다.
                    message = {"payload": message}
                self.handle_message(message)
        except KeyboardInterrupt:
            print("\n[PolicyBridge] Interrupted, shutting down.")
        finally:
            self.zmq_socket.close()
            self.zmq_context.term()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="ZeroMQ Policy Bridge")
    parser.add_argument("--backend-url", type=str, default="http://localhost:4000")
    parser.add_argument("--device-id", type=str, default="robot_001")
    args = parser.parse_args()

    bridge = PolicyBridge(backend_url=args.backend_url, device_id=args.device_id)
    bridge.run()
