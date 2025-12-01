"""
Federated Learning ZMQ Ingestor
--------------------------------
FR3(GRU Predictor)가 송신하는 컨텍스트 시퀀스/예측 결과를 ZeroMQ로 구독하여
FedPer 파이프라인에서 학습/모니터링용으로 활용할 수 있는 형태로 저장합니다.
"""

import json
import os
import sys
import time
from pathlib import Path
from typing import Any, Dict

sys.path.append(os.path.join(os.path.dirname(__file__), ".."))
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))

import zmq
import numpy as np
from packages.config.zmq_endpoints import FEDERATED_STREAM


class FederatedZMQIngestor:
    def __init__(self, output_dir: str = "results/zmq_stream"):
        self.output_dir = Path(output_dir)
        self.output_dir.mkdir(parents=True, exist_ok=True)

        self.ctx = zmq.Context()
        self.socket = self.ctx.socket(zmq.SUB)
        self.socket.connect(FEDERATED_STREAM)
        self.socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[FedZMQ] SUB connected to {FEDERATED_STREAM}")

    def handle_message(self, message: Dict[str, Any]):
        payload = message.get("payload", {})
        context = np.array(payload.get("context_window", []), dtype=np.float32)
        prediction = np.array(payload.get("prediction", []), dtype=np.float32)

        timestamp = payload.get("timestamp", time.time())
        ts_str = time.strftime("%Y%m%d_%H%M%S", time.localtime(timestamp))
        base = self.output_dir / f"sequence_{ts_str}_{payload.get('prediction_index', 0):04d}"

        np.savez_compressed(
            base.with_suffix(".npz"),
            context=context,
            prediction=prediction,
            timestamp=timestamp,
            zones=np.array(payload.get("zones", [])),
        )

        with base.with_suffix(".json").open("w", encoding="utf-8") as f:
            json.dump(payload, f, ensure_ascii=False, indent=2)

        print(
            f"[FedZMQ] Stored context window shape={context.shape} "
            f"prediction={prediction} → {base.name}"
        )

    def run(self):
        print("[FedZMQ] Waiting for GRU predictor stream...")
        try:
            while True:
                message = self.socket.recv_pyobj()
                self.handle_message(message)
        except KeyboardInterrupt:
            print("\n[FedZMQ] Interrupted, closing.")
        finally:
            self.socket.close()
            self.ctx.term()


if __name__ == "__main__":
    import argparse

    parser = argparse.ArgumentParser(description="Subscribe to FR3 → FR4 ZeroMQ stream.")
    parser.add_argument(
        "--output-dir",
        type=str,
        default="results/zmq_stream",
        help="Directory to store streamed context windows.",
    )
    args = parser.parse_args()

    ingestor = FederatedZMQIngestor(output_dir=args.output_dir)
    ingestor.run()
