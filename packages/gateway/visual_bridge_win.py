"""
Windows Visual Bridge → LOCUS SENSOR_STREAM

- SUB: WIN_VISUAL_STREAM (from sensor_visual_win.py, JPEG frames)
- PROC: YOLO model → visual feature vector
- PUB: SENSOR_STREAM (schema 유사: {type: 'visual', data: vec, timestamp, frame_id})
"""

import sys
import os
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parents[2]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
AI_PATH = PROJECT_ROOT / "ai"
if str(AI_PATH) not in sys.path:
    sys.path.append(str(AI_PATH))

import time
import zmq
import numpy as np
import cv2

from packages.config.zmq_endpoints import WIN_VISUAL_STREAM, SENSOR_STREAM

YOLO_MODEL_PATH = (Path(__file__).parent / "../ai/models/yolo/best.pt").resolve()
from ultralytics import YOLO



class WindowsVisualBridge:
    def __init__(self) -> None:
        print("=" * 60)
        print("Windows Visual → YOLO Bridge Initializing...")
        print("=" * 60)

        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(WIN_VISUAL_STREAM)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[Bridge] SUB visual from {WIN_VISUAL_STREAM}")

        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(SENSOR_STREAM)
        print(f"[Bridge] PUB features to {SENSOR_STREAM}")

        print(f"[Bridge] Loading YOLO model from: {YOLO_MODEL_PATH}")
        self.yolo = YOLO(YOLO_MODEL_PATH)
        print("[Bridge] YOLO ready!")

    def _extract_features(self, frame_bgr: np.ndarray) -> np.ndarray:
        """
        YOLO detection 결과를 LOCUS용 visual feature 벡터로 변환.
        여기 구현은 기존 sensor_visual.py의 로직을 그대로 복붙하는 게 베스트.

        지금은 placeholder로 일단 14차원 zero 벡터를 보냄.
        나중에:
          - self.yolo(frame_bgr) 호출
          - class별 카운트/존재 여부를 one-hot or frequency로 맵핑
        """
        # TODO: 기존 sensor_visual.py의 feature 추출 로직 이식
        feature_dim = 14  # DatasetBuilder.visual_dim이랑 맞춰야 함
        return np.zeros(feature_dim, dtype=np.float32)

    def run(self) -> None:
        frame_count = 0
        print("\n[Bridge] Visual bridge loop started. Ctrl+C to stop.\n")
        try:
            while True:
                msg = self.sub.recv_pyobj()
                ts = msg.get("timestamp", time.time())
                jpeg_bytes = msg.get("image_jpeg", None)

                if jpeg_bytes is None:
                    print("[Bridge] No image_jpeg in message, skipping.")
                    continue

                # JPEG → BGR 이미지
                buf = np.frombuffer(jpeg_bytes, dtype=np.uint8)
                frame = cv2.imdecode(buf, cv2.IMREAD_COLOR)
                if frame is None:
                    print("[Bridge] Failed to decode JPEG, skipping.")
                    continue

                # YOLO feature 추출
                feats = self._extract_features(frame)

                out = {
                    "type": "visual",
                    "data": feats,
                    "timestamp": ts,
                    "frame_id": msg.get("frame_id", frame_count),
                }

                self.pub.send_pyobj(out)
                print(f"[Bridge] Sent visual features frame={frame_count}")
                frame_count += 1

        except KeyboardInterrupt:
            print("\n[Bridge] Stopping visual bridge...")

        finally:
            self.sub.close()
            self.pub.close()
            print("[Bridge] Visual bridge stopped.")


if __name__ == "__main__":
    bridge = WindowsVisualBridge()
    bridge.run()
