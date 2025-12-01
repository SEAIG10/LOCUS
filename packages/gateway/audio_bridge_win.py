"""
Windows Audio Bridge → LOCUS SENSOR_STREAM

- SUB: WIN_AUDIO_STREAM (from sensor_audio_win.py, raw PCM audio)
- PROC: YAMNet 17-class head (YamnetProcessor)
- PUB: SENSOR_STREAM (same schema as original sensor_audio.py)
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

from packages.config.zmq_endpoints import WIN_AUDIO_STREAM, SENSOR_STREAM
from core.audio_recognition.yamnet_processor import YamnetProcessor


class WindowsAudioBridge:
    def __init__(self, sample_rate: int = 16000) -> None:
        self.sample_rate = sample_rate
        print("=" * 60)
        print("Windows Audio → YAMNet Bridge Initializing...")
        print("=" * 60)

        # ZMQ SUB (Windows raw audio)
        self.ctx = zmq.Context.instance()
        self.sub = self.ctx.socket(zmq.SUB)
        self.sub.connect(WIN_AUDIO_STREAM)
        self.sub.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[Bridge] SUB audio from {WIN_AUDIO_STREAM}")

        # ZMQ PUB (LOCUS internal sensor bus)
        self.pub = self.ctx.socket(zmq.PUB)
        self.pub.connect(SENSOR_STREAM)
        print(f"[Bridge] PUB features to {SENSOR_STREAM}")

        # YAMNet processor
        print("[Bridge] Loading YamnetProcessor...")
        self.yamnet = YamnetProcessor()
        print("[Bridge] YamnetProcessor ready!")

    def run(self) -> None:
        sample_count = 0
        print("\n[Bridge] Audio bridge loop started. Ctrl+C to stop.\n")
        try:
            while True:
                msg = self.sub.recv_pyobj()
                ts = msg.get("timestamp", time.time())
                pcm = np.asarray(msg.get("pcm", []), dtype=np.float32)

                if pcm.size == 0:
                    print("[Bridge] Received empty PCM, skipping.")
                    continue

                # YAMNet 17-class 확률 벡터 계산
                try:
                    probs = self.yamnet.get_audio_embedding(pcm, self.sample_rate)
                except Exception as e:
                    print(f"[Bridge] YAMNet error: {e}")
                    continue

                out = {
                    "type": "audio",
                    "data": probs,              # (17,) float vector
                    "timestamp": ts,
                    "sample_count": msg.get("sample_id", sample_count),
                }

                self.pub.send_pyobj(out)
                print(f"[Bridge] Sent audio features sample={sample_count}")
                sample_count += 1

        except KeyboardInterrupt:
            print("\n[Bridge] Stopping audio bridge...")

        finally:
            self.sub.close()
            self.pub.close()
            # ctx는 다른 모듈에서도 쓸 수 있으니 term()는 여기서 안 함
            print("[Bridge] Audio bridge stopped.")


if __name__ == "__main__":
    bridge = WindowsAudioBridge(sample_rate=16000)
    bridge.run()
