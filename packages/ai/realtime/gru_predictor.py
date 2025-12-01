"""
실시간 데모 - GRU 예측기
ZeroMQ로 센서 데이터를 수집 후 GRU 모델로 오염도를 예측합니다.
"""

import sys
import os
from pathlib import Path

# packages/ai 및 리포지토리 루트를 path에 추가
PROJECT_ROOT = Path(__file__).resolve().parents[3]
if str(PROJECT_ROOT) not in sys.path:
    sys.path.append(str(PROJECT_ROOT))
sys.path.append(os.path.join(os.path.dirname(__file__), '..'))

import zmq
import time
import numpy as np
import tensorflow as tf
from collections import deque

# 내부 모듈 임포트
from core.context_fusion.attention_context_encoder import create_attention_encoder
from core.model.gru_model import FedPerGRUModel
from realtime.utils import print_prediction_result, ZONES
from packages.config.zmq_endpoints import (
    SENSOR_STREAM,
    FEDERATED_STREAM,
    POLICY_STREAM,
)
from core.context_fusion.time_sync_buffer import TimeSyncBuffer

# 모델 경로
GRU_MODEL_PATH = os.path.join(os.path.dirname(__file__), '..', 'models', 'gru', 'gru_model.keras')

# 컨텍스트 버퍼 설정
CONTEXT_BUFFER_SIZE = 30  # 30 타임스텝

# ROS ApproximateTimeSynchronizer 방식의 동기화 설정
QUEUE_SIZE = 10  # 각 센서별 큐 크기
SLOP = 3       # 허용 오차 (초)


class GRUPredictor:
    """
    GRU Predictor
    ZeroMQ로 센서 데이터를 수신하여 AttentionContextEncoder를 거친 후, GRU 모델로 예측을 수행합니다.
    예측 결과는 FR4(Federated Learning)와 FR5(Policy Engine)로 ZeroMQ를 통해 송신됩니다.
    """

    def __init__(self):
        print("=" * 60)
        print("GRU Predictor Initializing...")
        print("=" * 60)

        # ZeroMQ Subscriber (센서 버스 호스트: SUB가 bind, 센서들은 connect)
        self.zmq_context = zmq.Context.instance()
        self.zmq_socket = self.zmq_context.socket(zmq.SUB)
        self.zmq_socket.bind(SENSOR_STREAM)
        self.zmq_socket.setsockopt_string(zmq.SUBSCRIBE, "")
        print(f"[GRU] ZeroMQ SUB bound to {SENSOR_STREAM}")
        print("[GRU] Subscribed to all sensor messages")

        # ZeroMQ Publishers (FR4, FR5)
        self.zmq_pub_federated = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_federated.bind(FEDERATED_STREAM)
        print(f"[GRU] ZeroMQ PUB bound to {FEDERATED_STREAM} for FR4")

        self.zmq_pub_policy = self.zmq_context.socket(zmq.PUB)
        self.zmq_pub_policy.bind(POLICY_STREAM)
        print(f"[GRU] ZeroMQ PUB bound to {POLICY_STREAM} for FR5")

        # 모델 로드
        print("\n[GRU] Loading models...")
        print("  1. AttentionContextEncoder...")
        self.attention_encoder = create_attention_encoder(
            visual_dim=14,
            audio_dim=17,
            pose_dim=51,
            spatial_dim=4,   # balcony, bedroom, kitchen, living_room
            time_dim=10,
            context_dim=160
        )
        print("     AttentionContextEncoder loaded!")

        print(f"  2. GRU Model from {GRU_MODEL_PATH}...")
        self.gru_model = FedPerGRUModel(num_zones=4, context_dim=160)
        self.gru_model.load(GRU_MODEL_PATH)
        print("     GRU Model loaded!")

        # TimeSyncBuffer: 실제 센서 타입에 맞게
        self.time_sync = TimeSyncBuffer(
            required_sensors=['visual', 'audio', 'pose', 'spatial', 'time'],
            queue_size=QUEUE_SIZE,
            slop=SLOP,
            on_sync=self.process_context,
        )

        # 30타임스텝 컨텍스트 버퍼
        self.context_buffer = deque(maxlen=CONTEXT_BUFFER_SIZE)

        # 통계
        self.timestep_count = 0
        self.prediction_count = 0

        print("\nGRU Predictor ready!\n")

    # ---------------------------------------------------------------------- ZMQ
    def receive_messages(self):
        """
        ZeroMQ 메시지를 TimeSyncBuffer로 전달.
        """
        try:
            if self.zmq_socket.poll(timeout=100):  # 100ms 폴링
                message = self.zmq_socket.recv_pyobj()

                sensor_type = message.get('type')
                timestamp = message.get('timestamp')
                data = message.get('data')

                if sensor_type is None or timestamp is None or data is None:
                    return

                # 디버그 로그
                print(f"[GRU] recv sensor={sensor_type} ts={timestamp}")

                # 타임싱크 버퍼에 push
                self.time_sync.push(sensor_type, timestamp, data)

        except Exception as e:
            print(f"Error in receive_messages: {e}")

    # ----------------------------------------------------------------- CONTEXT
    def process_context(self, sensor_data, timestamp_bucket):
        """
        동기화된 센서 데이터(visual, audio, pose, spatial, time)로 컨텍스트 생성.
        AttentionContextEncoder → 160차원 벡터 → 버퍼에 적재.
        """
        try:
            visual_vec = sensor_data['visual']
            audio_vec = sensor_data['audio']
            pose_vec = sensor_data['pose']
            spatial_vec = sensor_data['spatial']
            time_vec = sensor_data['time']

            context_dict = {
                'visual': tf.constant([visual_vec], dtype=tf.float32),
                'audio': tf.constant([audio_vec], dtype=tf.float32),
                'pose': tf.constant([pose_vec], dtype=tf.float32),
                'spatial': tf.constant([spatial_vec], dtype=tf.float32),
                'time': tf.constant([time_vec], dtype=tf.float32),
            }

            # (1, 160) → [0]으로 꺼내서 (160,) 벡터
            context_160 = self.attention_encoder(context_dict, training=False)[0].numpy()

            self.context_buffer.append(context_160)
            self.timestep_count += 1

            print(
                f"[{self.timestep_count:04d}] "
                f"Synced timestep @ {timestamp_bucket:.2f}s → "
                f"Buffer: {len(self.context_buffer)}/{CONTEXT_BUFFER_SIZE}"
            )

            if len(self.context_buffer) == CONTEXT_BUFFER_SIZE:
                self.predict()

        except Exception as e:
            print(f"Error in process_context: {e}")
            import traceback
            traceback.print_exc()

    # ---------------------------------------------------------------- PREDICT
    def predict(self):
        """
        GRU 모델로 예측 수행 후, 결과를 출력 및 FR4/FR5로 송신.
        """
        try:
            print("\n" + "=" * 60)
            print(f"Running GRU Prediction #{self.prediction_count + 1}...")
            print("=" * 60)

            X = np.array(self.context_buffer).reshape(1, CONTEXT_BUFFER_SIZE, 160)

            prediction = self.gru_model.predict(X)[0]

            print_prediction_result(prediction, ZONES)

            self.publish_prediction(prediction)

            self.prediction_count += 1

            self.context_buffer.clear()
            print(f"\nBuffer cleared. Collecting next {CONTEXT_BUFFER_SIZE} timesteps...")
            print("=" * 60 + "\n")

        except Exception as e:
            print(f"Error in predict: {e}")
            import traceback
            traceback.print_exc()

    # -------------------------------------------------------------- PUBLISHING
    def publish_prediction(self, prediction: np.ndarray):
        """
        예측 결과와 컨텍스트 시퀀스를 ZeroMQ로 송신 (FR4, FR5).
        """
        timestamp = time.time()
        context_window = np.array(self.context_buffer, dtype=np.float32).tolist()

        payload = {
            "timestamp": timestamp,
            "zones": ZONES,
            "prediction": prediction.astype(float).tolist(),
            "context_window": context_window,
            "sequence_length": CONTEXT_BUFFER_SIZE,
            "prediction_index": self.prediction_count,
        }

        try:
            self.zmq_pub_federated.send_pyobj(
                {
                    "source": "gru_predictor",
                    "target": "federated_learning",
                    "payload": payload,
                }
            )
            self.zmq_pub_policy.send_pyobj(
                {
                    "source": "gru_predictor",
                    "target": "policy_engine",
                    "payload": payload,
                }
            )
            print("📡 Published prediction to FR4(FedPer) and FR5(Policy) ZeroMQ buses.")
        except Exception as exc:
            print(f"Error while publishing prediction over ZeroMQ: {exc}")

    # -------------------------------------------------------------------- LOOP
    def run(self):
        """
        예측기 메인 루프 (폴링).
        """
        print("GRU Predictor started!")
        print(f"  - Waiting for {CONTEXT_BUFFER_SIZE} timesteps of synced sensor data...")
        print("  - Sensors expected: visual, audio, pose, spatial, time")
        print("  - Press Ctrl+C to quit\n")

        try:
            while True:
                self.receive_messages()
        except KeyboardInterrupt:
            print("\nKeyboard interrupt, stopping...")
        finally:
            self.cleanup()

    # ----------------------------------------------------------------- CLEANUP
    def cleanup(self):
        """사용한 리소스를 정리합니다."""
        print("\nCleaning up GRU Predictor...")
        self.zmq_socket.close()
        self.zmq_pub_federated.close()
        self.zmq_pub_policy.close()
        self.zmq_context.term()
        print("GRU Predictor stopped!")
        print("\nStatistics:")
        print(f"  - Total timesteps collected: {self.timestep_count}")
        print(f"  - Total predictions made: {self.prediction_count}")
        print(f"  - Sync failures (dropped): {self.time_sync.dropped}")


if __name__ == "__main__":
    predictor = GRUPredictor()
    predictor.run()
