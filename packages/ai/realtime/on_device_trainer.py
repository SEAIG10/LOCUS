# LOCUS/packages/ai/realtime/on_device_trainer.py
from __future__ import annotations

"""
온디바이스 Head Layer 학습 모듈

역할:
1. 실시간 예측 입력(context sequence)과 실제 오염도 피드백을 버퍼에 저장
2. 버퍼가 일정 크기 이상 쌓이면 백그라운드에서 Head Layer만 학습
3. Base Layer는 frozen, Head Layer만 학습
4. 학습된 모델을 자동 저장하고, 필요시 MQTT로 상태 전송
"""

import os
import threading
import time
from collections import deque
from typing import Optional, Deque, Any, Dict

import numpy as np

from packages.federated.config import (
    SEQUENCE_LENGTH,
    CONTEXT_DIM,
    GRU_NUM_ZONES,
    FEDPER_MODEL_PATH,
)

from packages.ai.models.fedper_gru import FedPerGRUModel


class OnDeviceTrainer:
    """
    온디바이스 Head Layer 학습기

    - GRU Base Layer는 고정(frozen)
    - Head Layer만 로컬 피드백으로 지속 학습
    - 일정 샘플 이상 쌓이면 백그라운드 스레드로 학습 수행
    """

    def __init__(
        self,
        gru_model: FedPerGRUModel,
        buffer_size: int = 300,
        min_samples_for_training: int = 100,
        batch_size: int = 16,
        epochs_per_update: int = 5,
        learning_rate: float = 5e-4,
        auto_save_path: Optional[str] = None,
        mqtt_client: Any = None,
    ) -> None:
        """
        Args:
            gru_model: FedPerGRUModel 인스턴스 (base/head 포함 전체 모델)
            buffer_size: 버퍼 최대 길이 (샘플 개수 기준)
            min_samples_for_training: 학습을 시작하기 위한 최소 샘플 수
            batch_size: 학습 배치 크기
            epochs_per_update: 한 번 트리거될 때 수행할 epoch 수
            learning_rate: Head Layer 학습률
            auto_save_path: 학습 후 모델 자동 저장 경로 (.h5 / SavedModel 등)
            mqtt_client: 학습 상태를 publish할 수 있는 MQTT 클라이언트 (선택)
                         - 인터페이스: publish_training_status(status, **kwargs)
        """
        self.gru_model = gru_model
        self.buffer_size = buffer_size
        self.min_samples_for_training = min_samples_for_training
        self.batch_size = batch_size
        self.epochs_per_update = epochs_per_update
        self.learning_rate = learning_rate
        self.auto_save_path = auto_save_path
        self.mqtt_client = mqtt_client

        # 데이터 버퍼 (X: context sequences, y: pollution labels)
        self.X_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)
        self.y_buffer: Deque[np.ndarray] = deque(maxlen=buffer_size)

        # 학습 통계
        self.total_samples_collected: int = 0
        self.total_training_runs: int = 0
        self.last_training_time: Optional[float] = None

        # 백그라운드 학습 스레드
        self.training_thread: Optional[threading.Thread] = None
        self.is_training: bool = False
        self.stop_flag: bool = False

        # Base / Head trainable 설정
        self._freeze_base_layer()

        print(f"\n{'=' * 60}")
        print("OnDeviceTrainer Initialized")
        print(f"{'-' * 60}")
        print(f"  Buffer size: {buffer_size}")
        print(f"  Min samples for training: {min_samples_for_training}")
        print(f"  Batch size: {batch_size}")
        print(f"  Epochs per update: {epochs_per_update}")
        print(f"  Learning rate: {learning_rate}")
        print(f"  Auto-save path: {auto_save_path or 'Disabled'}")
        print(f"{'=' * 60}\n")

    # --------------------------------------------------------------------- util
    def _freeze_base_layer(self) -> None:
        """FedPer 구조에서 Base Layer는 학습하지 않고, Head Layer만 학습되도록 설정."""
        if getattr(self.gru_model, "base_model", None) is not None:
            self.gru_model.base_model.trainable = False
            print("Base Layer frozen (학습 안 함)")

        if getattr(self.gru_model, "head_model", None) is not None:
            self.gru_model.head_model.trainable = True
            print("Head Layer trainable (학습 대상)")

    # ----------------------------------------------------------------- data API
    def add_sample(self, context_sequence: np.ndarray, pollution_label: np.ndarray) -> None:
        """
        학습 샘플을 버퍼에 추가.

        Args:
            context_sequence: (SEQUENCE_LENGTH, CONTEXT_DIM) context 벡터 시퀀스
            pollution_label: (GRU_NUM_ZONES,) 각 zone별 오염도 또는 타깃 값
        """
        # Shape 검증
        if context_sequence.shape != (SEQUENCE_LENGTH, CONTEXT_DIM):
            print(
                f"[OnDeviceTrainer] Warning: Invalid context_sequence shape: "
                f"{context_sequence.shape}, expected: ({SEQUENCE_LENGTH}, {CONTEXT_DIM})"
            )
            return

        if pollution_label.shape != (GRU_NUM_ZONES,):
            print(
                f"[OnDeviceTrainer] Warning: Invalid pollution_label shape: "
                f"{pollution_label.shape}, expected: ({GRU_NUM_ZONES},)"
            )
            return

        # 버퍼에 추가
        self.X_buffer.append(context_sequence.astype(np.float32, copy=False))
        self.y_buffer.append(pollution_label.astype(np.float32, copy=False))
        self.total_samples_collected += 1

        # 학습 트리거 체크
        if (
            len(self.X_buffer) >= self.min_samples_for_training
            and not self.is_training
        ):
            print(
                f"\n[OnDeviceTrainer] Training triggered: "
                f"{len(self.X_buffer)} samples collected"
            )
            self.start_background_training()

    # -------------------------------------------------------------- training API
    def start_background_training(self) -> None:
        """백그라운드 스레드에서 학습을 시작."""
        if self.is_training:
            print("[OnDeviceTrainer] Warning: Training already in progress, skipping...")
            return

        self.training_thread = threading.Thread(
            target=self._training_worker, daemon=True
        )
        self.training_thread.start()

    def _training_worker(self) -> None:
        """백그라운드 학습 워커."""
        self.is_training = True

        try:
            print(f"\n{'=' * 60}")
            print("On-Device Head Training Started")
            print(f"{'-' * 60}")

            # 버퍼 snapshot
            X_train = np.array(list(self.X_buffer), dtype=np.float32)
            y_train = np.array(list(self.y_buffer), dtype=np.float32)

            n_samples = len(X_train)
            print(f"  Training samples: {n_samples}")
            print(f"  X shape: {X_train.shape}")
            print(f"  y shape: {y_train.shape}")

            if n_samples < self.min_samples_for_training:
                print(
                    "[OnDeviceTrainer] Not enough samples at training time, "
                    f"required={self.min_samples_for_training}, got={n_samples}"
                )
                return

            # Train / Val split (80 / 20)
            split_idx = int(n_samples * 0.8)
            X_tr, y_tr = X_train[:split_idx], y_train[:split_idx]
            X_val, y_val = X_train[split_idx:], y_train[split_idx:]

            print(f"  Train samples: {len(X_tr)}")
            print(f"  Val samples:   {len(X_val)}")

            # 모델 재컴파일 (Head만 학습, 회귀 문제 가정)
            self.gru_model.compile_model(
                learning_rate=self.learning_rate,
                loss="mse",
                metrics=["mae"],
            )

            # 학습
            print(f"\n  Training for {self.epochs_per_update} epochs...\n")
            history = self.gru_model.model.fit(
                X_tr,
                y_tr,
                validation_data=(X_val, y_val) if len(X_val) > 0 else None,
                epochs=self.epochs_per_update,
                batch_size=self.batch_size,
                verbose=1,
            )

            self.total_training_runs += 1
            self.last_training_time = time.time()

            final_loss = float(history.history["loss"][-1])
            final_val_loss = float(history.history["val_loss"][-1]) if "val_loss" in history.history else None

            print(f"\n{'=' * 60}")
            print("On-Device Training Complete")
            print(f"{'-' * 60}")
            print(f"  Final train loss: {final_loss:.6f}")
            if final_val_loss is not None:
                print(f"  Final val loss:   {final_val_loss:.6f}")
            print(f"  Total training runs: {self.total_training_runs}")
            print(f"  Total samples collected: {self.total_samples_collected}")
            print(f"{'=' * 60}\n")

            # 모델 자동 저장
            if self.auto_save_path:
                self.save_model()

            # MQTT로 상태 전송
            if self.mqtt_client is not None:
                self.mqtt_client.publish_training_status(
                    "completed",
                    samples_used=n_samples,
                    epochs=self.epochs_per_update,
                    final_train_loss=final_loss,
                    final_val_loss=final_val_loss,
                    total_training_runs=self.total_training_runs,
                )

        except Exception as e:  # noqa: BLE001
            print(f"\n[OnDeviceTrainer] Error during on-device training: {e}")
            import traceback

            traceback.print_exc()

            if self.mqtt_client is not None:
                self.mqtt_client.publish_training_status(
                    "failed",
                    error=str(e),
                )

        finally:
            self.is_training = False

    # ----------------------------------------------------------------- save/API
    def save_model(self, save_path: Optional[str] = None) -> None:
        """
        학습된 모델 저장.

        Args:
            save_path: 저장 경로 (None이면 auto_save_path 사용)
        """
        path = save_path or self.auto_save_path
        if path is None:
            print("[OnDeviceTrainer] Warning: No save path specified, skip saving")
            return

        try:
            self.gru_model.save(path)
            file_size_mb = os.path.getsize(path) / (1024 * 1024)
            print(f"[OnDeviceTrainer] Model saved: {path} ({file_size_mb:.2f} MB)")
        except Exception as e:  # noqa: BLE001
            print(f"[OnDeviceTrainer] Error: Failed to save model: {e}")

    # -------------------------------------------------------------- stats / ctrl
    def get_statistics(self) -> Dict[str, object]:
        """학습 통계를 dict로 반환."""
        return {
            "buffer_size": len(self.X_buffer),
            "total_samples_collected": self.total_samples_collected,
            "total_training_runs": self.total_training_runs,
            "last_training_time": self.last_training_time,
            "is_training": self.is_training,
        }

    def clear_buffer(self) -> None:
        """데이터 버퍼 초기화."""
        self.X_buffer.clear()
        self.y_buffer.clear()
        print("[OnDeviceTrainer] Training buffer cleared")

    def stop(self) -> None:
        """학습을 중지하고 리소스 정리."""
        self.stop_flag = True
        if self.training_thread and self.training_thread.is_alive():
            self.training_thread.join(timeout=5.0)
        print("[OnDeviceTrainer] OnDeviceTrainer stopped")


# ---------------------------------------------------------------------- simple test

def _test_on_device_trainer() -> None:
    gru_model = FedPerGRUModel(
        num_zones=GRU_NUM_ZONES,
        context_dim=CONTEXT_DIM,
    )
    gru_model.compile_model()

    trainer = OnDeviceTrainer(
        gru_model=gru_model,
        buffer_size=300,
        min_samples_for_training=50,
        batch_size=16,
        epochs_per_update=3,
        learning_rate=5e-4,
        auto_save_path=str(FEDPER_MODEL_PATH),
    )

if __name__ == "__main__":
    _test_on_device_trainer()
