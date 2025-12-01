"""MQTT-based FedPer client that reuses the pretrained Keras GRU model."""

from __future__ import annotations

import base64
import json
import pickle
import queue
import threading
import time
from datetime import datetime
from pathlib import Path
from typing import Any, Dict, List, Tuple

import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf

from config import (
    CLIENT_ID,
    CONTEXT_MODE,
    LOCAL_BATCH_SIZE,
    LOCAL_CKPT_DIR,
    LOCAL_EPOCHS,
    LR,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_KEEPALIVE,
    MQTT_TOPIC_NAMESPACE,
    PRETRAINED_MODEL_PATH,
    TRAIN_DATASET_PATH,
)
from packages.ai.core.dataset.dataset_builder import DatasetBuilder

Weights = List[np.ndarray]

# ------------------------------------------------------------------ FL logging

LOG_PATH = Path(__file__).resolve().parent / "logs" / "fl_events.log.jsonl"


def _log_fl_event(event: Dict[str, Any]) -> None:
    """Append a single FL event as JSON line for FR4 dashboard probe."""
    try:
        LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
        event = dict(event)
        event.setdefault("ts", datetime.now().isoformat(timespec="seconds"))
        with LOG_PATH.open("a", encoding="utf-8") as f:
            f.write(json.dumps(event, ensure_ascii=False) + "\n")
    except Exception:
        # 대시보드용 로깅 실패는 학습에 영향 주지 않도록 무시
        pass


class MQTTFLClient:
    """Federated client that trains locally and exchanges weights via MQTT."""

    def __init__(
        self,
        client_id: str = CLIENT_ID,
        broker_host: str = MQTT_BROKER_HOST,
        broker_port: int = MQTT_BROKER_PORT,
        topic_namespace: str = MQTT_TOPIC_NAMESPACE,
        local_epochs: int = LOCAL_EPOCHS,
        learning_rate: float = LR,
        keepalive: int = MQTT_KEEPALIVE,
        model_path: str | Path = PRETRAINED_MODEL_PATH,
    ) -> None:
        self.client_id = client_id
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_namespace = topic_namespace.rstrip("/")
        self.local_epochs = local_epochs
        self.learning_rate = learning_rate
        self.keepalive = keepalive
        self.round_idx = 0

        self.topic_global_models = f"{self.topic_namespace}/global_model"
        self.topic_direct = f"{self.topic_namespace}/clients/{self.client_id}"
        self.topic_updates = f"{self.topic_namespace}/updates"
        self.topic_registrations = f"{self.topic_namespace}/registrations"

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found at {self.model_path}. "
                "Please ensure packages/ai/models/gru/gru_model.keras exists."
            )
        self.model = tf.keras.models.load_model(self.model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        (
            self.train_dataset,
            self.val_dataset,
            self.train_sample_count,
            self.val_sample_count,
        ) = self._prepare_datasets()

        self._event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._connected = threading.Event()
        self._mqtt = mqtt.Client(client_id=self.client_id, clean_session=True)
        self._mqtt.on_connect = self._on_connect
        self._mqtt.on_disconnect = self._on_disconnect
        self._mqtt.on_message = self._on_message

        # 현재 라운드 측정용 메타데이터
        self._current_round_started_at: float | None = None

        _log_fl_event(
            {
                "role": "client",
                "type": "client_init",
                "client_id": self.client_id,
                "train_samples": self.train_sample_count,
                "val_samples": self.val_sample_count,
            }
        )

    # --------------------------------------------------------------------- MQTT
    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Dict[str, Any],
        rc: int,
    ) -> None:
        if rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[FLClient:{self.client_id}] MQTT connect failed (rc={rc}).")
            _log_fl_event(
                {
                    "role": "client",
                    "type": "connect_failed",
                    "client_id": self.client_id,
                    "rc": rc,
                }
            )
            return
        client.subscribe([(self.topic_global_models, 1), (self.topic_direct, 1)])
        self._connected.set()
        print(
            f"[FLClient:{self.client_id}] Connected to MQTT broker "
            f"{self.broker_host}:{self.broker_port}"
        )
        _log_fl_event(
            {
                "role": "client",
                "type": "connect",
                "client_id": self.client_id,
                "broker": f"{self.broker_host}:{self.broker_port}",
            }
        )
        self._publish(
            self.topic_registrations,
            {
                "type": "register",
                "client_id": self.client_id,
            },
        )

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        print(f"[FLClient:{self.client_id}] Disconnected from MQTT broker (rc={rc}).")
        self._connected.clear()
        _log_fl_event(
            {
                "role": "client",
                "type": "disconnect",
                "client_id": self.client_id,
                "rc": rc,
            }
        )

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            print(f"[FLClient:{self.client_id}] Received malformed MQTT message.")
            return

        msg_type = payload.get("type")
        if msg_type in {"welcome", "global_model"}:
            encoded_state = payload.get("state")
            if not encoded_state:
                print(f"[FLClient:{self.client_id}] Missing state in message: {payload}")
                return
            try:
                weights = self._decode_weights(encoded_state)
            except (pickle.UnpicklingError, ValueError) as exc:
                print(f"[FLClient:{self.client_id}] Failed to decode state: {exc}")
                return
            round_idx = int(payload.get("round", self.round_idx))
            self._event_queue.put(
                {
                    "type": "model_broadcast",
                    "round": round_idx,
                    "state": weights,
                    "source": msg_type,
                }
            )
        elif msg_type == "update_received":
            # 서버가 업데이트 수신 확인
            _log_fl_event(
                {
                    "role": "client",
                    "type": "update_ack",
                    "client_id": self.client_id,
                    "round": payload.get("round"),
                }
            )

    def _publish(self, topic: str, message: Dict[str, Any]) -> None:
        payload = json.dumps(message).encode("utf-8")
        result = self._mqtt.publish(topic, payload, qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(
                f"[FLClient:{self.client_id}] MQTT publish failed (rc={result.rc}) "
                f"for topic {topic}"
            )

    @staticmethod
    def _encode_weights(weights: Weights) -> str:
        raw = pickle.dumps(weights)
        return base64.b64encode(raw).decode("ascii")

    @staticmethod
    def _decode_weights(encoded: str) -> Weights:
        raw = base64.b64decode(encoded.encode("ascii"))
        weights: Weights = pickle.loads(raw)
        return [np.array(w) for w in weights]

    # ------------------------------------------------------------------- public
    def run(self) -> None:
        self._mqtt.connect(self.broker_host, self.broker_port, self.keepalive)
        self._mqtt.loop_start()
        try:
            self._connected.wait()
            while True:
                event = self._event_queue.get()
                if event.get("type") == "model_broadcast":
                    self._handle_global_model(event)
        except KeyboardInterrupt:
            print(f"[FLClient:{self.client_id}] Interrupted, shutting down.")
        finally:
            self._mqtt.loop_stop()
            self._mqtt.disconnect()

    # ----------------------------------------------------------- training logic
    def _prepare_datasets(self) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
        builder = DatasetBuilder(mode=CONTEXT_MODE)
        dataset_path = Path(TRAIN_DATASET_PATH)
        expected_dim = builder.context_dim

        def needs_rebuild(x: np.ndarray) -> bool:
            return x.shape[-1] != expected_dim

        if dataset_path.exists():
            X_train, y_train, X_val, y_val = builder.load_dataset(str(dataset_path))
            if needs_rebuild(X_train) or needs_rebuild(X_val):
                print(
                    f"[FLClient:{self.client_id}] 재생성을 수행합니다. "
                    f"(현재 차원: {X_train.shape[-1]}, 기대 차원: {expected_dim})"
                )
                X_train, y_train, X_val, y_val = builder.build_dataset()
                builder.save_dataset(
                    X_train, y_train, X_val, y_val, save_path=str(dataset_path)
                )
        else:
            dataset_path.parent.mkdir(parents=True, exist_ok=True)
            X_train, y_train, X_val, y_val = builder.build_dataset()
            builder.save_dataset(
                X_train, y_train, X_val, y_val, save_path=str(dataset_path)
            )
        train_samples = len(X_train)
        val_samples = len(X_val)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(max(train_samples, 1), reshuffle_each_iteration=True)
            .batch(LOCAL_BATCH_SIZE)
        )
        val_dataset = (
            tf.data.Dataset.from_tensor_slices((X_val, y_val))
            .batch(LOCAL_BATCH_SIZE)
        )

        print(
            f"[FLClient:{self.client_id}] Prepared dataset "
            f"(train={train_samples}, val={val_samples})"
        )
        _log_fl_event(
            {
                "role": "client",
                "type": "dataset_ready",
                "client_id": self.client_id,
                "train_samples": train_samples,
                "val_samples": val_samples,
            }
        )
        return train_dataset, val_dataset, train_samples, val_samples

    def _handle_global_model(self, event: Dict[str, Any]) -> None:
        self.round_idx = int(event.get("round", self.round_idx))
        self.model.set_weights(event["state"])
        source = event.get("source", "global_model")
        print(f"[FLClient:{self.client_id}] Synced round {self.round_idx} ({source})")

        self._current_round_started_at = time.time()
        _log_fl_event(
            {
                "role": "client",
                "type": "round_start",
                "client_id": self.client_id,
                "round": self.round_idx,
                "source": source,
            }
        )

        train_loss, val_loss, val_acc = self._local_train()

        elapsed_ms: int | None = None
        if self._current_round_started_at is not None:
            elapsed_ms = int((time.time() - self._current_round_started_at) * 1000)

        self._publish_update(
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            latency_ms=elapsed_ms,
        )

    def _local_train(self) -> Tuple[float, float, float]:
        last_avg_loss = 0.0
        last_val_loss = 0.0
        last_val_acc = 0.0

        for epoch in range(1, self.local_epochs + 1):
            epoch_loss = 0.0
            processed = 0
            for batch_x, batch_y in self.train_dataset:
                result = self.model.train_on_batch(batch_x, batch_y)
                if isinstance(result, (list, tuple)):
                    loss_value = float(result[0])
                else:
                    loss_value = float(result)
                batch_size = batch_x.shape[0]
                if batch_size is None:
                    batch_size = int(tf.shape(batch_x)[0])
                else:
                    batch_size = int(batch_size)
                epoch_loss += loss_value * batch_size
                processed += batch_size
            avg_loss = epoch_loss / max(processed, 1)
            val_loss, val_acc = self._evaluate()
            last_avg_loss, last_val_loss, last_val_acc = avg_loss, val_loss, val_acc
            print(
                f"[FLClient:{self.client_id}] "
                f"Round {self.round_idx} Epoch {epoch}/{self.local_epochs} "
                f"train_loss={avg_loss:.4f} val_loss={val_loss:.4f} val_acc={val_acc:.3f}"
            )

        ckpt_path = Path(LOCAL_CKPT_DIR) / f"{self.client_id}_round_{self.round_idx}.keras"
        ckpt_path.parent.mkdir(parents=True, exist_ok=True)
        self.model.save(ckpt_path)

        return last_avg_loss, last_val_loss, last_val_acc

    def _evaluate(self) -> Tuple[float, float]:
        metrics = self.model.evaluate(self.val_dataset, verbose=0, return_dict=True)
        val_loss = float(metrics.get("loss", 0.0))
        val_acc = float(
            metrics.get("accuracy")
            or metrics.get("acc")
            or metrics.get("binary_accuracy", 0.0)
        )
        return val_loss, val_acc

    def _publish_update(
        self,
        train_loss: float,
        val_loss: float,
        val_acc: float,
        latency_ms: int | None,
    ) -> None:
        weights = self.model.get_weights()
        encoded_state = self._encode_weights(weights)
        payload: Dict[str, Any] = {
            "type": "update",
            "client_id": self.client_id,
            "round": self.round_idx,
            "state": encoded_state,
            "train_loss": float(train_loss),
            "val_loss": float(val_loss),
            "val_acc": float(val_acc),
        }
        if latency_ms is not None:
            payload["latency_ms"] = int(latency_ms)

        self._publish(self.topic_updates, payload)

        _log_fl_event(
            {
                "role": "client",
                "type": "local_done",
                "client_id": self.client_id,
                "round": self.round_idx,
                "loss": float(val_loss),
                "train_loss": float(train_loss),
                "val_acc": float(val_acc),
                "latency_ms": latency_ms,
            }
        )


__all__ = ["MQTTFLClient"]
