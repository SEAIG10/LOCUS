"""MQTT-based FedPer aggregation server backed by the pretrained Keras model."""

from __future__ import annotations

import base64
import json
import pickle
import queue
import threading
from collections import defaultdict
from datetime import datetime
from pathlib import Path
from typing import Any, DefaultDict, Dict, List, Set

import numpy as np
import paho.mqtt.client as mqtt
import tensorflow as tf

from config import (
    CLIENTS_PER_ROUND,
    GLOBAL_CKPT_DIR,
    MQTT_BROKER_HOST,
    MQTT_BROKER_PORT,
    MQTT_KEEPALIVE,
    MQTT_TOPIC_NAMESPACE,
    PRETRAINED_MODEL_PATH,
)

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
        # 대시보드용 로깅 실패는 서버 동작에 영향 주지 않도록 무시
        pass


class MQTTFLServer:
    """Collects MQTT updates, averages model weights, and broadcasts new rounds."""

    def __init__(
        self,
        broker_host: str = MQTT_BROKER_HOST,
        broker_port: int = MQTT_BROKER_PORT,
        topic_namespace: str = MQTT_TOPIC_NAMESPACE,
        clients_per_round: int = CLIENTS_PER_ROUND,
        keepalive: int = MQTT_KEEPALIVE,
        model_path: str | Path = PRETRAINED_MODEL_PATH,
        server_id: str = "locus_fl_server",
    ) -> None:
        self.broker_host = broker_host
        self.broker_port = broker_port
        self.topic_namespace = topic_namespace.rstrip("/")
        self.clients_per_round = clients_per_round
        self.keepalive = keepalive

        self.topic_updates = f"{self.topic_namespace}/updates"
        self.topic_registrations = f"{self.topic_namespace}/registrations"
        self.topic_global_models = f"{self.topic_namespace}/global_model"
        self.topic_direct_prefix = f"{self.topic_namespace}/clients"

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found at {self.model_path}. "
                "Please ensure packages/ai/models/gru/gru_model.keras exists."
            )
        self.model = tf.keras.models.load_model(self.model_path)
        self.global_round = 0

        self.pending_updates: DefaultDict[int, List[Weights]] = defaultdict(list)
        self.pending_clients: DefaultDict[int, List[str]] = defaultdict(list)
        self.pending_losses: DefaultDict[int, List[float]] = defaultdict(list)
        self.registered_clients: Set[str] = set()

        self.ckpt_dir = Path(GLOBAL_CKPT_DIR)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        self._event_queue: queue.Queue[Dict[str, Any]] = queue.Queue()
        self._connected = threading.Event()
        self._mqtt = mqtt.Client(client_id=server_id, clean_session=True)
        self._mqtt.on_connect = self._on_connect
        self._mqtt.on_disconnect = self._on_disconnect
        self._mqtt.on_message = self._on_message

    # ------------------------------------------------------------------ helpers
    def _publish(self, topic: str, message: Dict[str, Any]) -> None:
        payload = json.dumps(message).encode("utf-8")
        result = self._mqtt.publish(topic, payload, qos=1)
        if result.rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[FLServer] Failed to publish to {topic} (rc={result.rc}).")

    @staticmethod
    def _encode_weights(weights: Weights) -> str:
        raw = pickle.dumps(weights)
        return base64.b64encode(raw).decode("ascii")

    @staticmethod
    def _decode_weights(encoded: str) -> Weights:
        raw = base64.b64decode(encoded.encode("ascii"))
        weights: Weights = pickle.loads(raw)
        return [np.array(w) for w in weights]

    def _direct_topic(self, client_id: str) -> str:
        return f"{self.topic_direct_prefix}/{client_id}"

    def _send_state(self, client_id: str | None, message_type: str) -> None:
        encoded_state = self._encode_weights(self.model.get_weights())
        payload = {
            "type": message_type,
            "round": self.global_round,
            "state": encoded_state,
        }
        topic = (
            self._direct_topic(client_id) if client_id else self.topic_global_models
        )
        self._publish(topic, payload)

    def _broadcast_state(self, client_ids: List[str], message_type: str) -> None:
        # 새로운 global_round에 대한 round_start 이벤트
        _log_fl_event(
            {
                "role": "server",
                "type": "round_start",
                "round": self.global_round,
                "targets": list(set(client_ids)),
                "message_type": message_type,
            }
        )
        self._send_state(None, message_type)
        for client_id in set(client_ids):
            self._send_state(client_id, message_type)

    def _send_ack(self, client_id: str, round_idx: int) -> None:
        self._publish(
            self._direct_topic(client_id),
            {"type": "update_received", "round": round_idx},
        )

    # ------------------------------------------------------------------- mqtt io
    def _on_connect(
        self,
        client: mqtt.Client,
        userdata: Any,
        flags: Dict[str, Any],
        rc: int,
    ) -> None:
        if rc != mqtt.MQTT_ERR_SUCCESS:
            print(f"[FLServer] MQTT connect failed (rc={rc}).")
            _log_fl_event(
                {
                    "role": "server",
                    "type": "connect_failed",
                    "rc": rc,
                }
            )
            return
        client.subscribe([(self.topic_registrations, 1), (self.topic_updates, 1)])
        self._connected.set()
        print(
            f"[FLServer] Connected to MQTT broker "
            f"{self.broker_host}:{self.broker_port}"
        )
        _log_fl_event(
            {
                "role": "server",
                "type": "connect",
                "broker": f"{self.broker_host}:{self.broker_port}",
            }
        )

    def _on_disconnect(self, client: mqtt.Client, userdata: Any, rc: int) -> None:
        print(f"[FLServer] Disconnected from MQTT broker (rc={rc}).")
        self._connected.clear()
        _log_fl_event(
            {
                "role": "server",
                "type": "disconnect",
                "rc": rc,
            }
        )

    def _on_message(self, client: mqtt.Client, userdata: Any, msg: mqtt.MQTTMessage) -> None:
        try:
            payload = json.loads(msg.payload.decode("utf-8"))
        except (UnicodeDecodeError, json.JSONDecodeError):
            print("[FLServer] Received malformed MQTT message.")
            return

        msg_type = payload.get("type")
        if msg_type == "register":
            client_id = payload.get("client_id")
            if not client_id:
                print("[FLServer] Registration missing client_id.")
                return
            self._event_queue.put({"type": "register", "client_id": client_id})
        elif msg_type == "update":
            encoded_state = payload.get("state")
            client_id = payload.get("client_id")
            if not encoded_state or not client_id:
                print("[FLServer] Update missing state or client_id.")
                return
            try:
                weights = self._decode_weights(encoded_state)
            except (pickle.UnpicklingError, ValueError) as exc:
                print(f"[FLServer] Failed to decode update state: {exc}")
                return
            round_idx = int(payload.get("round", self.global_round))
            val_loss = payload.get("val_loss")
            latency_ms = payload.get("latency_ms")
            self._event_queue.put(
                {
                    "type": "update",
                    "client_id": client_id,
                    "round": round_idx,
                    "state": weights,
                    "val_loss": val_loss,
                    "latency_ms": latency_ms,
                }
            )

    # ------------------------------------------------------------------- public
    def serve_forever(self) -> None:
        self._mqtt.connect(self.broker_host, self.broker_port, self.keepalive)
        self._mqtt.loop_start()
        try:
            self._connected.wait()
            while True:
                event = self._event_queue.get()
                etype = event.get("type")
                if etype == "register":
                    self._handle_registration(event["client_id"])
                elif etype == "update":
                    self._handle_update(
                        client_id=event["client_id"],
                        round_idx=event["round"],
                        weights=event["state"],
                        val_loss=event.get("val_loss"),
                        latency_ms=event.get("latency_ms"),
                    )
        except KeyboardInterrupt:
            print("[FLServer] Interrupted, shutting down.")
        finally:
            self._mqtt.loop_stop()
            self._mqtt.disconnect()

    # ------------------------------------------------------------- event logic
    def _handle_registration(self, client_id: str) -> None:
        self.registered_clients.add(client_id)
        print(f"[FLServer] Registered client {client_id}")
        _log_fl_event(
            {
                "role": "server",
                "type": "register",
                "client_id": client_id,
            }
        )
        self._send_state(client_id, "welcome")

    def _handle_update(
        self,
        client_id: str,
        round_idx: int,
        weights: Weights,
        val_loss: float | None,
        latency_ms: float | None,
    ) -> None:
        self.pending_updates[round_idx].append(weights)
        self.pending_clients[round_idx].append(client_id)
        if val_loss is not None:
            self.pending_losses[round_idx].append(float(val_loss))
        self._send_ack(client_id, round_idx)
        count = len(self.pending_updates[round_idx])
        print(
            f"[FLServer] Received update from {client_id} "
            f"(round {round_idx}, {count}/{self.clients_per_round})"
        )

        _log_fl_event(
            {
                "role": "server",
                "type": "update_received",
                "client_id": client_id,
                "round": round_idx,
                "val_loss": val_loss,
                "latency_ms": latency_ms,
                "count": count,
                "clients_per_round": self.clients_per_round,
            }
        )

        if count >= self.clients_per_round:
            contributors = self.pending_clients.pop(round_idx)
            updates = self.pending_updates.pop(round_idx)
            losses = self.pending_losses.pop(round_idx, [])
            avg_loss = float(np.mean(losses)) if losses else None

            aggregated = self._aggregate(updates)
            self.model.set_weights(aggregated)
            ckpt_path = self.ckpt_dir / f"round_{round_idx + 1}.keras"
            self.model.save(ckpt_path)
            self.global_round = round_idx + 1

            _log_fl_event(
                {
                    "role": "server",
                    "type": "round_agg",
                    "round": round_idx,
                    "avg_loss": avg_loss,
                    "contributors": contributors,
                }
            )
            _log_fl_event(
                {
                    "role": "server",
                    "type": "round_end",
                    "round": round_idx,
                    "avg_loss": avg_loss,
                }
            )

            self._broadcast_state(contributors, "global_model")

    def _aggregate(self, updates: List[Weights]) -> Weights:
        if not updates:
            return self.model.get_weights()
        aggregated: Weights = []
        for tensors in zip(*updates):
            stacked = np.stack(tensors, axis=0)
            aggregated.append(np.mean(stacked, axis=0))
        return aggregated


__all__ = ["MQTTFLServer"]
