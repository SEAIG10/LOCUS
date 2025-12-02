"""Flower-based FedPer client for the LOCUS GRU predictor."""

from __future__ import annotations

import time
from pathlib import Path
from typing import Any, Dict, Tuple

import flwr as fl
import numpy as np
import tensorflow as tf

from config import (
    CLIENT_ID,
    CONTEXT_DIM,
    LOCAL_BATCH_SIZE,
    LOCAL_CKPT_DIR,
    LOCAL_EPOCHS,
    LR,
    PRETRAINED_MODEL_PATH,
    TRAIN_DATASET_PATH,
)
from fl_utils import log_fl_event, resolve_dataset_path


class LocusFlowerClient(fl.client.NumPyClient):
    """Federated Flower client that keeps the personalized head on device."""

    def __init__(
        self,
        client_id: str = CLIENT_ID,
        model_path: str | Path = PRETRAINED_MODEL_PATH,
        dataset_path: str | Path = TRAIN_DATASET_PATH,
        local_epochs: int = LOCAL_EPOCHS,
        batch_size: int = LOCAL_BATCH_SIZE,
        learning_rate: float = LR,
    ) -> None:
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate
        self.last_round = 0

        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found at {self.model_path}. "
                "Please make sure packages/ai/models/gru/gru_model.keras exists."
            )

        self.model = tf.keras.models.load_model(self.model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        dataset_abs_path = resolve_dataset_path(str(dataset_path))
        (
            self.train_dataset,
            self.val_dataset,
            self.train_sample_count,
            self.val_sample_count,
        ) = self._load_dataset(dataset_abs_path)

        self.ckpt_dir = Path(LOCAL_CKPT_DIR)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)
        self._current_round_started_at: float | None = None

        log_fl_event(
            "client",
            "client_init",
            client_id=self.client_id,
            train_samples=self.train_sample_count,
            val_samples=self.val_sample_count,
        )

    # ------------------------------------------------------------------ dataset
    def _load_dataset(
        self, dataset_path: Path
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
        if not dataset_path.exists():
            raise FileNotFoundError(
                f"Training dataset not found at {dataset_path}. "
                "Run packages/ai/training/prepare_data.py first."
            )

        with np.load(dataset_path, allow_pickle=False) as data:
            required_keys = {"X_train", "y_train", "X_val", "y_val"}
            missing = required_keys.difference(data.files)
            if missing:
                raise ValueError(
                    f"Dataset {dataset_path} is missing keys: {sorted(missing)}"
                )

            X_train = data["X_train"].astype(np.float32)
            y_train = data["y_train"].astype(np.float32)
            X_val = data["X_val"].astype(np.float32)
            y_val = data["y_val"].astype(np.float32)

        if X_train.shape[-1] != CONTEXT_DIM or X_val.shape[-1] != CONTEXT_DIM:
            raise ValueError(
                "Dataset context dimension mismatch: "
                f"found {X_train.shape[-1]}, expected {CONTEXT_DIM}"
            )

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(max(len(X_train), 1), reshuffle_each_iteration=True)
            .batch(self.batch_size)
        )
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
            self.batch_size
        )

        log_fl_event(
            "client",
            "dataset_ready",
            client_id=self.client_id,
            path=str(dataset_path),
            train_samples=len(X_train),
            val_samples=len(X_val),
        )

        return train_dataset, val_dataset, len(X_train), len(X_val)

    # ---------------------------------------------------------------- interface
    def get_parameters(self, config: Dict[str, Any] | None = None):
        return self.model.get_weights()

    def fit(
        self,
        parameters,
        config: Dict[str, Any] | None = None,
    ):
        self.model.set_weights(parameters)
        round_idx = int((config or {}).get("server_round", 0))
        self.last_round = round_idx

        self._current_round_started_at = time.time()
        log_fl_event(
            "client",
            "round_start",
            client_id=self.client_id,
            round=round_idx,
            epochs=self.local_epochs,
        )

        history = self.model.fit(
            self.train_dataset,
            epochs=self.local_epochs,
            verbose=0,
        )

        train_loss = float(history.history.get("loss", [0.0])[-1])
        val_loss, val_acc = self._evaluate_and_log(round_idx, stage="fit")

        elapsed_ms = None
        if self._current_round_started_at is not None:
            elapsed_ms = int((time.time() - self._current_round_started_at) * 1000)

        self._save_checkpoint(round_idx)

        log_fl_event(
            "client",
            "local_done",
            client_id=self.client_id,
            round=round_idx,
            train_loss=train_loss,
            val_loss=val_loss,
            val_acc=val_acc,
            latency_ms=elapsed_ms,
        )

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }
        return self.model.get_weights(), self.train_sample_count, metrics

    def evaluate(self, parameters, config: Dict[str, Any] | None = None):
        self.model.set_weights(parameters)
        round_idx = int((config or {}).get("server_round", 0))
        val_loss, val_acc = self._evaluate_and_log(round_idx, stage="evaluate")

        return float(val_loss), self.val_sample_count, {"val_acc": val_acc}

    # ----------------------------------------------------------------- helpers
    def _evaluate_and_log(self, round_idx: int, stage: str) -> Tuple[float, float]:
        metrics = self.model.evaluate(self.val_dataset, verbose=0, return_dict=True)
        val_loss = float(metrics.get("loss", 0.0))
        val_acc = float(
            metrics.get("accuracy")
            or metrics.get("binary_accuracy")
            or metrics.get("acc", 0.0)
        )

        log_fl_event(
            "client",
            "evaluate",
            client_id=self.client_id,
            round=round_idx,
            stage=stage,
            val_loss=val_loss,
            val_acc=val_acc,
        )

        return val_loss, val_acc

    def _save_checkpoint(self, round_idx: int) -> None:
        ckpt_path = self.ckpt_dir / f"{self.client_id}_round_{round_idx}.keras"
        self.model.save(ckpt_path)


__all__ = ["LocusFlowerClient"]
