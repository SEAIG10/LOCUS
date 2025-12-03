"""Flower ClientApp for LOCUS edge devices (SuperNode architecture)."""

from __future__ import annotations

from pathlib import Path
from typing import Tuple

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.client import ClientApp
from flwr.common import (
    Code,
    EvaluateIns,
    EvaluateRes,
    FitIns,
    FitRes,
    GetParametersIns,
    GetParametersRes,
    Status,
    ndarrays_to_parameters,
    parameters_to_ndarrays,
)

from packages.federated.config import (
    CLIENT_ID,
    LOCAL_BATCH_SIZE,
    LOCAL_EPOCHS,
    LR,
    PRETRAINED_MODEL_PATH,
    TRAIN_DATASET_PATH,
)
from packages.federated.fl_utils import log_fl_event, resolve_dataset_path


class LocusClient(fl.client.Client):
    """TensorFlow-based Flower client compatible with Flower 1.24 ClientApps."""

    def __init__(
        self,
        model_path: str | Path = PRETRAINED_MODEL_PATH,
        dataset_path: str | Path = TRAIN_DATASET_PATH,
        local_epochs: int = LOCAL_EPOCHS,
        batch_size: int = LOCAL_BATCH_SIZE,
        learning_rate: float = LR,
        client_id: str = CLIENT_ID,
    ) -> None:
        self.client_id = client_id
        self.local_epochs = local_epochs
        self.batch_size = batch_size
        self.learning_rate = learning_rate

        self.model_path = Path(model_path).expanduser().resolve()
        self.dataset_path = resolve_dataset_path(str(dataset_path))

        if not self.model_path.exists():
            raise FileNotFoundError(f"Model not found at {self.model_path}")
        if not self.dataset_path.exists():
            raise FileNotFoundError(f"Dataset not found at {self.dataset_path}")

        self.model = tf.keras.models.load_model(self.model_path)
        self.model.compile(
            optimizer=tf.keras.optimizers.Adam(learning_rate=self.learning_rate),
            loss="binary_crossentropy",
            metrics=["accuracy"],
        )

        (
            self.train_dataset,
            self.val_dataset,
            self.train_examples,
            self.val_examples,
        ) = self._prepare_datasets(self.dataset_path)

        log_fl_event(
            "client",
            "client_init",
            client_id=self.client_id,
            train_samples=self.train_examples,
            val_samples=self.val_examples,
        )

    # --------------------------------------------------------------- client API
    def get_parameters(self, ins: GetParametersIns) -> GetParametersRes:  # noqa: D401
        parameters = ndarrays_to_parameters(self.model.get_weights())
        return GetParametersRes(
            status=Status(code=Code.OK, message="parameters_ready"),
            parameters=parameters,
        )

    def fit(self, ins: FitIns) -> FitRes:  # noqa: D401
        weights = parameters_to_ndarrays(ins.parameters)
        self.model.set_weights(weights)
        server_round = int(ins.config.get("server_round", 0))

        log_fl_event(
            "client",
            "round_start",
            client_id=self.client_id,
            round=server_round,
            epochs=self.local_epochs,
        )

        history = self.model.fit(
            self.train_dataset,
            epochs=self.local_epochs,
            verbose=0,
        )
        train_loss = float(history.history.get("loss", [0.0])[-1])
        val_loss, val_acc = self._evaluate(stage="fit", round_idx=server_round)

        metrics = {
            "train_loss": train_loss,
            "val_loss": val_loss,
            "val_acc": val_acc,
        }

        log_fl_event(
            "client",
            "local_done",
            client_id=self.client_id,
            round=server_round,
            **metrics,
        )

        return FitRes(
            status=Status(code=Code.OK, message="fit_success"),
            parameters=ndarrays_to_parameters(self.model.get_weights()),
            num_examples=self.train_examples,
            metrics=metrics,
        )

    def evaluate(self, ins: EvaluateIns) -> EvaluateRes:  # noqa: D401
        weights = parameters_to_ndarrays(ins.parameters)
        self.model.set_weights(weights)
        server_round = int(ins.config.get("server_round", 0))

        val_loss, val_acc = self._evaluate(stage="evaluate", round_idx=server_round)
        metrics = {"val_acc": val_acc}

        return EvaluateRes(
            status=Status(code=Code.OK, message="evaluate_success"),
            loss=float(val_loss),
            num_examples=self.val_examples,
            metrics=metrics,
        )

    # ------------------------------------------------------------ local helpers
    def _prepare_datasets(
        self,
        dataset_path: Path,
    ) -> Tuple[tf.data.Dataset, tf.data.Dataset, int, int]:
        with np.load(dataset_path, allow_pickle=False) as data:
            required = {"X_train", "y_train", "X_val", "y_val"}
            missing = required.difference(data.files)
            if missing:
                raise ValueError(
                    f"Dataset {dataset_path} is missing keys: {sorted(missing)}"
                )
            X_train = data["X_train"].astype(np.float32)
            y_train = data["y_train"].astype(np.float32)
            X_val = data["X_val"].astype(np.float32)
            y_val = data["y_val"].astype(np.float32)

        train_dataset = (
            tf.data.Dataset.from_tensor_slices((X_train, y_train))
            .shuffle(max(len(X_train), 1), reshuffle_each_iteration=True)
            .batch(self.batch_size)
        )
        val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(
            self.batch_size
        )
        return train_dataset, val_dataset, len(X_train), len(X_val)

    def _evaluate(self, stage: str, round_idx: int) -> Tuple[float, float]:
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


def client_app() -> ClientApp:
    """Entry-point referenced by `flower-client-app --app=...`."""

    model_path = PRETRAINED_MODEL_PATH
    dataset_path = resolve_dataset_path(TRAIN_DATASET_PATH)

    def client_fn(context: fl.common.Context) -> fl.client.Client:  # noqa: ANN001
        return LocusClient(
            model_path=model_path,
            dataset_path=dataset_path,
            local_epochs=LOCAL_EPOCHS,
            batch_size=LOCAL_BATCH_SIZE,
            learning_rate=LR,
            client_id=CLIENT_ID,
        )

    return ClientApp(client_fn=client_fn)


if __name__ == "__main__":
    import sys

    # 사용법: python -m packages.ai.realtime.fl_client <server_ip:port>
    server_address = sys.argv[1] if len(sys.argv) > 1 else "127.0.0.1:8080"

    print(f"[FL client] Connecting to Flower server at {server_address} ...")

    client = LocusClient()
    fl.client.start_client(
        server_address=server_address,
        client=client.to_client(),  # NumPyClient → 통신용 클라이언트로 변환
    )

