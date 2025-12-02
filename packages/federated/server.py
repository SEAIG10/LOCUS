"""Flower-based aggregation server that tracks dashboard events."""

from __future__ import annotations

from pathlib import Path
from typing import Dict, List, Sequence, Tuple

import flwr as fl
import numpy as np
import tensorflow as tf
from flwr.common import FitIns, Parameters, ndarrays_to_parameters, parameters_to_ndarrays
from flwr.server.client_manager import ClientManager
from flwr.server.client_proxy import ClientProxy

from config import CLIENTS_PER_ROUND, GLOBAL_CKPT_DIR, PRETRAINED_MODEL_PATH
from fl_utils import log_fl_event


class LocusFedAvg(fl.server.strategy.FedAvg):
    """Custom FedAvg strategy that emits FL events and saves checkpoints."""

    def __init__(
        self,
        model_path: str | Path = PRETRAINED_MODEL_PATH,
        clients_per_round: int = CLIENTS_PER_ROUND,
    ) -> None:
        self.model_path = Path(model_path).expanduser().resolve()
        if not self.model_path.exists():
            raise FileNotFoundError(
                f"Pretrained model not found at {self.model_path}. "
                "Please make sure packages/ai/models/gru/gru_model.keras exists."
            )

        self.template_model = tf.keras.models.load_model(self.model_path)
        initial_parameters = ndarrays_to_parameters(self.template_model.get_weights())
        self.ckpt_dir = Path(GLOBAL_CKPT_DIR)
        self.ckpt_dir.mkdir(parents=True, exist_ok=True)

        super().__init__(
            fraction_fit=1.0,
            fraction_evaluate=1.0,
            min_fit_clients=clients_per_round,
            min_available_clients=clients_per_round,
            min_evaluate_clients=clients_per_round,
            initial_parameters=initial_parameters,
        )

    # ------------------------------------------------------------- config hooks
    def configure_fit(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        instructions = super().configure_fit(server_round, parameters, client_manager)
        instructions = self._attach_round_config(instructions, server_round)

        targets = [client.cid for client, _ in instructions]
        log_fl_event(
            "server",
            "round_start",
            round=server_round,
            targets=targets,
        )
        return instructions

    def configure_evaluate(
        self,
        server_round: int,
        parameters: Parameters,
        client_manager: ClientManager,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        instructions = super().configure_evaluate(server_round, parameters, client_manager)
        return self._attach_round_config(instructions, server_round)

    def _attach_round_config(
        self,
        instructions: List[Tuple[ClientProxy, FitIns]],
        server_round: int,
    ) -> List[Tuple[ClientProxy, FitIns]]:
        updated: List[Tuple[ClientProxy, FitIns]] = []
        for client, fit_ins in instructions:
            fit_ins.config = {**(fit_ins.config or {}), "server_round": server_round}
            updated.append((client, fit_ins))
        return updated

    # ------------------------------------------------------------ aggregation
    def aggregate_fit(
        self,
        server_round: int,
        results: List[Tuple[ClientProxy, fl.common.FitRes]],
        failures: List[BaseException],
    ):
        for client, fit_res in results:
            metrics = fit_res.metrics or {}
            log_fl_event(
                "server",
                "update_received",
                client_id=client.cid,
                round=server_round,
                train_loss=metrics.get("train_loss"),
                val_loss=metrics.get("val_loss"),
                val_acc=metrics.get("val_acc"),
                num_examples=fit_res.num_examples,
            )

        aggregated = super().aggregate_fit(server_round, results, failures)
        parameters, metrics = aggregated

        if parameters is not None:
            weights = parameters_to_ndarrays(parameters)
            self.template_model.set_weights(weights)
            ckpt_path = self.ckpt_dir / f"round_{server_round}.keras"
            self.template_model.save(ckpt_path)

            avg_loss = self._average_metric(results, "val_loss")
            contributors = [client.cid for client, _ in results]

            log_fl_event(
                "server",
                "round_agg",
                round=server_round,
                avg_loss=avg_loss,
                contributors=contributors,
            )
            log_fl_event(
                "server",
                "round_end",
                round=server_round,
                avg_loss=avg_loss,
            )

        return parameters, metrics

    # ----------------------------------------------------------- util helpers
    @staticmethod
    def _average_metric(
        results: Sequence[Tuple[ClientProxy, fl.common.FitRes]],
        key: str,
    ) -> float | None:
        values = []
        for _, fit_res in results:
            metrics = fit_res.metrics or {}
            if key in metrics and metrics[key] is not None:
                values.append(float(metrics[key]))
        if not values:
            return None
        return float(np.mean(values))


__all__ = ["LocusFedAvg"]
