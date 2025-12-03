"""Flower ServerApp entrypoint used by the SuperNode process."""

from __future__ import annotations

import flwr as fl
from flwr.server import ServerApp

from .config import CLIENTS_PER_ROUND, PRETRAINED_MODEL_PATH, SERVER_ROUNDS
from .server import LocusFedAvg


def main() -> ServerApp:
    """Return a ServerApp consumed by `flower-supernode --app=...`."""
    strategy = LocusFedAvg(
        model_path=PRETRAINED_MODEL_PATH,
        clients_per_round=CLIENTS_PER_ROUND
    )

    return ServerApp(
        config=fl.server.ServerConfig(num_rounds=SERVER_ROUNDS),
        strategy=strategy,
    )


__all__ = ["main"]
