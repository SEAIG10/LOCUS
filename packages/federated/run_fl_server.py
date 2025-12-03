"""Classic Flower FedAvg server entrypoint."""

from __future__ import annotations

from flwr.server import ServerConfig, start_server

from .config import CLIENTS_PER_ROUND, SERVER_ADDRESS, SERVER_INITIAL_WEIGHTS_PATH, SERVER_ROUNDS
from .server import LocusFedAvg


def main() -> None:
    """Start the Flower gRPC server on 0.0.0.0:8080."""

    strategy = LocusFedAvg(
        weights_path=SERVER_INITIAL_WEIGHTS_PATH,
        clients_per_round=CLIENTS_PER_ROUND,
    )
    start_server(
        server_address=SERVER_ADDRESS,
        config=ServerConfig(num_rounds=SERVER_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
