# packages/federated/run_fl_server_compat.py
from __future__ import annotations

import flwr as fl

from .server import LocusFedAvg
from .config import SERVER_ROUNDS


def main() -> None:
    """Legacy-style Flower server, no SuperLink/SuperNode."""
    strategy = LocusFedAvg()

    fl.server.start_server(
        server_address="0.0.0.0:8080",
        config=fl.server.ServerConfig(num_rounds=SERVER_ROUNDS),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
