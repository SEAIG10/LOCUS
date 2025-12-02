"""Entry point for the Flower-based aggregation server."""

from __future__ import annotations

import argparse

import flwr as fl

from config import (
    CLIENTS_PER_ROUND,
    FLOWER_SERVER_ADDRESS,
    PRETRAINED_MODEL_PATH,
    SERVER_ROUNDS,
)
from server import LocusFedAvg


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOCUS Flower FedPer server")
    parser.add_argument(
        "--server-address",
        default=FLOWER_SERVER_ADDRESS,
        help="Flower server bind address in host:port format",
    )
    parser.add_argument(
        "--rounds",
        default=SERVER_ROUNDS,
        type=int,
        help="Number of FL rounds to run",
    )
    parser.add_argument(
        "--clients-per-round",
        default=CLIENTS_PER_ROUND,
        type=int,
        help="Minimum number of client updates per round",
    )
    parser.add_argument(
        "--model-path",
        default=str(PRETRAINED_MODEL_PATH),
        help="Path to the pretrained Keras GRU model",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    strategy = LocusFedAvg(
        model_path=args.model_path,
        clients_per_round=args.clients_per_round,
    )
    fl.server.start_server(
        server_address=args.server_address,
        config=fl.server.ServerConfig(num_rounds=args.rounds),
        strategy=strategy,
    )


if __name__ == "__main__":
    main()
