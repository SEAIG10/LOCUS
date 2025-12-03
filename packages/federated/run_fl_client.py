"""Entry point for the Flower-based FedPer client."""

from __future__ import annotations

import argparse

import flwr as fl

from client import LocusFlowerClient
from .config import (
    CLIENT_ID,
    FLOWER_SERVER_ADDRESS,
    LOCAL_BATCH_SIZE,
    LOCAL_EPOCHS,
    LR,
    PRETRAINED_MODEL_PATH,
    TRAIN_DATASET_PATH,
)


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(description="LOCUS Flower FedPer client")
    parser.add_argument(
        "--server-address",
        default=FLOWER_SERVER_ADDRESS,
        help="Flower server address in host:port format",
    )
    parser.add_argument("--client-id", default=CLIENT_ID, help="Unique client id")
    parser.add_argument(
        "--model-path",
        default=str(PRETRAINED_MODEL_PATH),
        help="Path to the pretrained Keras GRU model",
    )
    parser.add_argument(
        "--dataset-path",
        default=str(TRAIN_DATASET_PATH),
        help="Path to the federated training dataset (.npz)",
    )
    parser.add_argument(
        "--local-epochs",
        default=LOCAL_EPOCHS,
        type=int,
        help="Number of local epochs per round",
    )
    parser.add_argument(
        "--batch-size",
        default=LOCAL_BATCH_SIZE,
        type=int,
        help="Local batch size",
    )
    parser.add_argument(
        "--learning-rate",
        default=LR,
        type=float,
        help="Local learning rate",
    )
    return parser.parse_args()


def main() -> None:
    args = parse_args()
    client = LocusFlowerClient(
        client_id=args.client_id,
        model_path=args.model_path,
        dataset_path=args.dataset_path,
        local_epochs=args.local_epochs,
        batch_size=args.batch_size,
        learning_rate=args.learning_rate,
    )
    fl.client.start_numpy_client(server_address=args.server_address, client=client)


if __name__ == "__main__":
    main()
