"""(Legacy) Flower ServerApp used by the SuperNode/SuperLink stack.

현재 LOCUS는 Flower 1.x의 클래식 gRPC 서버/클라이언트 모드를 사용합니다.
SuperNode 기반 배포를 다시 시도할 때 참고용으로만 남겨두었습니다.
"""

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
