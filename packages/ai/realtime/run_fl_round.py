# LOCUS/packages/ai/realtime/run_fl_round.py
from __future__ import annotations

from pathlib import Path
import sys   # ← 바로 이 줄이 없어서 터진 것

from packages.federated.config import SERVER_ADDRESS, TRAIN_DATASET_PATH
from packages.ai.realtime.fl_client import run_fl_client
from packages.ai.realtime.dataset_builder import build_fl_dataset
from packages.ai.realtime.update_gru_from_global import main as update_gru_from_global


def build_local_fl_dataset() -> None:
    output_path = Path(TRAIN_DATASET_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_fl_dataset(output_path=output_path)


def main(server_address: str | None = None) -> None:
    # 1) 로컬 로그 → npz 데이터셋 생성
    build_local_fl_dataset()

    # 2) FL 서버 접속
    addr = server_address or SERVER_ADDRESS
    print(f"[LOCUS FL Client] Connecting to Flower server at {addr} ...")
    run_fl_client(addr)

    # 3) FL 글로벌 베이스 업데이트
    print("[LOCUS FL Client] Updating local GRU with latest global checkpoint...")
    update_gru_from_global()


if __name__ == "__main__":
    cli_addr = sys.argv[1] if len(sys.argv) > 1 else None
    main(server_address=cli_addr)
