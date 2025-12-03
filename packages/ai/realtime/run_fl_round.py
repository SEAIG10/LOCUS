# LOCUS/packages/ai/realtime/run_fl_round.py
from __future__ import annotations

from pathlib import Path

from packages.federated.config import SERVER_ADDRESS, TRAIN_DATASET_PATH
from packages.ai.realtime.fl_client import run_fl_client
from packages.ai.realtime.dataset_builder import build_fl_dataset
from packages.ai.realtime.update_gru_from_global import main as update_gru_from_global


def build_local_fl_dataset() -> None:
    output_path = Path(TRAIN_DATASET_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)
    build_fl_dataset(output_path=output_path)


def main() -> None:
    # 1) 로컬 로그 → npz 데이터셋 생성 (현재는 더미)
    build_local_fl_dataset()

    # 2) Flower 서버에 접속해 1라운드 학습 참가
    print(f"[LOCUS FL Client] Connecting to Flower server at {SERVER_ADDRESS} ...")
    run_fl_client(SERVER_ADDRESS)

    # 3) 서버 쪽 글로벌 체크포인트에서 최신 weights 가져와 로컬 GRU 업데이트
    print("[LOCUS FL Client] Updating local GRU with latest global checkpoint...")
    update_gru_from_global()


if __name__ == "__main__":
    main()
