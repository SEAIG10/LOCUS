# LOCUS/packages/ai/realtime/run_fl_round.py

from __future__ import annotations

from pathlib import Path

from packages.federated.config import (
    SERVER_ADDRESS,
    TRAIN_DATASET_PATH,
)
from packages.ai.realtime.fl_client import run_fl_client


def build_local_fl_dataset() -> None:
    """
    로컬 로그를 읽어 FL용 training_dataset.npz로 변환하는 함수.

    지금은 예시라서 'TODO'로 두고, 실제 구현은
    - 최근 N시간 / 하루치 로그 불러오기
    - (sequence_length, context_dim) 형태로 전처리
    - X_train, y_train, X_val, y_val로 split
    - np.savez(TRAIN_DATASET_PATH, ...)
    이런 로직을 네가 갖고 있는 모듈로 교체하면 됨.
    """
    from packages.ai.realtime.dataset_builder import build_fl_dataset  # 예시 경로

    output_path = Path(TRAIN_DATASET_PATH)
    output_path.parent.mkdir(parents=True, exist_ok=True)

    build_fl_dataset(output_path=output_path)


def main() -> None:
    # 1) 로컬 로그 → npz 데이터셋 생성
    build_local_fl_dataset()

    # 2) Flower 서버에 접속해 1라운드 학습 참가
    #    (서버는 이미 docker/container/systemd로 떠 있다고 가정)
    print(f"[LOCUS FL Client] Connecting to Flower server at {SERVER_ADDRESS} ...")
    run_fl_client(SERVER_ADDRESS)


if __name__ == "__main__":
    main()
