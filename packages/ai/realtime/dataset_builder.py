# LOCUS/packages/ai/realtime/dataset_builder.py
from __future__ import annotations

from pathlib import Path

import numpy as np

from packages.federated.config import SEQUENCE_LENGTH, CONTEXT_DIM, ZONE_NAMES


def build_fl_dataset(output_path: Path, n_train: int = 256, n_val: int = 64) -> None:
    """
    연합학습용 더미 데이터셋 생성.
    실제로는 FR 로그 → (X, y)로 전처리한 뒤 이 포맷으로 저장하면 됨.

    X: (num_samples, SEQUENCE_LENGTH, CONTEXT_DIM)
    y: (num_samples,) 또는 (num_samples, 1) 정수 클래스 (0 ~ num_zones-1)
    """

    num_zones = len(ZONE_NAMES)

    # 더미 입력 (예: 30타임스텝 × 160차원 컨텍스트)
    X_train = np.random.randn(n_train, SEQUENCE_LENGTH, CONTEXT_DIM).astype("float32")
    X_val = np.random.randn(n_val, SEQUENCE_LENGTH, CONTEXT_DIM).astype("float32")

    # 더미 라벨: 0 ~ num_zones-1
    y_train = np.random.randint(0, num_zones, size=(n_train, 1), dtype="int32")
    y_val = np.random.randint(0, num_zones, size=(n_val, 1), dtype="int32")

    output_path.parent.mkdir(parents=True, exist_ok=True)
    np.savez(
        output_path,
        X_train=X_train,
        y_train=y_train,
        X_val=X_val,
        y_val=y_val,
    )

    print(f"[dataset_builder] Saved dummy dataset to {output_path}")
