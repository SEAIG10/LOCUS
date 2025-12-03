from __future__ import annotations

import re
from pathlib import Path
from typing import List

import numpy as np

from packages.federated.config import GLOBAL_CKPT_DIR

_round_pattern = re.compile(r"round_(\d+)\.npy$")


def _extract_round_idx(path: Path) -> int:
    m = _round_pattern.search(path.name)
    if not m:
        return -1
    return int(m.group(1))


def load_latest_global_weights(ckpt_dir: Path | None = None) -> List[np.ndarray]:
    """
    results/fl_global/round_*.npy 중 가장 최신 라운드의 weight 리스트를 로드합니다.

    반환:
        weights: List[np.ndarray]  # Flower 서버가 저장한 순서 그대로
    """
    ckpt_dir = ckpt_dir or GLOBAL_CKPT_DIR
    ckpt_dir = Path(ckpt_dir)

    if not ckpt_dir.exists():
        raise FileNotFoundError(f"Checkpoint dir not found: {ckpt_dir}")

    candidates = [p for p in ckpt_dir.glob("round_*.npy") if p.is_file()]
    if not candidates:
        raise FileNotFoundError(f"No checkpoints found in {ckpt_dir}")

    latest = max(candidates, key=_extract_round_idx)
    arr = np.load(latest, allow_pickle=True)

    if isinstance(arr, np.ndarray):
        weights = [np.asarray(w) for w in arr.tolist()]
    else:
        weights = [np.asarray(arr)]

    print(f"[checkpoint_utils] Loaded global weights from {latest}")
    return weights
