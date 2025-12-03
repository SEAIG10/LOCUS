# LOCUS/packages/federated/config.py
"""Global configuration shared across modules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
# /packages/federated/config.py → BASE_DIR.parents[2] == 프로젝트 루트
ROOT_DIR = BASE_DIR.parents[2]

# 루트(또는 패키지 루트)를 sys.path에 추가
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from packages.config.settings import AI_DEFAULTS, FEDERATED_DEFAULTS  # noqa: E402

# --------------------------------------------------------------------------- 기본 설정
CLIENT_ID = "home_001"

# Classic Flower server/client endpoints
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"

# --------------------------------------------------------------------------- 모델/가중치 경로
GRU_MODEL_PATH: Path = AI_DEFAULTS["gru_model"]
SERVER_INITIAL_WEIGHTS_PATH: Path = GRU_MODEL_PATH.with_name("gru_initial_weights.npy")

# FedPer 온디바이스 모델 경로 (없으면 gru_model 옆에 fedper_gru_model.keras로 떨어지게)
FEDPER_MODEL_PATH: Path = AI_DEFAULTS.get(
    "fedper_model",
    GRU_MODEL_PATH.with_name("fedper_gru_model.keras"),
)

# Backwards compatibility aliases (legacy docs/scripts)
PRETRAINED_MODEL_PATH: Path = GRU_MODEL_PATH
INITIAL_WEIGHTS_PATH: Path = SERVER_INITIAL_WEIGHTS_PATH

# --------------------------------------------------------------------------- ZMQ 엔드포인트
ZMQ_ENDPOINTS: Dict[str, str] = {
    "location": "ipc:///tmp/locus.location",
    "visual": "ipc:///tmp/locus.visual",
    "audio": "ipc:///tmp/locus.audio",
    "context": "ipc:///tmp/locus.context",
    "telemetry": "ipc:///tmp/locus.telemetry",
}

# --------------------------------------------------------------------------- GRU/컨텍스트 설정
TIMESYNC_WINDOW_MS = 100
SEQUENCE_LENGTH = 30
CONTEXT_DIM = 160
CONTEXT_MODE = "attention"  # attention encoder produces 160-d vectors

ZONE_NAMES: List[str] = [
    "bathroom",
    "bedroom_1",
    "bedroom_2",
    "corridor",
    "garden_balcony",
    "kitchen",
    "living_room",
]

GRU_INPUT_DIM = CONTEXT_DIM
GRU_HIDDEN_DIM = 64
GRU_NUM_ZONES = len(ZONE_NAMES)

# --------------------------------------------------------------------------- Federated Learning 설정
CLIENTS_PER_ROUND = FEDERATED_DEFAULTS["clients_per_round"]
LOCAL_EPOCHS = FEDERATED_DEFAULTS["local_epochs"]
LR = FEDERATED_DEFAULTS["learning_rate"]
LOCAL_BATCH_SIZE = FEDERATED_DEFAULTS["local_batch_size"]
SERVER_ROUNDS = FEDERATED_DEFAULTS["server_rounds"]

# 데이터셋/체크포인트 경로를 프로젝트 루트 기준으로 지정
TRAIN_DATASET_PATH: Path = ROOT_DIR / "packages" / "ai" / "data" / "training_dataset.npz"
GLOBAL_CKPT_DIR: Path = ROOT_DIR / "results" / "fl_global"
