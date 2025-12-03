"""Global configuration shared across modules."""

from __future__ import annotations

import sys
from pathlib import Path
from typing import Dict, List

BASE_DIR = Path(__file__).resolve().parent
ROOT_DIR = BASE_DIR.parents[1]
if str(ROOT_DIR) not in sys.path:
    sys.path.append(str(ROOT_DIR))

from packages.config.settings import AI_DEFAULTS, FEDERATED_DEFAULTS  # noqa: E402

CLIENT_ID = "home_001"

# Classic Flower server/client endpoints
SERVER_HOST = "0.0.0.0"
SERVER_PORT = 8080
SERVER_ADDRESS = f"{SERVER_HOST}:{SERVER_PORT}"

# 모델/가중치 경로
GRU_MODEL_PATH = AI_DEFAULTS["gru_model"]
SERVER_INITIAL_WEIGHTS_PATH = GRU_MODEL_PATH.with_name("gru_initial_weights.npy")

# Backwards compatibility aliases (legacy docs/scripts)
PRETRAINED_MODEL_PATH = GRU_MODEL_PATH
INITIAL_WEIGHTS_PATH = SERVER_INITIAL_WEIGHTS_PATH

ZMQ_ENDPOINTS: Dict[str, str] = {
    "location": "ipc:///tmp/locus.location",
    "visual": "ipc:///tmp/locus.visual",
    "audio": "ipc:///tmp/locus.audio",
    "context": "ipc:///tmp/locus.context",
    "telemetry": "ipc:///tmp/locus.telemetry",
}

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

CLIENTS_PER_ROUND = FEDERATED_DEFAULTS["clients_per_round"]
LOCAL_EPOCHS = FEDERATED_DEFAULTS["local_epochs"]
LR = FEDERATED_DEFAULTS["learning_rate"]
LOCAL_BATCH_SIZE = FEDERATED_DEFAULTS["local_batch_size"]
SERVER_ROUNDS = FEDERATED_DEFAULTS["server_rounds"]
TRAIN_DATASET_PATH = "packages/ai/data/training_dataset.npz"
GLOBAL_CKPT_DIR = "results/fl_global"
