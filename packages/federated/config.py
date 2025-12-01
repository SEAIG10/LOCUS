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

SERVER_BIND = "tcp://0.0.0.0:5555"
CLIENT_ID = "home_001"

MQTT_BROKER_HOST = "13.124.210.11"
MQTT_BROKER_PORT = 1883
MQTT_KEEPALIVE = 30
MQTT_TOPIC_NAMESPACE = "locus/fl"
PRETRAINED_MODEL_PATH = AI_DEFAULTS["gru_model"]

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
TRAIN_DATASET_PATH = "data/training_dataset.npz"
GLOBAL_CKPT_DIR = "results/fl_global"
LOCAL_CKPT_DIR = "results/fl_local"
