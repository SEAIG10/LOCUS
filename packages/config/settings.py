from __future__ import annotations

from pathlib import Path

ROOT_DIR = Path(__file__).resolve().parents[2]
PACKAGES_DIR = ROOT_DIR / "packages"
AI_MODELS_DIR = PACKAGES_DIR / "ai" / "models"

AI_DEFAULTS = {
    "gru_model": AI_MODELS_DIR / "gru" / "gru_model.keras",
}

FEDERATED_DEFAULTS = {
    "clients_per_round": 1,
    "local_epochs": 3,
    "learning_rate": 1e-3,
    "local_batch_size": 32,
    "server_rounds": 3,
}
