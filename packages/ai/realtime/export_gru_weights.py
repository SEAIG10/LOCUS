"""
Export Keras GRU model weights into a NumPy .npy file
that will be used by the federated learning SERVER (NumPy-only).

The server will NOT load the .keras model, but only use the weights as ndarrays.
"""

from pathlib import Path

import numpy as np
import tensorflow as tf

# Input Keras model
MODEL_PATH = Path("packages/ai/models/gru/gru_model.keras")

# Output NumPy weight file
OUTPUT_PATH = Path("packages/ai/models/gru/gru_initial_weights.npy")


def main():
    print(f"[INFO] Loading Keras model from: {MODEL_PATH}")

    if not MODEL_PATH.exists():
        raise FileNotFoundError(f"Keras model not found: {MODEL_PATH}")

    # Load the model
    model = tf.keras.models.load_model(MODEL_PATH)

    # Extract weights: this is a Python list of ndarrays
    weights = model.get_weights()

    print(f"[INFO] Number of weight tensors: {len(weights)}")
    for i, w in enumerate(weights):
        print(f"  - Weight[{i}] shape = {w.shape}")

    # Wrap as dtype=object so heterogeneous shapes are allowed
    weights_array = np.array(weights, dtype=object)

    OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
    np.save(OUTPUT_PATH, weights_array, allow_pickle=True)

    print(f"[SUCCESS] Weights exported to: {OUTPUT_PATH}")
    print(
        "[INFO] On the server, load with np.load(..., allow_pickle=True) "
        "and call .tolist() to get back the list of ndarrays."
    )


if __name__ == "__main__":
    main()
