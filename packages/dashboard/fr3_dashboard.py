"""FR3 Dashboard - Sequential GRU Prediction"""

from __future__ import annotations

from core import load_sample, print_header, print_table


def render_fr3():
    data = load_sample()["fr3"]
    print_header("FR3 · Sequential GRU Prediction")
    print("Context Timeline")
    print_table(data["timeline"], ["time", "summary"])
    print("\nAttention Weights")
    print_table(data["attention"], ["label", "weight"])
    print("\nPrediction Result")
    print_table(data["prediction"], ["zone", "value"])


if __name__ == "__main__":
    render_fr3()
