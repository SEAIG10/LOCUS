"""FR2 Dashboard - Visual & Audio Context"""

from __future__ import annotations

from core import load_sample, print_header, print_table


def render_fr2():
    data = load_sample()["fr2"]
    print_header("FR2 · Visual & Audio Context Awareness")
    print("YOLO Detections")
    print_table(data["yolo"], ["label", "confidence"])
    print("\nYAMNet Top Classes")
    print_table(data["yamnet"], ["label", "score"])
    print("\nTimeSync Timeline")
    print_table(data["sync"], ["step", "latency"])


if __name__ == "__main__":
    render_fr2()
