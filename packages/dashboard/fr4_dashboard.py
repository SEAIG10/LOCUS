"""FR4 Dashboard - Personalized Federated Learning"""

from __future__ import annotations

from core import load_sample, print_header, print_table


def render_fr4():
    data = load_sample()["fr4"]
    summary = data["summary"]
    print_header("FR4 · Personalized Federated Learning")
    for key, value in summary.items():
        print(f"{key}: {value}")
    print("\nClient Nodes")
    print_table(data["clients"], ["id", "name", "status", "latency", "loss"])
    print("\nFlow Log")
    for line in data["flow"]:
        print(f" - {line}")
    print("\nRecent Events")
    print_table(data["events"], ["time", "source", "event"])


if __name__ == "__main__":
    render_fr4()
