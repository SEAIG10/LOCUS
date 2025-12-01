"""FR5 Dashboard - Policy Engine & Cleaning Decision"""

from __future__ import annotations

from core import load_sample, print_header, print_table


def render_fr5():
    data = load_sample()["fr5"]
    print_header("FR5 · Policy Engine & Cleaning Decision")
    print(f"Action: {data['action']}")
    print(f"Reason: {data['reason']}")
    print(f"Estimated total: {data['eta']} min, Battery: {data['battery']}%")
    print("\nMap Pollution Scores")
    print_table(data["map"], ["zone", "score"])
    print("\nCleaning Path")
    print_table(data["path"], ["zone", "eta", "prob"])
    print("\nNotes")
    for note in data["notes"]:
        print(f" - {note}")


if __name__ == "__main__":
    render_fr5()
