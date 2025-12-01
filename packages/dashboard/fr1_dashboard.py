"""FR1 Dashboard - Home Structure & Location Intelligence"""

from __future__ import annotations

import argparse
from typing import Literal

from core import print_header, print_table
from core.loaders import get_fr1_data  # 새로 추가할 함수

SourceType = Literal["sample", "live"]


def render_fr1(source: SourceType = "sample") -> None:
    data = get_fr1_data(source)

    print_header("FR1 · Home Structure & Location Intelligence")

    # 1) Zones
    print("Zones")
    print_table(
        data["zones"],
        ["name", "occupancy", "humidity", "lighting"],
    )

    # 2) Mobile Tracker
    tracker = data["tracker"]
    print("\nMobile Tracker")
    for key, value in tracker.items():
        print(f"  {key}: {value}")

    # 3) Recent Events
    print("\nRecent Events")
    print_table(data["events"], ["time", "label"])


def main() -> None:
    parser = argparse.ArgumentParser(description="FR1 Dashboard")
    parser.add_argument(
        "--source",
        choices=["sample", "live"],
        default="sample",
        help="데이터 소스 선택 (sample: 샘플 JSON, live: 실제 모듈 연동)",
    )
    args = parser.parse_args()
    render_fr1(source=args.source)  # type: ignore[arg-type]


if __name__ == "__main__":
    main()
