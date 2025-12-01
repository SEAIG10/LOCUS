from __future__ import annotations

from typing import Dict, Iterable, List


def print_header(title: str) -> None:
    bar = "=" * len(title)
    print(f"\n{bar}\n{title}\n{bar}")


def print_table(rows: List[Dict[str, object]], columns: Iterable[str]) -> None:
    rows = rows or []
    columns = list(columns)
    if not columns:
        return
    col_widths = []
    for col in columns:
        width = max(len(col), max((len(str(row.get(col, ""))) for row in rows), default=0))
        col_widths.append(width)
    header = " | ".join(col.ljust(width) for col, width in zip(columns, col_widths))
    print(header)
    print("-" * len(header))
    for row in rows:
        line = " | ".join(str(row.get(col, "")).ljust(width) for col, width in zip(columns, col_widths))
        print(line)


def print_list(items: Iterable[str]) -> None:
    for item in items:
        print(f" - {item}")
