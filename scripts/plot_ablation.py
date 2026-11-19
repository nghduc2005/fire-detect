"""Vẽ biểu đồ so sánh 4 cấu hình ablation từ file CSV — Thành viên 4.

Cách dùng::

    python -m scripts.plot_ablation
    python -m scripts.plot_ablation --csv outputs/ablation_results.csv \
        --out-dir docs/report
"""

from __future__ import annotations

import argparse
import csv
from pathlib import Path
from typing import Dict, List, Sequence

import matplotlib

matplotlib.use("Agg")
import matplotlib.pyplot as plt  # noqa: E402

SHORT_NAMES = ["Color", "+ Motion", "+ Temporal", "+ Persistence"]
COLORS = ["#8c8c8c", "#4c72b0", "#dd8452", "#55a868"]


def read_results(path: str) -> List[Dict[str, str]]:
    with open(path, newline="", encoding="utf-8") as fh:
        return list(csv.DictReader(fh))


def _bar_labels(ax, bars, fmt: str = "{:.1f}") -> None:
    for bar in bars:
        height = bar.get_height()
        ax.annotate(
            fmt.format(height),
            xy=(bar.get_x() + bar.get_width() / 2, height),
            xytext=(0, 3),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=9,
        )


def plot_f1_fpr(rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    labels = SHORT_NAMES[: len(rows)]
    f1 = [float(r["F1-Score (%)"]) for r in rows]
    fpr = [float(r["FPR (%)"]) for r in rows]

    fig, axes = plt.subplots(1, 2, figsize=(11, 4), constrained_layout=True)
    for ax, values, title, ylabel in (
        (axes[0], f1, "F1-Score theo cấu hình", "F1-score (%)"),
        (axes[1], fpr, "False Positive Rate theo cấu hình", "FPR (%)"),
    ):
        bars = ax.bar(labels, values, color=COLORS[: len(rows)], width=0.6)
        _bar_labels(ax, bars)
        ax.set_title(title)
        ax.set_ylabel(ylabel)
        ax.set_ylim(0, 110)
        ax.grid(axis="y", linestyle="--", alpha=0.35)
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def plot_precision_recall(rows: Sequence[Dict[str, str]], out_path: Path) -> None:
    labels = SHORT_NAMES[: len(rows)]
    precision = [float(r["Precision (%)"]) for r in rows]
    recall = [float(r["Recall (%)"]) for r in rows]

    fig, ax = plt.subplots(figsize=(7, 4), constrained_layout=True)
    width = 0.36
    x = range(len(rows))
    bars_p = ax.bar([i - width / 2 for i in x], precision, width, label="Precision", color="#4c72b0")
    bars_r = ax.bar([i + width / 2 for i in x], recall, width, label="Recall", color="#dd8452")
    _bar_labels(ax, bars_p)
    _bar_labels(ax, bars_r)
    ax.set_xticks(list(x), labels)
    ax.set_ylim(0, 110)
    ax.set_ylabel("(%)")
    ax.set_title("Precision / Recall theo cấu hình")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.legend()
    fig.savefig(out_path, dpi=160)
    plt.close(fig)


def main(argv: Sequence[str] | None = None) -> None:
    parser = argparse.ArgumentParser(description="Vẽ biểu đồ ablation từ CSV")
    parser.add_argument("--csv", default="outputs/ablation_results.csv")
    parser.add_argument("--out-dir", default="docs/report")
    args = parser.parse_args(argv)

    rows = read_results(args.csv)
    if not rows:
        raise SystemExit(f"File CSV rỗng: {args.csv}")

    out_dir = Path(args.out_dir)
    out_dir.mkdir(parents=True, exist_ok=True)
    f1_fpr = out_dir / "ablation_f1_fpr.png"
    pr = out_dir / "ablation_precision_recall.png"
    plot_f1_fpr(rows, f1_fpr)
    plot_precision_recall(rows, pr)
    print(f"Đã lưu biểu đồ: {f1_fpr}")
    print(f"Đã lưu biểu đồ: {pr}")


if __name__ == "__main__":
    main()
