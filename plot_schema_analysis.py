import csv
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np


def parse_bool(value: str) -> bool:
    return str(value).strip().lower() in {"true", "1", "yes"}


def load_rows(csv_path: Path):
    rows = []
    with csv_path.open(newline="", encoding="utf-8") as f:
        reader = csv.DictReader(f)
        for row in reader:
            if row.get("error"):
                continue
            rows.append(
                {
                    "is_linear": parse_bool(row["is_linear"]),
                    "schema_size": int(row["schema_size"]),
                }
            )
    return rows


def plot_class_distribution(rows, out_path: Path):
    linear_count = sum(1 for r in rows if r["is_linear"])
    general_count = len(rows) - linear_count
    labels = ["Linear", "General CFG"]
    counts = [linear_count, general_count]
    colors = ["#2a9d8f", "#e76f51"]
    total = max(len(rows), 1)

    fig, ax = plt.subplots(figsize=(8, 5))
    bars = ax.bar(labels, counts, color=colors, edgecolor="black", linewidth=0.8)
    ax.set_title("Grammar Class Distribution")
    ax.set_ylabel("Schema Count")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    max_count = max(counts) if counts else 1
    ax.set_ylim(0, max_count * 1.12)
    ax.margins(x=0.08)

    for bar, count in zip(bars, counts):
        pct = count / total * 100
        ax.annotate(
            f"{count}\n({pct:.1f}%)",
            xy=(bar.get_x() + bar.get_width() / 2, bar.get_height()),
            xytext=(0, 4),
            textcoords="offset points",
            ha="center",
            va="bottom",
            fontsize=10,
            clip_on=True,
        )

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def plot_schema_size_vs_is_linear(rows, out_path: Path):
    linear_sizes = [r["schema_size"] for r in rows if r["is_linear"]]
    general_sizes = [r["schema_size"] for r in rows if not r["is_linear"]]

    fig, ax = plt.subplots(figsize=(8, 5))
    bp = ax.boxplot(
        [linear_sizes, general_sizes],
        tick_labels=["Linear", "General CFG"],
        patch_artist=True,
        showfliers=False,
    )
    for patch, color in zip(bp["boxes"], ["#2a9d8f", "#e76f51"]):
        patch.set_facecolor(color)
        patch.set_alpha(0.65)

    medians = [linear_sizes, general_sizes]
    for i, values in enumerate(medians, start=1):
        if values:
            median_value = float(np.median(values))
            ax.annotate(
                f"med: {int(round(median_value))}",
                xy=(i, median_value),
                xytext=(0, 6),
                textcoords="offset points",
                ha="center",
                va="bottom",
                fontsize=8,
                zorder=5,
                clip_on=True,
            )

    ax.set_yscale("log")
    ax.set_ylabel("Schema Size (bytes, log scale)")
    ax.set_title("Schema Size vs Grammar Class")
    ax.grid(axis="y", linestyle="--", alpha=0.35)
    ax.set_axisbelow(True)
    if linear_sizes and general_sizes:
        all_sizes = linear_sizes + general_sizes
        ymin = max(min(all_sizes) * 0.8, 1)
        ymax = max(all_sizes) * 1.25
        ax.set_ylim(ymin, ymax)
    ax.margins(x=0.12)

    fig.tight_layout(pad=1.0)
    fig.savefig(out_path, dpi=220)
    plt.close(fig)


def main():
    csv_path = Path("schema_analysis.csv")
    if not csv_path.exists():
        raise FileNotFoundError("schema_analysis.csv not found in current directory")

    rows = load_rows(csv_path)
    if not rows:
        raise RuntimeError("No valid rows available to plot.")

    plot_class_distribution(rows, Path("class_distribution.png"))
    plot_schema_size_vs_is_linear(rows, Path("schema_size_vs_is_linear.png"))

    print("Generated:")
    print("  class_distribution.png")
    print("  schema_size_vs_is_linear.png")


if __name__ == "__main__":
    main()
