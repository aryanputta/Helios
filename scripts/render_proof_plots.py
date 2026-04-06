#!/usr/bin/env python3
"""
Render visual proof artifacts from Helios benchmark and validation outputs.
"""

from __future__ import annotations

import argparse
import json
from pathlib import Path
from typing import Any


def load_json(path: Path) -> dict[str, Any]:
    with path.open("r", encoding="utf-8") as handle:
        payload = json.load(handle)
    if not isinstance(payload, dict):
        raise ValueError(f"{path} did not contain a JSON object")
    return payload


def parse_float(payload: dict[str, Any], key: str, default: float = 0.0) -> float:
    raw = payload.get(key, default)
    try:
        return float(raw)
    except (TypeError, ValueError):
        return default


def parse_bool(payload: dict[str, Any], key: str, default: bool = False) -> bool:
    raw = payload.get(key)
    if raw is None:
        return default
    if isinstance(raw, bool):
        return raw
    return str(raw).strip().lower() in {"1", "true", "yes"}


def short_dataset_label(payload: dict[str, Any]) -> str:
    workload = str(payload.get("workload", "workload"))
    dataset = str(payload.get("dataset_input_path") or payload.get("dataset") or workload)
    if dataset.startswith("synthetic_dense_sanity"):
        rows = payload.get("rows", "?")
        cols = payload.get("cols", "?")
        return f"{workload}:{rows}x{cols}"
    return f"{workload}:{Path(dataset).stem}"


def time_saved_pct(speedup_vs_scalar: float) -> float:
    if speedup_vs_scalar <= 0.0:
        return 0.0
    return max(0.0, (1.0 - (1.0 / speedup_vs_scalar)) * 100.0)


def collect_result_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        selected_backend = str(payload.get("selected_backend", "selected"))
        scalar_ms = parse_float(payload, "scalar_median_ms")
        selected_ms = parse_float(payload, f"{selected_backend}_median_ms", parse_float(payload, "median_ms"))
        speedup_vs_scalar = parse_float(payload, f"{selected_backend}_speedup_vs_scalar")
        if speedup_vs_scalar <= 0.0 and scalar_ms > 0.0 and selected_ms > 0.0:
            speedup_vs_scalar = scalar_ms / selected_ms
        rows.append(
            {
                "label": short_dataset_label(payload),
                "workload": str(payload.get("workload", "")),
                "selected_backend": selected_backend,
                "scalar_ms": scalar_ms,
                "selected_ms": selected_ms,
                "speedup_vs_scalar": speedup_vs_scalar,
                "time_saved_pct": time_saved_pct(speedup_vs_scalar),
                "effective_gflops": parse_float(payload, f"{selected_backend}_effective_gflops"),
                "is_real_data": "dataset_input_path" in payload,
            }
        )
    return rows


def collect_validation_rows(paths: list[Path]) -> list[dict[str, Any]]:
    rows: list[dict[str, Any]] = []
    for path in paths:
        payload = load_json(path)
        rows.append(
            {
                "label": short_dataset_label(payload),
                "status": parse_bool(payload, "validation_passed"),
            }
        )
    return rows


def render_dashboard(
    report_payload: dict[str, Any],
    result_rows: list[dict[str, Any]],
    validation_rows: list[dict[str, Any]],
    output_path: Path,
) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [row["label"] for row in result_rows]
    scalar_ms = [row["scalar_ms"] for row in result_rows]
    selected_ms = [row["selected_ms"] for row in result_rows]
    speedups = [row["speedup_vs_scalar"] for row in result_rows]
    saved_pcts = [row["time_saved_pct"] for row in result_rows]
    colors = ["#1f9d8b" if row["is_real_data"] else "#457b9d" for row in result_rows]

    planner_accuracy = parse_float(report_payload, "planner_accuracy") * 100.0
    planner_wins = int(float(report_payload.get("planner_selected_won_count", 0)))
    planner_total = int(float(report_payload.get("planner_observation_count", 0)))

    fig = plt.figure(figsize=(14, 10), constrained_layout=True)
    fig.patch.set_facecolor("#f7f5ef")
    grid = fig.add_gridspec(2, 2, height_ratios=[1.2, 1.0])

    ax_latency = fig.add_subplot(grid[0, 0])
    ax_speedup = fig.add_subplot(grid[0, 1])
    ax_planner = fig.add_subplot(grid[1, 0])
    ax_validation = fig.add_subplot(grid[1, 1])

    x = np.arange(len(labels))
    width = 0.34
    ax_latency.bar(x - width / 2, scalar_ms, width, label="scalar baseline", color="#d9c27c")
    ax_latency.bar(x + width / 2, selected_ms, width, label="selected backend", color=colors)
    ax_latency.set_title("Median latency vs scalar baseline", fontsize=13, weight="bold")
    ax_latency.set_ylabel("milliseconds")
    ax_latency.set_xticks(x)
    ax_latency.set_xticklabels(labels, rotation=12)
    ax_latency.legend(frameon=False)
    ax_latency.grid(axis="y", alpha=0.2)

    ax_speedup.bar(labels, speedups, color=colors)
    ax_speedup.set_title("Measured speedup and time saved", fontsize=13, weight="bold")
    ax_speedup.set_ylabel("speedup vs scalar (x)")
    ax_speedup.grid(axis="y", alpha=0.2)
    for idx, speedup in enumerate(speedups):
        ax_speedup.text(
            idx,
            speedup,
            f"{speedup:.2f}x\n{saved_pcts[idx]:.1f}% saved",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )

    ax_planner.set_xlim(0, 100)
    ax_planner.set_ylim(0, 1)
    ax_planner.barh([0.55], [planner_accuracy], height=0.24, color="#2a9d8f")
    ax_planner.barh([0.55], [100.0 - planner_accuracy], left=[planner_accuracy], height=0.24, color="#e9dcc3")
    ax_planner.set_title("Planner proof", fontsize=13, weight="bold")
    ax_planner.set_xlabel("accuracy (%)")
    ax_planner.set_yticks([])
    ax_planner.text(
        2,
        0.18,
        f"selected backend matched measured winner on {planner_wins}/{planner_total} compare-baseline runs",
        fontsize=11,
    )
    ax_planner.text(
        min(planner_accuracy + 1.5, 90),
        0.55,
        f"{planner_accuracy:.1f}%",
        va="center",
        fontsize=15,
        weight="bold",
    )

    ax_validation.axis("off")
    ax_validation.set_title("Correctness checks", fontsize=13, weight="bold")
    if validation_rows:
        for idx, row in enumerate(validation_rows):
            status_text = "PASS" if row["status"] else "FAIL"
            facecolor = "#d8f3dc" if row["status"] else "#f4c7c3"
            ax_validation.text(
                0.02,
                0.88 - idx * 0.22,
                f"{row['label']}  {status_text}",
                transform=ax_validation.transAxes,
                fontsize=12,
                weight="bold",
                bbox={"boxstyle": "round,pad=0.45", "facecolor": facecolor, "edgecolor": "none"},
            )
    else:
        ax_validation.text(0.02, 0.88, "No validation JSONs were provided.", transform=ax_validation.transAxes)

    fig.suptitle("Helios Proof Dashboard", fontsize=18, weight="bold")
    fig.text(
        0.5,
        0.02,
        "Green/blue selected-backend bars show measured wins. Gold bars show scalar baseline cost.",
        ha="center",
        fontsize=10,
    )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def render_latency_chart(result_rows: list[dict[str, Any]], output_path: Path) -> None:
    import matplotlib

    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    import numpy as np

    output_path.parent.mkdir(parents=True, exist_ok=True)

    labels = [row["label"] for row in result_rows]
    scalar_ms = [row["scalar_ms"] for row in result_rows]
    selected_ms = [row["selected_ms"] for row in result_rows]
    colors = ["#1f9d8b" if row["is_real_data"] else "#457b9d" for row in result_rows]

    x = np.arange(len(labels))
    width = 0.34
    fig, ax = plt.subplots(figsize=(12, 6), constrained_layout=True)
    fig.patch.set_facecolor("#fcfbf7")
    ax.bar(x - width / 2, scalar_ms, width, label="scalar baseline", color="#d9c27c")
    ax.bar(x + width / 2, selected_ms, width, label="selected backend", color=colors)
    ax.set_title("Helios latency proof", fontsize=16, weight="bold")
    ax.set_ylabel("median milliseconds")
    ax.set_xticks(x)
    ax.set_xticklabels(labels, rotation=12)
    ax.legend(frameon=False)
    ax.grid(axis="y", alpha=0.2)
    for idx, row in enumerate(result_rows):
        ax.text(
            idx + width / 2,
            row["selected_ms"],
            f"{row['speedup_vs_scalar']:.2f}x\n{row['time_saved_pct']:.1f}% saved",
            ha="center",
            va="bottom",
            fontsize=10,
            weight="bold",
        )
    fig.savefig(output_path, dpi=180)
    plt.close(fig)


def main() -> int:
    parser = argparse.ArgumentParser(description="Render proof images from Helios outputs.")
    parser.add_argument("--report-json", required=True, help="Path to proof_report.json")
    parser.add_argument("--result", action="append", default=[], help="Path to a benchmark result JSON")
    parser.add_argument("--validation", action="append", default=[], help="Path to a validation JSON")
    parser.add_argument("--output-dir", default="results/reports", help="Directory for PNG outputs")
    args = parser.parse_args()

    report_path = Path(args.report_json).resolve()
    result_paths = [Path(path).resolve() for path in args.result]
    validation_paths = [Path(path).resolve() for path in args.validation]
    output_dir = Path(args.output_dir).resolve()

    if not report_path.exists():
        raise SystemExit(f"Missing report JSON: {report_path}")

    try:
        import matplotlib  # noqa: F401
    except ImportError:
        print("[plots] matplotlib is not installed; skipping visual proof artifacts.")
        return 0

    report_payload = load_json(report_path)
    result_rows = collect_result_rows(result_paths)
    validation_rows = collect_validation_rows(validation_paths)

    if not result_rows:
        print("[plots] No benchmark result JSONs were provided; skipping visual proof artifacts.")
        return 0

    dashboard_path = output_dir / "proof_dashboard.png"
    latency_path = output_dir / "proof_latency_vs_scalar.png"
    render_dashboard(report_payload, result_rows, validation_rows, dashboard_path)
    render_latency_chart(result_rows, latency_path)

    print(f"[plots] dashboard={dashboard_path}")
    print(f"[plots] latency={latency_path}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
