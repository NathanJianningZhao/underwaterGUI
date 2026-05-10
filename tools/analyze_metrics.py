#!/usr/bin/env python3
from __future__ import annotations

import argparse
import csv
import json
from pathlib import Path
from typing import Dict, Iterable, List, Tuple


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Analyze ZED mapping run bundles and generate paper-friendly summary tables."
    )
    parser.add_argument(
        "--runs-dir",
        default="exports/runs",
        help="Directory containing per-run folders created by the GUI.",
    )
    parser.add_argument(
        "--output-dir",
        default="exports/analysis",
        help="Directory where analysis outputs will be written.",
    )
    return parser.parse_args()


def safe_float(value, default: float = 0.0) -> float:
    try:
        return float(value)
    except (TypeError, ValueError):
        return default


def safe_int(value, default: int = 0) -> int:
    try:
        return int(float(value))
    except (TypeError, ValueError):
        return default


def mean(values: Iterable[float]) -> float:
    values = list(values)
    if not values:
        return 0.0
    return sum(values) / len(values)


def find_summary_file(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*_summary.json"))
    return candidates[0] if candidates else None


def find_frames_file(run_dir: Path) -> Path | None:
    candidates = sorted(run_dir.glob("*_frames.csv"))
    return candidates[0] if candidates else None


def load_json(path: Path) -> Dict:
    with path.open("r", encoding="utf-8") as f:
        return json.load(f)


def load_csv_rows(path: Path) -> List[Dict[str, str]]:
    with path.open("r", encoding="utf-8", newline="") as f:
        return list(csv.DictReader(f))


def summarize_run(run_dir: Path) -> Dict | None:
    summary_path = find_summary_file(run_dir)
    frames_path = find_frames_file(run_dir)
    if summary_path is None or frames_path is None:
        return None

    summary = load_json(summary_path)
    frames = load_csv_rows(frames_path)
    config = summary.get("config", {})

    tracking_ok = safe_int(summary.get("tracking_ok_count"))
    tracking_lost = safe_int(summary.get("tracking_lost_count"))
    tracking_total = tracking_ok + tracking_lost
    tracking_success_rate = (tracking_ok / tracking_total) if tracking_total else 0.0

    run_row = {
        "run_name": run_dir.name,
        "run_label": str(config.get("run_label", "")),
        "source_name": str(summary.get("source_name", "")),
        "path_hint": str(summary.get("path_hint", "")),
        "processed_frames": safe_int(summary.get("processed_frames")),
        "tracking_ok_count": tracking_ok,
        "tracking_lost_count": tracking_lost,
        "tracking_success_rate": tracking_success_rate,
        "final_map_points": safe_int(summary.get("final_map_points")),
        "final_traj_points": safe_int(summary.get("final_traj_points")),
        "avg_input_points": mean(safe_float(r.get("input_points")) for r in frames),
        "avg_post_underwater_points": mean(safe_float(r.get("post_underwater_points")) for r in frames),
        "avg_post_radius_points": mean(safe_float(r.get("post_radius_points")) for r in frames),
        "avg_post_voxel_points": mean(safe_float(r.get("post_voxel_points")) for r in frames),
        "avg_radius_removed": mean(safe_float(r.get("radius_removed")) for r in frames),
        "avg_voxel_removed": mean(safe_float(r.get("voxel_removed")) for r in frames),
        "depth_mode_name": str(config.get("depth_mode_name", "")),
        "zed_confidence_threshold": safe_int(config.get("zed_confidence_threshold")),
        "zed_texture_confidence_threshold": safe_int(config.get("zed_texture_confidence_threshold")),
        "depth_minimum_distance_m": safe_float(config.get("depth_minimum_distance_m")),
        "depth_maximum_distance_m": safe_float(config.get("depth_maximum_distance_m")),
        "zed_depth_stabilization": safe_int(config.get("zed_depth_stabilization")),
        "enable_radius_outlier_filter": bool(config.get("enable_radius_outlier_filter", False)),
        "radius_filter_radius_m": safe_float(config.get("radius_filter_radius_m")),
        "radius_filter_min_neighbors": safe_int(config.get("radius_filter_min_neighbors")),
        "enable_voxel_downsampling": bool(config.get("enable_voxel_downsampling", False)),
        "voxel_size_m": safe_float(config.get("voxel_size_m")),
        "underwater_enabled": bool(config.get("underwater_enabled", False)),
        "summary_path": str(summary_path),
        "frames_path": str(frames_path),
    }
    return run_row


def write_csv(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    if not rows:
        with path.open("w", encoding="utf-8", newline="") as f:
            f.write("")
        return
    fieldnames = list(rows[0].keys())
    with path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(rows)


def markdown_table(rows: List[Dict], columns: List[Tuple[str, str]]) -> str:
    header = "| " + " | ".join(label for _, label in columns) + " |"
    sep = "| " + " | ".join("---" for _ in columns) + " |"
    body = []
    for row in rows:
        values = []
        for key, _ in columns:
            value = row.get(key, "")
            if isinstance(value, float):
                if "rate" in key:
                    values.append(f"{value:.4f}")
                else:
                    values.append(f"{value:.2f}")
            else:
                values.append(str(value))
        body.append("| " + " | ".join(values) + " |")
    return "\n".join([header, sep, *body])


def write_markdown(path: Path, rows: List[Dict]) -> None:
    path.parent.mkdir(parents=True, exist_ok=True)
    run_level_columns = [
        ("run_name", "run"),
        ("run_label", "setting"),
        ("tracking_success_rate", "tracking_success_rate"),
        ("processed_frames", "processed_frames"),
        ("final_map_points", "final_map_points"),
        ("final_traj_points", "final_traj_points"),
    ]
    filter_columns = [
        ("run_name", "run"),
        ("run_label", "setting"),
        ("avg_input_points", "avg_input_points"),
        ("avg_post_radius_points", "avg_post_radius_points"),
        ("avg_post_voxel_points", "avg_post_voxel_points"),
        ("avg_radius_removed", "avg_radius_removed"),
        ("avg_voxel_removed", "avg_voxel_removed"),
    ]
    best_combined_columns = [
        ("run_name", "run"),
        ("run_label", "setting"),
        ("tracking_success_rate", "tracking_success_rate"),
        ("final_map_points", "final_map_points"),
        ("avg_post_voxel_points", "avg_post_voxel_points"),
    ]

    lines = [
        "# ZED Mapping Analysis",
        "",
        f"Runs analyzed: {len(rows)}",
        "",
        "## Table 1: Run-level metrics",
        "",
        markdown_table(rows, run_level_columns) if rows else "_No runs found._",
        "",
        "## Table 2: Filtering effect",
        "",
        markdown_table(rows, filter_columns) if rows else "_No runs found._",
        "",
        "## Table 3: Best combined setting candidates",
        "",
        markdown_table(rows, best_combined_columns) if rows else "_No runs found._",
        "",
    ]
    path.write_text("\n".join(lines), encoding="utf-8")


def analyze_runs(runs_dir: Path | str, output_dir: Path | str) -> Dict[str, object]:
    runs_dir = Path(runs_dir)
    output_dir = Path(output_dir)
    output_dir.mkdir(parents=True, exist_ok=True)
    rows: List[Dict] = []
    if runs_dir.exists():
        for run_dir in sorted(p for p in runs_dir.iterdir() if p.is_dir()):
            row = summarize_run(run_dir)
            if row is not None:
                rows.append(row)

    rows.sort(key=lambda row: row["run_name"])

    csv_path = output_dir / "run_summary.csv"
    md_path = output_dir / "analysis_report.md"
    write_csv(csv_path, rows)
    write_markdown(md_path, rows)

    return {
        "runs_dir_exists": runs_dir.exists(),
        "runs_analyzed": len(rows),
        "csv_path": csv_path,
        "markdown_path": md_path,
    }


def main() -> int:
    args = parse_args()
    result = analyze_runs(args.runs_dir, args.output_dir)

    if not result["runs_dir_exists"]:
        print(f"Runs directory not found yet: {args.runs_dir}")
    print(f"Analyzed {result['runs_analyzed']} runs")
    print(f"CSV: {result['csv_path']}")
    print(f"Markdown: {result['markdown_path']}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
