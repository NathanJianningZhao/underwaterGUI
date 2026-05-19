"""Export and run-bundle helpers for maps, trajectories, and metrics."""

from __future__ import annotations

import csv
import datetime as dt
import json
from pathlib import Path
from typing import Any, Dict, List

import numpy as np

PROJECT_ROOT = Path(__file__).resolve().parents[2]


def default_export_path(hint: str, suffix: str = "map", ext: str = ".ply") -> str:
    """Build a timestamped export path under the local exports directory."""
    stem = Path(hint).stem if hint and hint != "demo" else "demo"
    exports_dir = PROJECT_ROOT / "exports"
    exports_dir.mkdir(parents=True, exist_ok=True)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return str(exports_dir / f"{stem}_{suffix}_{timestamp}{ext}")


def sanitize_run_label(value: str) -> str:
    cleaned = "".join(ch if ch.isalnum() or ch in ("-", "_") else "_" for ch in str(value).strip())
    cleaned = cleaned.strip("_")
    return cleaned or "run"


def build_metrics_run_stem(path_hint: str) -> str:
    stem = Path(path_hint).stem if path_hint and path_hint != "demo" else path_hint or "demo"
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    return f"{sanitize_run_label(stem)}_{timestamp}"


def build_run_stem(path_hint: str, run_label: str = "") -> str:
    source_stem = Path(path_hint).stem if path_hint and path_hint != "demo" else path_hint or "demo"
    label = sanitize_run_label(run_label)
    timestamp = dt.datetime.now().strftime("%Y%m%d_%H%M%S")
    if run_label.strip():
        return f"{label}_{sanitize_run_label(source_stem)}_{timestamp}"
    return f"{sanitize_run_label(source_stem)}_{timestamp}"


def build_run_bundle_dir(path_hint: str, run_label: str = "") -> Path:
    return PROJECT_ROOT / "exports" / "runs" / build_run_stem(path_hint, run_label)


def experiment_settings_dict(config: Any) -> Dict[str, Any]:
    return {
        "depth_mode_name": config.depth_mode_name,
        "process_every": int(config.process_every),
        "offline_frame_points": int(config.offline_frame_points),
        "offline_global_cap": int(config.offline_global_cap),
        "max_distance_m": float(config.max_distance_m),
        "playback_speed": float(config.playback_speed),
        "zed_confidence_threshold": int(config.zed_confidence_threshold),
        "zed_texture_confidence_threshold": int(config.zed_texture_confidence_threshold),
        "depth_minimum_distance_m": float(config.depth_minimum_distance_m),
        "depth_maximum_distance_m": float(config.depth_maximum_distance_m),
        "zed_depth_stabilization": int(config.zed_depth_stabilization),
        "enable_radius_outlier_filter": bool(config.enable_radius_outlier_filter),
        "radius_filter_radius_m": float(config.radius_filter_radius_m),
        "radius_filter_min_neighbors": int(config.radius_filter_min_neighbors),
        "enable_voxel_downsampling": bool(config.enable_voxel_downsampling),
        "voxel_size_m": float(config.voxel_size_m),
        "log_run_metrics": bool(config.log_run_metrics),
        "run_metrics_dir": str(config.run_metrics_dir),
        "run_label": str(config.run_label),
        "auto_export_run_bundle": bool(config.auto_export_run_bundle),
        "underwater_enabled": bool(config.underwater.enabled),
    }


def write_ply(path: str, points: np.ndarray, colors_u8: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8") as f:
        f.write("ply\nformat ascii 1.0\n")
        f.write(f"element vertex {points.shape[0]}\n")
        f.write("property float x\nproperty float y\nproperty float z\n")
        f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
        f.write("end_header\n")
        for p, c in zip(points, colors_u8):
            f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")


def write_traj_csv(path: str, traj: np.ndarray) -> None:
    out = Path(path)
    out.parent.mkdir(parents=True, exist_ok=True)
    with out.open("w", encoding="utf-8", newline="") as f:
        f.write("x,y,z\n")
        for p in traj:
            f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")


def write_run_metrics(
    *,
    source_name: str,
    path_hint: str,
    processed_count: int,
    tracking_ok_count: int,
    tracking_lost_count: int,
    map_points: np.ndarray,
    traj: np.ndarray,
    frame_metrics_rows: List[Dict[str, Any]],
    config: Any,
    run_bundle_dir: Path,
    metrics_run_stem: str,
) -> tuple[str, str]:
    metrics_dir = run_bundle_dir if config.auto_export_run_bundle else Path(config.run_metrics_dir)
    metrics_dir.mkdir(parents=True, exist_ok=True)
    summary_path = metrics_dir / f"{metrics_run_stem}_summary.json"
    frames_path = metrics_dir / f"{metrics_run_stem}_frames.csv"
    summary = {
        "source_name": source_name,
        "path_hint": path_hint,
        "processed_frames": int(processed_count),
        "tracking_ok_count": int(tracking_ok_count),
        "tracking_lost_count": int(tracking_lost_count),
        "final_map_points": int(map_points.shape[0]),
        "final_traj_points": int(traj.shape[0]),
        "config": experiment_settings_dict(config),
    }
    with summary_path.open("w", encoding="utf-8") as f:
        json.dump(summary, f, indent=2)
    fieldnames = [
        "frame_index",
        "source_pos",
        "source_time_s",
        "tracking_state",
        "pose_valid",
        "input_points",
        "post_underwater_points",
        "post_radius_points",
        "post_voxel_points",
        "radius_removed",
        "voxel_removed",
        "depth_mode_name",
        "process_every",
        "offline_frame_points",
        "offline_global_cap",
        "max_distance_m",
        "playback_speed",
        "zed_confidence_threshold",
        "zed_texture_confidence_threshold",
        "depth_minimum_distance_m",
        "depth_maximum_distance_m",
        "zed_depth_stabilization",
        "enable_radius_outlier_filter",
        "radius_filter_radius_m",
        "radius_filter_min_neighbors",
        "enable_voxel_downsampling",
        "voxel_size_m",
        "log_run_metrics",
        "run_metrics_dir",
        "run_label",
        "auto_export_run_bundle",
        "underwater_enabled",
    ]
    with frames_path.open("w", encoding="utf-8", newline="") as f:
        writer = csv.DictWriter(f, fieldnames=fieldnames)
        writer.writeheader()
        writer.writerows(frame_metrics_rows)
    return str(summary_path), str(frames_path)


def write_run_bundle(
    run_bundle_dir: Path,
    map_points: np.ndarray,
    map_colors: np.ndarray,
    traj: np.ndarray,
    config: Any,
) -> Dict[str, str]:
    run_bundle_dir.mkdir(parents=True, exist_ok=True)
    map_path = run_bundle_dir / "map.ply"
    traj_path = run_bundle_dir / "trajectory.csv"
    write_ply(str(map_path), map_points, map_colors)
    write_traj_csv(str(traj_path), traj)
    config_path = run_bundle_dir / "run_config.json"
    with config_path.open("w", encoding="utf-8") as f:
        json.dump(experiment_settings_dict(config), f, indent=2)
    return {
        "map": str(map_path),
        "traj": str(traj_path),
        "config": str(config_path),
    }
