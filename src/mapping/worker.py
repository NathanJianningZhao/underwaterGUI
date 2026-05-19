"""Qt worker and configuration state for point-cloud mapping runs."""

from __future__ import annotations

import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

import numpy as np
from PySide6 import QtCore

from src.mapping.export import (
    PROJECT_ROOT,
    build_metrics_run_stem,
    build_run_bundle_dir,
    experiment_settings_dict,
    write_ply,
    write_run_bundle,
    write_run_metrics,
    write_traj_csv,
)
from src.underwater import UnderwaterConfig, UnderwaterPipeline
from src.underwater.ops import radius_outlier_filter, voxel_downsample


@dataclass
class WorkerConfig:
    process_every: int = 3
    live_frame_points: int = 400
    offline_frame_points: int = 3000
    live_global_cap: int = 20000
    offline_global_cap: int = 300000
    live_update_every: int = 20
    max_distance_m: float = 5.0
    point_size: float = 2.0
    show_grid: bool = True
    show_axes: bool = True
    show_trajectory: bool = True
    show_camera_marker: bool = True
    depth_mode_name: str = "PERFORMANCE"
    playback_speed: float = 1.0
    freeze_live_view: bool = False
    live_downsample_method: str = "RANDOM"
    offline_downsample_method: str = "RANDOM"
    live_voxel_size: float = 0.06
    offline_voxel_size: float = 0.08
    live_keep_frames: int = 60
    color_mode: str = "RGB"
    zed_confidence_threshold: int = 100
    zed_texture_confidence_threshold: int = 100
    depth_minimum_distance_m: float = 0.4
    depth_maximum_distance_m: float = 5.0
    zed_depth_stabilization: int = 30
    enable_radius_outlier_filter: bool = False
    radius_filter_radius_m: float = 0.10
    radius_filter_min_neighbors: int = 5
    enable_voxel_downsampling: bool = False
    voxel_size_m: float = 0.03
    log_run_metrics: bool = True
    run_metrics_dir: str = str(PROJECT_ROOT / "exports" / "metrics")
    run_label: str = ""
    auto_export_run_bundle: bool = True
    underwater: UnderwaterConfig = field(default_factory=UnderwaterConfig)


class ConfigStore:
    def __init__(self, config: WorkerConfig):
        self._config = config
        self._lock = threading.Lock()

    def update(self, **kwargs) -> None:
        with self._lock:
            self._config = replace(self._config, **kwargs)

    def snapshot(self) -> WorkerConfig:
        with self._lock:
            return replace(self._config)


class MappingWorker(QtCore.QObject):
    stats_updated = QtCore.Signal(dict)
    cloud_updated = QtCore.Signal(object, object)
    traj_updated = QtCore.Signal(object)
    camera_updated = QtCore.Signal(object)
    log = QtCore.Signal(str)
    finished = QtCore.Signal(bool, str)

    def __init__(self, source, source_name: str, config_store: ConfigStore, path_hint: str):
        super().__init__()
        self.source = source
        self.source_name = source_name
        self.config_store = config_store
        self.path_hint = path_hint
        self._underwater_pipeline = UnderwaterPipeline(self.config_store.snapshot().underwater)
        self._paused = False
        self._stop_requested = False
        self._export_path: Optional[str] = None
        self._traj_export_path: Optional[str] = None
        self._seek_fraction: Optional[float] = None
        self._reset_requested = False
        self._lock = threading.Lock()
        self._map_points = np.empty((0, 3), dtype=np.float32)
        self._map_colors = np.empty((0, 3), dtype=np.uint8)
        self._traj = np.empty((0, 3), dtype=np.float32)
        self._processed_count = 0
        self._last_stats: Dict = {}
        self._underwater_status = "OFF"
        self._underwater_metrics = "src=none Y=0.0 C=0.0 S=0.0 E=0.000 R/B=1.00 pts=0"
        self._underwater_notes: list[str] = []
        self._tracking_ok_count = 0
        self._tracking_lost_count = 0
        self._last_frame_metrics: Dict[str, Any] = {
            "pose_valid": False,
            "input_points": 0,
            "post_underwater_points": 0,
            "post_radius_points": 0,
            "post_voxel_points": 0,
            "radius_removed": 0,
            "voxel_removed": 0,
        }
        self._frame_metrics_rows: List[Dict[str, Any]] = []
        self._metrics_run_stem = build_metrics_run_stem(path_hint)
        initial_config = self.config_store.snapshot()
        self._run_bundle_dir = build_run_bundle_dir(path_hint, initial_config.run_label)

    @QtCore.Slot()
    def run(self) -> None:
        ok = True
        message = "Finished."
        try:
            self.source.open()
            while True:
                with self._lock:
                    stop_requested = self._stop_requested
                    paused = self._paused
                if stop_requested:
                    break
                if paused:
                    time.sleep(0.05)
                    continue
                config = self.config_store.snapshot()
                self._underwater_pipeline.update_config(config.underwater)
                self._handle_side_effect_requests()
                frame = self.source.read(config)
                if frame is None:
                    message = "Reached end of source."
                    break
                source_pos = int(frame["source_pos"])
                if config.process_every > 1 and source_pos % int(config.process_every) != 0:
                    self._emit_stats(frame, config)
                    self._sleep_for_playback(config)
                    continue
                self._processed_count += 1
                points_world = np.asarray(frame["points_world"], dtype=np.float32)
                colors_u8 = np.asarray(frame["colors_u8"], dtype=np.uint8)
                camera_pos = np.asarray(frame["camera_pos"], dtype=np.float32).reshape(3)
                pose_valid = bool(frame.get("pose_valid", True))
                input_points = int(points_world.shape[0])
                tracking_state = str(frame.get("tracking_state", "OK"))
                if tracking_state == "OK":
                    self._tracking_ok_count += 1
                else:
                    self._tracking_lost_count += 1
                result = self._underwater_pipeline.process_frame(
                    points_world=points_world,
                    colors_u8=colors_u8,
                    image_bgr=frame.get("left_bgr"),
                    depth_confidence=frame.get("depth_confidence"),
                )
                points_world = result.points_world
                colors_u8 = result.colors_u8
                post_underwater_points = int(points_world.shape[0])
                radius_removed = 0
                voxel_removed = 0
                post_radius_points = post_underwater_points
                post_voxel_points = post_underwater_points
                if config.enable_radius_outlier_filter and points_world.shape[0] > 0:
                    before_radius = int(points_world.shape[0])
                    points_world, colors_u8, _ = radius_outlier_filter(
                        points_world,
                        colors_u8,
                        config.radius_filter_radius_m,
                        config.radius_filter_min_neighbors,
                    )
                    post_radius_points = int(points_world.shape[0])
                    radius_removed = before_radius - post_radius_points
                if config.enable_voxel_downsampling and points_world.shape[0] > 0:
                    before_voxel = int(points_world.shape[0])
                    points_world, colors_u8, _ = voxel_downsample(
                        points_world,
                        colors_u8,
                        config.voxel_size_m,
                    )
                    post_voxel_points = int(points_world.shape[0])
                    voxel_removed = before_voxel - post_voxel_points
                else:
                    post_voxel_points = int(points_world.shape[0])
                self._underwater_status = result.status
                self._underwater_metrics = result.metrics.to_summary()
                frame_notes = frame.get("frame_notes", [])
                self._underwater_notes = list(frame_notes) + result.diagnostics
                self._last_frame_metrics = {
                    "pose_valid": bool(pose_valid),
                    "input_points": int(input_points),
                    "post_underwater_points": int(post_underwater_points),
                    "post_radius_points": int(post_radius_points),
                    "post_voxel_points": int(post_voxel_points),
                    "radius_removed": int(radius_removed),
                    "voxel_removed": int(voxel_removed),
                }
                if config.log_run_metrics:
                    frame_row = {
                        "frame_index": int(self._processed_count),
                        "source_pos": int(frame["source_pos"]),
                        "source_time_s": float(frame["source_time_s"]),
                        "tracking_state": tracking_state,
                        "pose_valid": int(bool(pose_valid)),
                        **self._last_frame_metrics,
                        **experiment_settings_dict(config),
                    }
                    self._frame_metrics_rows.append(frame_row)
                if self._underwater_notes and self._processed_count % max(1, int(config.live_update_every)) == 0:
                    self.log.emit("Underwater: " + " | ".join(self._underwater_notes[:3]))
                self._accumulate(points_world, colors_u8, camera_pos, pose_valid, config)
                live_pts, live_cols = self._build_live_cloud(config)
                self.cloud_updated.emit(live_pts, live_cols)
                self.traj_updated.emit(self._traj.copy())
                if pose_valid:
                    self.camera_updated.emit(camera_pos.copy())
                self._emit_stats(frame, config, live_pts.shape[0])
                self._sleep_for_playback(config)
            self._handle_side_effect_requests(flush=True)
            final_config = self.config_store.snapshot()
            if final_config.auto_export_run_bundle:
                bundle_paths = write_run_bundle(self._run_bundle_dir, self._map_points, self._map_colors, self._traj, final_config)
                self._last_stats["last_export_path"] = bundle_paths["map"]
                self._last_stats["last_traj_path"] = bundle_paths["traj"]
                self.log.emit(f"Run bundle saved: {self._run_bundle_dir}")
            if final_config.log_run_metrics:
                summary_path, frames_path = write_run_metrics(
                        source_name=self.source_name,
                        path_hint=self.path_hint,
                        processed_count=self._processed_count,
                        tracking_ok_count=self._tracking_ok_count,
                        tracking_lost_count=self._tracking_lost_count,
                        map_points=self._map_points,
                        traj=self._traj,
                        frame_metrics_rows=self._frame_metrics_rows,
                        config=final_config,
                        run_bundle_dir=self._run_bundle_dir,
                        metrics_run_stem=self._metrics_run_stem,
                    )
                self._last_stats["last_summary_path"] = summary_path
                self._last_stats["last_frames_path"] = frames_path
                self.stats_updated.emit(dict(self._last_stats))
                self.log.emit(f"Metrics written: {summary_path}")
                self.log.emit(f"Per-frame metrics written: {frames_path}")
        except Exception as exc:
            ok = False
            message = str(exc)
            self.log.emit(f"Worker error: {exc}")
        finally:
            try:
                self.source.close()
            except Exception:
                pass
            self.finished.emit(ok, message)

    def _accumulate(
        self,
        points_world: np.ndarray,
        colors_u8: np.ndarray,
        camera_pos: np.ndarray,
        pose_valid: bool,
        config: WorkerConfig,
    ) -> None:
        if self._reset_requested:
            self._map_points = np.empty((0, 3), dtype=np.float32)
            self._map_colors = np.empty((0, 3), dtype=np.uint8)
            self._traj = np.empty((0, 3), dtype=np.float32)
            self._reset_requested = False
        if points_world.size:
            self._map_points = np.vstack([self._map_points, points_world])
            self._map_colors = np.vstack([self._map_colors, colors_u8])
            if self._map_points.shape[0] > config.offline_global_cap:
                idx = np.random.choice(self._map_points.shape[0], int(config.offline_global_cap), replace=False)
                self._map_points = self._map_points[idx]
                self._map_colors = self._map_colors[idx]
        if pose_valid:
            self._traj = np.vstack([self._traj, camera_pos.reshape(1, 3)])

    def _build_live_cloud(self, config: WorkerConfig):
        if self._map_points.shape[0] == 0:
            return np.empty((0, 3), dtype=np.float32), np.empty((0, 3), dtype=np.uint8)
        # Keep rendering focused around the latest camera position so long
        # mapping runs stay responsive even when the full map is larger.
        keep_frames = max(1, int(config.live_keep_frames))
        traj_count = min(self._traj.shape[0], keep_frames)
        recent_centers = self._traj[-traj_count:]
        if recent_centers.size == 0:
            recent_mask = np.arange(self._map_points.shape[0])
        else:
            last_center = recent_centers[-1]
            dist = np.linalg.norm(self._map_points - last_center, axis=1)
            recent_mask = np.flatnonzero(dist < max(1.0, config.max_distance_m * 1.5))
            if recent_mask.size == 0:
                recent_mask = np.arange(self._map_points.shape[0])
        pts = self._map_points[recent_mask]
        cols = self._map_colors[recent_mask]
        if pts.shape[0] > config.live_global_cap:
            idx = np.random.choice(pts.shape[0], int(config.live_global_cap), replace=False)
            pts = pts[idx]
            cols = cols[idx]
        return pts.astype(np.float32), cols.astype(np.uint8)

    def _emit_stats(self, frame: Dict, config: WorkerConfig, live_rendered_points: int = 0) -> None:
        stats = {
            "backend": self.source_name,
            "frame_count": int(frame["source_pos"]),
            "processed_count": self._processed_count,
            "tracking_state": frame.get("tracking_state", "OK"),
            "live_rendered_points": int(live_rendered_points),
            "source_pos": int(frame["source_pos"]),
            "source_len": int(frame["source_len"]),
            "source_time_s": float(frame["source_time_s"]),
            "last_export_path": self._last_stats.get("last_export_path", ""),
            "last_traj_path": self._last_stats.get("last_traj_path", ""),
            "last_summary_path": self._last_stats.get("last_summary_path", ""),
            "last_frames_path": self._last_stats.get("last_frames_path", ""),
            "last_error": self._last_stats.get("last_error", ""),
            "underwater_status": self._underwater_status,
            "underwater_metrics": self._underwater_metrics,
            "tracking_ok_count": self._tracking_ok_count,
            "tracking_lost_count": self._tracking_lost_count,
            "input_points": self._last_frame_metrics["input_points"],
            "post_underwater_points": self._last_frame_metrics["post_underwater_points"],
            "post_radius_points": self._last_frame_metrics["post_radius_points"],
            "post_voxel_points": self._last_frame_metrics["post_voxel_points"],
        }
        self._last_stats = stats
        self.stats_updated.emit(stats)

    def _handle_side_effect_requests(self, flush: bool = False) -> None:
        # File exports and seeking are requested by the UI thread, then
        # performed here on the worker thread beside the frame loop.
        with self._lock:
            export_path = self._export_path
            traj_export_path = self._traj_export_path
            seek_fraction = self._seek_fraction
            self._export_path = None
            self._traj_export_path = None
            self._seek_fraction = None
        if seek_fraction is not None:
            self.source.seek(seek_fraction)
            self.log.emit(f"Seeked to {seek_fraction:.2%}.")
        if export_path is not None:
            try:
                write_ply(export_path, self._map_points, self._map_colors)
                self._last_stats["last_export_path"] = export_path
                self.log.emit(f"Exported map: {export_path}")
            except Exception as exc:
                self._last_stats["last_error"] = str(exc)
                self.log.emit(f"Export failed: {exc}")
        if traj_export_path is not None:
            try:
                write_traj_csv(traj_export_path, self._traj)
                self.log.emit(f"Exported trajectory: {traj_export_path}")
            except Exception as exc:
                self._last_stats["last_error"] = str(exc)
                self.log.emit(f"Trajectory export failed: {exc}")
        if flush:
            return

    def _sleep_for_playback(self, config: WorkerConfig) -> None:
        speed = max(0.25, float(config.playback_speed))
        remaining = 0.01 / speed
        while remaining > 0:
            with self._lock:
                if self._stop_requested or self._paused:
                    return
            step = min(0.02, remaining)
            time.sleep(step)
            remaining -= step

    @QtCore.Slot(bool)
    def set_paused(self, paused: bool) -> None:
        with self._lock:
            self._paused = bool(paused)
        self.log.emit("Paused." if paused else "Resumed.")

    @QtCore.Slot()
    def request_stop(self) -> None:
        with self._lock:
            self._stop_requested = True

    @QtCore.Slot(str)
    def request_export(self, path: str) -> None:
        with self._lock:
            self._export_path = path

    @QtCore.Slot(str)
    def request_export_traj(self, path: str) -> None:
        with self._lock:
            self._traj_export_path = path

    @QtCore.Slot(float)
    def request_seek(self, fraction: float) -> None:
        with self._lock:
            self._seek_fraction = float(fraction)

    @QtCore.Slot()
    def request_reset_mapping(self) -> None:
        self._reset_requested = True
        self.log.emit("Mapping reset requested.")

    def snapshot_map(self):
        with self._lock:
            return self._map_points.copy(), self._map_colors.copy()

    def snapshot_traj(self):
        with self._lock:
            return self._traj.copy()
