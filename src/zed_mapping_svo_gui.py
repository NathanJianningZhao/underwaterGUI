#!/usr/bin/env python3
"""
Docked PyVista GUI for ZED SVO mapping.

This shareable frontend keeps GUI responsibilities focused on controls,
visualization, monitoring, and worker orchestration.

Runtime expectations:
- `pyzed.sl` must be installed to run real SVO playback.
- Without `pyzed`, the app can still run in Demo mode for UI testing.
- The central viewer is lazy-loaded so the main window can open before PyVista/VTK
  initializes.
"""

from __future__ import annotations

import csv
import datetime as dt
import json
import math
import os
import sys
import threading
import time
from dataclasses import dataclass, field, replace
from pathlib import Path
from typing import Any, Dict, List, Optional

# The Windows launchers run this file directly, so make the project root
# importable without requiring users to install the package first.
PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

EXTRA_SITE_PACKAGES = os.environ.get("ZED_MAPPING_EXTRA_SITE_PACKAGES", "").strip()
if EXTRA_SITE_PACKAGES:
    for entry in EXTRA_SITE_PACKAGES.split(os.pathsep):
        site_path = Path(entry.strip())
        if site_path.exists() and str(site_path) not in sys.path:
            sys.path.append(str(site_path))

if os.name == "nt" and hasattr(os, "add_dll_directory"):
    # ZED's Python bindings depend on native DLLs that are not always on PATH
    # in fresh Windows shells. Add the common SDK locations before importing.
    zed_sdk_root = Path(os.environ.get("ZED_SDK_ROOT_DIR", r"C:\Program Files (x86)\ZED SDK"))
    dll_dirs = [
        zed_sdk_root / "bin",
        zed_sdk_root / "dependencies" / "freeglut_2.8" / "x64",
        zed_sdk_root / "dependencies" / "glew-1.12.0" / "x64",
        zed_sdk_root / "dependencies" / "opencv_3.1.0" / "x64",
        Path(r"C:\Windows\System32"),
    ]
    extra_dll_dirs = os.environ.get("ZED_MAPPING_EXTRA_DLL_DIRS", "").strip()
    if extra_dll_dirs:
        dll_dirs.extend(Path(entry.strip()) for entry in extra_dll_dirs.split(os.pathsep) if entry.strip())
    for dll_dir in dll_dirs:
        if dll_dir.exists():
            try:
                os.add_dll_directory(str(dll_dir))
            except OSError:
                pass

import numpy as np
import pyvista as pv
from pyvistaqt import QtInteractor
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtGui import QColor, QPalette
from PySide6.QtWidgets import (
    QApplication,
    QCheckBox,
    QComboBox,
    QDockWidget,
    QFileDialog,
    QFormLayout,
    QGroupBox,
    QHBoxLayout,
    QLabel,
    QLineEdit,
    QMainWindow,
    QMessageBox,
    QPushButton,
    QScrollArea,
    QDoubleSpinBox,
    QSpinBox,
    QTextEdit,
    QVBoxLayout,
    QWidget,
)
from src.underwater import UnderwaterConfig, UnderwaterPipeline
from src.underwater.ops import radius_outlier_filter, voxel_downsample
from tools.analyze_metrics import analyze_runs

USE_EMBEDDED_BACKEND = True

SOURCE_DEMO = "Demo"
SOURCE_SVO = "SVO Playback"
SOURCE_LIVE_ZED = "Live ZED Camera"


if USE_EMBEDDED_BACKEND:
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


    class IntParamWidget(QSpinBox):
        def __init__(self, minimum: int, maximum: int, value: int):
            super().__init__()
            self.setRange(int(minimum), int(maximum))
            self.setValue(int(value))


    class FloatParamWidget(QDoubleSpinBox):
        def __init__(self, minimum: float, maximum: float, value: float, step: float):
            super().__init__()
            self.setRange(float(minimum), float(maximum))
            self.setSingleStep(float(step))
            self.setDecimals(3)
            self.setValue(float(value))


    def default_export_path(hint: str, suffix: str = "map", ext: str = ".ply") -> str:
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


    def experiment_settings_dict(config: WorkerConfig) -> Dict[str, Any]:
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


    def has_pyzed() -> bool:
        try:
            import pyzed.sl as sl  # noqa: F401
            return True
        except Exception:
            return False


    def pyzed_status_message() -> str:
        version = f"{sys.version_info.major}.{sys.version_info.minor}.{sys.version_info.micro}"
        return (
            "ZED SVO playback requires the Stereolabs ZED SDK Python bindings (`pyzed.sl`). "
            f"They are not available in this environment (Python {version}). "
            "Install a ZED SDK build that matches a supported Python version, then relaunch the GUI from that interpreter."
        )


    def quat_to_rot_matrix(qx: float, qy: float, qz: float, qw: float) -> np.ndarray:
        q = np.array([qx, qy, qz, qw], dtype=np.float32)
        n = float(np.linalg.norm(q))
        if n == 0.0:
            return np.eye(3, dtype=np.float32)
        x, y, z, w = q / n
        xx, yy, zz = x * x, y * y, z * z
        xy, xz, yz = x * y, x * z, y * z
        wx, wy, wz = w * x, w * y, w * z
        return np.array(
            [
                [1 - 2 * (yy + zz), 2 * (xy - wz), 2 * (xz + wy)],
                [2 * (xy + wz), 1 - 2 * (xx + zz), 2 * (yz - wx)],
                [2 * (xz - wy), 2 * (yz + wx), 1 - 2 * (xx + yy)],
            ],
            dtype=np.float32,
        )


    def zed_xyzrgba_to_xyz_rgb(point_cloud_mat, max_points: int, max_distance_m: float):
        pc_np = point_cloud_mat.get_data()
        if pc_np is None or pc_np.ndim != 3 or pc_np.shape[2] != 4:
            return None, None
        xyz = pc_np[:, :, 0:3].reshape(-1, 3)
        # The SDK packs RGBA bytes into the fourth float channel.
        rgba_float = pc_np[:, :, 3].reshape(-1).astype(np.float32)
        rgba_u8 = np.frombuffer(rgba_float.tobytes(), dtype=np.uint8).reshape(-1, 4)
        rgb = rgba_u8[:, :3].astype(np.uint8)
        valid = np.isfinite(xyz).all(axis=1)
        xyz = xyz[valid]
        rgb = rgb[valid]
        if xyz.size == 0:
            return None, None
        dist = np.linalg.norm(xyz, axis=1)
        keep = dist < float(max_distance_m)
        xyz = xyz[keep]
        rgb = rgb[keep]
        if xyz.size == 0:
            return None, None
        if xyz.shape[0] > max_points:
            idx = np.random.choice(xyz.shape[0], max_points, replace=False)
            xyz = xyz[idx]
            rgb = rgb[idx]
        return xyz.astype(np.float32), rgb.astype(np.uint8)


    def retrieve_zed_left_bgr(cam, image_mat, sl_module) -> tuple[Optional[np.ndarray], List[str]]:
        notes: List[str] = []
        try:
            cam.retrieve_image(image_mat, sl_module.VIEW.LEFT)
            image_rgba = image_mat.get_data()
            if image_rgba is not None and image_rgba.ndim == 3 and image_rgba.shape[2] >= 3:
                # ZED returns RGB/RGBA data, while OpenCV-style filters expect BGR.
                return np.ascontiguousarray(image_rgba[:, :, :3][:, :, ::-1]), notes
        except Exception:
            notes.append("Left image retrieval unavailable for this frame.")
        return None, notes


    def retrieve_zed_confidence(cam, confidence_mat, sl_module) -> tuple[Optional[np.ndarray], List[str]]:
        notes: List[str] = []
        try:
            if hasattr(sl_module.MEASURE, "CONFIDENCE"):
                cam.retrieve_measure(confidence_mat, sl_module.MEASURE.CONFIDENCE)
                return confidence_mat.get_data(), notes
            notes.append("Depth confidence measure is not exposed by this ZED SDK build.")
        except Exception:
            notes.append("Depth confidence retrieval unavailable for this frame.")
        return None, notes


    class DemoFrameSource:
        def __init__(self, total_frames: int = 8000, base_fps: float = 30.0):
            self.total_frames = int(total_frames)
            self.base_fps = float(base_fps)
            self.frame_idx = -1
            self.scene_points = self._build_scene()

        def _build_scene(self) -> np.ndarray:
            rng = np.random.default_rng(1234)
            floor_x = rng.uniform(-4.5, 4.5, 12000)
            floor_y = rng.uniform(-4.5, 4.5, 12000)
            floor_z = np.zeros_like(floor_x)
            cube = rng.uniform(-1.2, 1.2, (6000, 3))
            cube[:, 2] = np.abs(cube[:, 2]) * 1.2 + 0.1
            return np.vstack([np.column_stack([floor_x, floor_y, floor_z]), cube]).astype(np.float32)

        def open(self) -> None:
            self.frame_idx = -1

        def seek(self, fraction: float) -> None:
            self.frame_idx = int(np.clip(fraction, 0.0, 1.0) * max(0, self.total_frames - 1)) - 1

        def read(self, config: WorkerConfig):
            self.frame_idx += 1
            if self.frame_idx >= self.total_frames:
                return None
            theta = self.frame_idx * 0.03
            cam = np.array([2.0 * math.cos(theta), 2.0 * math.sin(theta), 1.1 + 0.1 * math.sin(theta * 0.5)], dtype=np.float32)
            dist = np.linalg.norm(self.scene_points - cam, axis=1)
            pts = self.scene_points[dist < max(1.0, config.max_distance_m)]
            if pts.shape[0] == 0:
                pts = self.scene_points[: min(len(self.scene_points), config.offline_frame_points)]
            if pts.shape[0] > config.offline_frame_points:
                idx = np.random.choice(pts.shape[0], config.offline_frame_points, replace=False)
                pts = pts[idx]
            colors = np.zeros((pts.shape[0], 3), dtype=np.uint8)
            colors[:, 0] = np.clip((pts[:, 0] + 5.0) / 10.0 * 255.0, 0, 255).astype(np.uint8)
            colors[:, 1] = np.clip((pts[:, 1] + 5.0) / 10.0 * 255.0, 0, 255).astype(np.uint8)
            colors[:, 2] = np.clip((pts[:, 2] + 1.0) / 3.0 * 255.0, 0, 255).astype(np.uint8)
            return {
                "points_world": pts,
                "colors_u8": colors,
                "camera_pos": cam,
                "source_pos": self.frame_idx,
                "source_len": self.total_frames,
                "source_time_s": self.frame_idx / self.base_fps,
                "tracking_state": "OK",
                "left_bgr": None,
                "depth_confidence": None,
                "frame_notes": ["Demo mode has no camera image or depth-confidence input."],
            }

        def close(self) -> None:
            # Demo mode owns no camera or SDK resources.
            pass


    class ZEDSVOFrameSource:
        def __init__(self, path: str, config: WorkerConfig):
            self.path = path
            self.depth_mode_name = config.depth_mode_name
            self._init_config = replace(config)
            self.cam = None
            self.runtime = None
            self.pose = None
            self.pc_mat = None
            self.sl = None
            self.total_frames = 0

        def open(self) -> None:
            import pyzed.sl as sl

            self.sl = sl
            init = sl.InitParameters()
            init.set_from_svo_file(self.path)
            init.coordinate_units = sl.UNIT.METER
            init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
            depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode_name, sl.DEPTH_MODE.PERFORMANCE)
            init.depth_mode = depth_mode
            init.depth_minimum_distance = float(self._init_config.depth_minimum_distance_m)
            init.depth_maximum_distance = float(self._init_config.depth_maximum_distance_m)
            if hasattr(init, "depth_stabilization"):
                init.depth_stabilization = int(self._init_config.zed_depth_stabilization)

            self.cam = sl.Camera()
            err = self.cam.open(init)
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to open SVO: {err}")

            tracking = sl.PositionalTrackingParameters()
            tracking.enable_imu_fusion = False
            err = self.cam.enable_positional_tracking(tracking)
            if err != sl.ERROR_CODE.SUCCESS:
                self.cam.close()
                raise RuntimeError(f"Failed to enable positional tracking: {err}")

            self.runtime = sl.RuntimeParameters()
            self.runtime.confidence_threshold = int(self._init_config.zed_confidence_threshold)
            if hasattr(self.runtime, "texture_confidence_threshold"):
                self.runtime.texture_confidence_threshold = int(self._init_config.zed_texture_confidence_threshold)
            self.pose = sl.Pose()
            self.pc_mat = sl.Mat()
            self.image_mat = sl.Mat()
            self.confidence_mat = sl.Mat()
            self.total_frames = int(self.cam.get_svo_number_of_frames())

        def seek(self, fraction: float) -> None:
            if self.cam is None or self.total_frames <= 0:
                return
            target = int(np.clip(fraction, 0.0, 1.0) * max(0, self.total_frames - 1))
            self.cam.set_svo_position(target)

        def read(self, config: WorkerConfig):
            sl = self.sl
            if self.cam is None or sl is None:
                raise RuntimeError("SVO source is not open.")
            self.runtime.confidence_threshold = int(config.zed_confidence_threshold)
            if hasattr(self.runtime, "texture_confidence_threshold"):
                self.runtime.texture_confidence_threshold = int(config.zed_texture_confidence_threshold)
            grab_err = self.cam.grab(self.runtime)
            if grab_err == sl.ERROR_CODE.END_OF_SVOFILE_REACHED:
                return None
            if grab_err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Grab failed: {grab_err}")
            state = self.cam.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
            if state != sl.POSITIONAL_TRACKING_STATE.OK:
                return {
                    "points_world": np.empty((0, 3), dtype=np.float32),
                    "colors_u8": np.empty((0, 3), dtype=np.uint8),
                    "camera_pos": np.zeros(3, dtype=np.float32),
                    "pose_valid": False,
                    "source_pos": int(self.cam.get_svo_position()),
                    "source_len": self.total_frames,
                    "source_time_s": float(self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()) / 1000.0,
                    "tracking_state": str(state),
                    "left_bgr": None,
                    "depth_confidence": None,
                    "frame_notes": ["Positional tracking is not OK; image and depth-confidence inputs were skipped."],
                }

            translation = self.pose.get_translation(sl.Translation())
            tx, ty, tz = translation.get()
            orientation = self.pose.get_orientation(sl.Orientation())
            qx, qy, qz, qw = orientation.get()
            self.cam.retrieve_measure(self.pc_mat, sl.MEASURE.XYZRGBA)
            xyz_cam, rgb = zed_xyzrgba_to_xyz_rgb(self.pc_mat, config.offline_frame_points, config.max_distance_m)
            left_bgr, frame_notes = retrieve_zed_left_bgr(self.cam, self.image_mat, sl)
            confidence_map, confidence_notes = retrieve_zed_confidence(self.cam, self.confidence_mat, sl)
            frame_notes.extend(confidence_notes)
            if xyz_cam is None:
                xyz_world = np.empty((0, 3), dtype=np.float32)
                rgb = np.empty((0, 3), dtype=np.uint8)
            else:
                rot = quat_to_rot_matrix(qx, qy, qz, qw)
                t = np.array([tx, ty, tz], dtype=np.float32)
                xyz_world = (rot @ xyz_cam.T).T + t
            return {
                "points_world": xyz_world,
                "colors_u8": rgb,
                "camera_pos": np.array([tx, ty, tz], dtype=np.float32),
                "pose_valid": True,
                "source_pos": int(self.cam.get_svo_position()),
                "source_len": self.total_frames,
                "source_time_s": float(self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()) / 1000.0,
                "tracking_state": "OK",
                "left_bgr": left_bgr,
                "depth_confidence": confidence_map,
                "frame_notes": frame_notes,
            }

        def close(self) -> None:
            if self.cam is not None and self.sl is not None:
                try:
                    self.cam.disable_positional_tracking()
                except Exception:
                    # Some SDK builds raise if tracking was never fully enabled.
                    pass
                self.cam.close()
            self.cam = None


    class ZEDLiveFrameSource:
        def __init__(self, config: WorkerConfig):
            self.depth_mode_name = config.depth_mode_name
            self._init_config = replace(config)
            self.cam = None
            self.runtime = None
            self.pose = None
            self.pc_mat = None
            self.image_mat = None
            self.confidence_mat = None
            self.sl = None
            self.frame_idx = 0

        def open(self) -> None:
            import pyzed.sl as sl

            self.sl = sl
            init = sl.InitParameters()
            init.coordinate_units = sl.UNIT.METER
            init.coordinate_system = sl.COORDINATE_SYSTEM.RIGHT_HANDED_Z_UP
            depth_mode = getattr(sl.DEPTH_MODE, self.depth_mode_name, sl.DEPTH_MODE.PERFORMANCE)
            init.depth_mode = depth_mode
            init.depth_minimum_distance = float(self._init_config.depth_minimum_distance_m)
            init.depth_maximum_distance = float(self._init_config.depth_maximum_distance_m)
            if hasattr(init, "depth_stabilization"):
                init.depth_stabilization = int(self._init_config.zed_depth_stabilization)

            self.cam = sl.Camera()
            err = self.cam.open(init)
            if err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Failed to open live ZED camera: {err}")

            tracking = sl.PositionalTrackingParameters()
            tracking.enable_imu_fusion = False
            err = self.cam.enable_positional_tracking(tracking)
            if err != sl.ERROR_CODE.SUCCESS:
                self.cam.close()
                raise RuntimeError(f"Failed to enable positional tracking: {err}")

            self.runtime = sl.RuntimeParameters()
            self.runtime.confidence_threshold = int(self._init_config.zed_confidence_threshold)
            if hasattr(self.runtime, "texture_confidence_threshold"):
                self.runtime.texture_confidence_threshold = int(self._init_config.zed_texture_confidence_threshold)
            self.pose = sl.Pose()
            self.pc_mat = sl.Mat()
            self.image_mat = sl.Mat()
            self.confidence_mat = sl.Mat()
            self.frame_idx = 0

        def seek(self, fraction: float) -> None:
            # Live cameras cannot seek; keep the same method shape as SVO/demo sources.
            return

        def read(self, config: WorkerConfig):
            sl = self.sl
            if self.cam is None or sl is None:
                raise RuntimeError("Live ZED source is not open.")
            self.runtime.confidence_threshold = int(config.zed_confidence_threshold)
            if hasattr(self.runtime, "texture_confidence_threshold"):
                self.runtime.texture_confidence_threshold = int(config.zed_texture_confidence_threshold)
            grab_err = self.cam.grab(self.runtime)
            if grab_err != sl.ERROR_CODE.SUCCESS:
                raise RuntimeError(f"Live grab failed: {grab_err}")

            self.frame_idx += 1
            state = self.cam.get_position(self.pose, sl.REFERENCE_FRAME.WORLD)
            timestamp_s = float(self.cam.get_timestamp(sl.TIME_REFERENCE.IMAGE).get_milliseconds()) / 1000.0
            if state != sl.POSITIONAL_TRACKING_STATE.OK:
                return {
                    "points_world": np.empty((0, 3), dtype=np.float32),
                    "colors_u8": np.empty((0, 3), dtype=np.uint8),
                    "camera_pos": np.zeros(3, dtype=np.float32),
                    "pose_valid": False,
                    "source_pos": self.frame_idx,
                    "source_len": 0,
                    "source_time_s": timestamp_s,
                    "tracking_state": str(state),
                    "left_bgr": None,
                    "depth_confidence": None,
                    "frame_notes": ["Positional tracking is not OK; image and depth-confidence inputs were skipped."],
                }

            translation = self.pose.get_translation(sl.Translation())
            tx, ty, tz = translation.get()
            orientation = self.pose.get_orientation(sl.Orientation())
            qx, qy, qz, qw = orientation.get()
            self.cam.retrieve_measure(self.pc_mat, sl.MEASURE.XYZRGBA)
            xyz_cam, rgb = zed_xyzrgba_to_xyz_rgb(self.pc_mat, config.live_frame_points, config.max_distance_m)
            left_bgr, frame_notes = retrieve_zed_left_bgr(self.cam, self.image_mat, sl)
            confidence_map, confidence_notes = retrieve_zed_confidence(self.cam, self.confidence_mat, sl)
            frame_notes.extend(confidence_notes)
            if xyz_cam is None:
                xyz_world = np.empty((0, 3), dtype=np.float32)
                rgb = np.empty((0, 3), dtype=np.uint8)
            else:
                rot = quat_to_rot_matrix(qx, qy, qz, qw)
                t = np.array([tx, ty, tz], dtype=np.float32)
                xyz_world = (rot @ xyz_cam.T).T + t
            return {
                "points_world": xyz_world,
                "colors_u8": rgb,
                "camera_pos": np.array([tx, ty, tz], dtype=np.float32),
                "pose_valid": True,
                "source_pos": self.frame_idx,
                "source_len": 0,
                "source_time_s": timestamp_s,
                "tracking_state": "OK",
                "left_bgr": left_bgr,
                "depth_confidence": confidence_map,
                "frame_notes": frame_notes,
            }

        def close(self) -> None:
            if self.cam is not None and self.sl is not None:
                try:
                    self.cam.disable_positional_tracking()
                except Exception:
                    # Some SDK builds raise if tracking was never fully enabled.
                    pass
                self.cam.close()
            self.cam = None


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
                    bundle_paths = self._write_run_bundle(final_config)
                    self._last_stats["last_export_path"] = bundle_paths["map"]
                    self._last_stats["last_traj_path"] = bundle_paths["traj"]
                    self.log.emit(f"Run bundle saved: {self._run_bundle_dir}")
                if final_config.log_run_metrics:
                    summary_path, frames_path = self._write_run_metrics(final_config)
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
                    self._write_ply(export_path)
                    self._last_stats["last_export_path"] = export_path
                    self.log.emit(f"Exported map: {export_path}")
                except Exception as exc:
                    self._last_stats["last_error"] = str(exc)
                    self.log.emit(f"Export failed: {exc}")
            if traj_export_path is not None:
                try:
                    self._write_traj_csv(traj_export_path)
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

        def _write_ply(self, path: str) -> None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8") as f:
                f.write("ply\nformat ascii 1.0\n")
                f.write(f"element vertex {self._map_points.shape[0]}\n")
                f.write("property float x\nproperty float y\nproperty float z\n")
                f.write("property uchar red\nproperty uchar green\nproperty uchar blue\n")
                f.write("end_header\n")
                for p, c in zip(self._map_points, self._map_colors):
                    f.write(f"{p[0]:.6f} {p[1]:.6f} {p[2]:.6f} {int(c[0])} {int(c[1])} {int(c[2])}\n")

        def _write_traj_csv(self, path: str) -> None:
            out = Path(path)
            out.parent.mkdir(parents=True, exist_ok=True)
            with out.open("w", encoding="utf-8", newline="") as f:
                f.write("x,y,z\n")
                for p in self._traj:
                    f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")

        def _write_run_metrics(self, config: WorkerConfig) -> tuple[str, str]:
            metrics_dir = self._run_bundle_dir if config.auto_export_run_bundle else Path(config.run_metrics_dir)
            metrics_dir.mkdir(parents=True, exist_ok=True)
            summary_path = metrics_dir / f"{self._metrics_run_stem}_summary.json"
            frames_path = metrics_dir / f"{self._metrics_run_stem}_frames.csv"
            summary = {
                "source_name": self.source_name,
                "path_hint": self.path_hint,
                "processed_frames": int(self._processed_count),
                "tracking_ok_count": int(self._tracking_ok_count),
                "tracking_lost_count": int(self._tracking_lost_count),
                "final_map_points": int(self._map_points.shape[0]),
                "final_traj_points": int(self._traj.shape[0]),
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
                writer.writerows(self._frame_metrics_rows)
            return str(summary_path), str(frames_path)

        def _write_run_bundle(self, config: WorkerConfig) -> Dict[str, str]:
            self._run_bundle_dir.mkdir(parents=True, exist_ok=True)
            map_path = self._run_bundle_dir / "map.ply"
            traj_path = self._run_bundle_dir / "trajectory.csv"
            self._write_ply(str(map_path))
            self._write_traj_csv(str(traj_path))
            config_path = self._run_bundle_dir / "run_config.json"
            with config_path.open("w", encoding="utf-8") as f:
                json.dump(experiment_settings_dict(config), f, indent=2)
            return {
                "map": str(map_path),
                "traj": str(traj_path),
                "config": str(config_path),
            }

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


class MainWindow(QMainWindow):
    sig_pause = QtCore.Signal(bool)
    sig_stop = QtCore.Signal()
    sig_export = QtCore.Signal(str)
    sig_export_traj = QtCore.Signal(str)
    sig_seek = QtCore.Signal(float)
    sig_reset_map = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZED SVO Mapping GUI")
        self.resize(1600, 980)

        self.config_store = ConfigStore(WorkerConfig())
        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker: Optional[MappingWorker] = None

        self.viewer: Optional[QtInteractor] = None
        self.live_cloud: Optional[pv.PolyData] = None
        self.live_cloud_actor = None
        self.traj_poly: Optional[pv.PolyData] = None
        self.traj_actor = None
        self.cam_poly: Optional[pv.PolyData] = None
        self.cam_actor = None
        self._grid_actor = None

        self._first_data = True
        self._render_failed = False
        self._freeze_live_view = False
        self._suppress_seek_emit = False
        self._fps_times: list[float] = []
        self._completed_map_points = np.empty((0, 3), dtype=np.float32)
        self._completed_map_colors = np.empty((0, 3), dtype=np.uint8)
        self._completed_traj = np.empty((0, 3), dtype=np.float32)
        self._default_svo_path = PROJECT_ROOT / "data" / "svo" / "sofa.svo"

        self._build_ui()
        self._apply_dark_style()
        self._init_log_panel_controls()
        self._apply_backend_availability()

    def _build_ui(self) -> None:
        self._viewport_container = QWidget()
        self._viewport_layout = QVBoxLayout(self._viewport_container)
        self._viewport_layout.setContentsMargins(0, 0, 0, 0)
        self._viewport_placeholder = QLabel(
            "PyVista viewport will initialize when processing starts.\n"
            "This keeps startup lighter on macOS and VTK-heavy environments."
        )
        self._viewport_placeholder.setAlignment(Qt.AlignCenter)
        self._viewport_layout.addWidget(self._viewport_placeholder)
        self.setCentralWidget(self._viewport_container)

        self._build_left_dock()
        self._build_right_dock()
        self._build_bottom_dock()

        self.status_line = QLabel("backend=- frame=0/0 t=0.0s processed=0 live=0 fps=0.0 tracking=OFF uw=OFF")
        self.statusBar().addPermanentWidget(self.status_line, 1)

    def _build_left_dock(self) -> None:
        dock = QDockWidget("Controls", self)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        container = QWidget()
        root = QVBoxLayout(container)

        source_group = QGroupBox("Source / Mode")
        source_form = QFormLayout(source_group)
        self.source_combo = QComboBox()
        self.source_combo.addItems([SOURCE_DEMO, SOURCE_SVO, SOURCE_LIVE_ZED])
        self.source_combo.currentTextChanged.connect(self.on_source_mode_changed)
        model = self.source_combo.model()
        idx_live = self.source_combo.findText(SOURCE_LIVE_ZED)
        if idx_live >= 0:
            item = model.item(idx_live)
            if item is not None:
                item.setEnabled(has_pyzed())

        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText("Optional source path")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.on_browse)
        source_row = QWidget()
        source_row_layout = QHBoxLayout(source_row)
        source_row_layout.setContentsMargins(0, 0, 0, 0)
        source_row_layout.addWidget(self.source_path)
        source_row_layout.addWidget(self.btn_browse)

        self.depth_combo = QComboBox()
        self._build_depth_combo()

        source_form.addRow("Data source", self.source_combo)
        source_form.addRow("Path", source_row)
        source_form.addRow("Depth mode", self.depth_combo)

        run_group = QGroupBox("Run Controls")
        run_layout = QVBoxLayout(run_group)
        self.btn_start = QPushButton("Start")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_export = QPushButton("Export PLY Now")
        self.btn_export_traj = QPushButton("Export Traj CSV")
        self.btn_reset_cam = QPushButton("Reset Camera")
        self.btn_clear_map = QPushButton("Reset Mapping")
        self.btn_analyze_runs = QPushButton("Analyze Runs")

        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(False)
        self.btn_export_traj.setEnabled(False)
        self.btn_clear_map.setEnabled(False)

        self.btn_start.clicked.connect(self.start_processing)
        self.btn_pause.clicked.connect(self.on_pause_resume)
        self.btn_stop.clicked.connect(self.stop_processing)
        self.btn_export.clicked.connect(self.export_now)
        self.btn_export_traj.clicked.connect(self.export_traj_now)
        self.btn_reset_cam.clicked.connect(self.reset_camera)
        self.btn_clear_map.clicked.connect(self.reset_mapping)
        self.btn_analyze_runs.clicked.connect(self.analyze_runs_now)

        for btn in [
            self.btn_start,
            self.btn_pause,
            self.btn_stop,
            self.btn_export,
            self.btn_export_traj,
            self.btn_reset_cam,
            self.btn_clear_map,
            self.btn_analyze_runs,
        ]:
            run_layout.addWidget(btn)

        params_group = QGroupBox("Mapping Parameters")
        params_form = QFormLayout(params_group)
        self.process_every = IntParamWidget(1, 30, 3)
        self.live_frame_points = IntParamWidget(100, 10000, 400)
        self.offline_frame_points = IntParamWidget(500, 30000, 3000)
        self.live_global_cap = IntParamWidget(1000, 300000, 20000)
        self.offline_global_cap = IntParamWidget(10000, 3000000, 300000)
        self.live_update_every = IntParamWidget(1, 200, 20)
        self.max_distance_m = FloatParamWidget(0.5, 30.0, 5.0, 0.1)
        self.point_size = FloatParamWidget(1.0, 10.0, 2.0, 0.1)
        self.live_keep_frames = IntParamWidget(1, 300, 60)
        self.zed_confidence_threshold = IntParamWidget(0, 100, 100)
        self.zed_texture_confidence_threshold = IntParamWidget(0, 100, 100)
        self.depth_minimum_distance_m = FloatParamWidget(0.1, 20.0, 0.4, 0.1)
        self.depth_maximum_distance_m = FloatParamWidget(0.5, 30.0, 5.0, 0.1)
        self.zed_depth_stabilization = IntParamWidget(0, 100, 30)
        self.enable_radius_outlier_filter = QCheckBox("enable")
        self.radius_filter_radius_m = FloatParamWidget(0.01, 1.0, 0.10, 0.01)
        self.radius_filter_min_neighbors = IntParamWidget(1, 64, 5)
        self.enable_voxel_downsampling = QCheckBox("enable")
        self.voxel_size_m = FloatParamWidget(0.005, 0.50, 0.03, 0.005)
        self.log_run_metrics = QCheckBox("enable")
        self.log_run_metrics.setChecked(True)
        self.run_metrics_dir = QLineEdit(str(PROJECT_ROOT / "exports" / "metrics"))
        self.run_label = QLineEdit()
        self.run_label.setPlaceholderText("Optional experiment label, e.g. confidence_zedconf70")
        self.auto_export_run_bundle = QCheckBox("enable")
        self.auto_export_run_bundle.setChecked(True)

        params_form.addRow("process_every", self.process_every)
        params_form.addRow("live_frame_points", self.live_frame_points)
        params_form.addRow("offline_frame_points", self.offline_frame_points)
        params_form.addRow("live_global_cap", self.live_global_cap)
        params_form.addRow("offline_global_cap", self.offline_global_cap)
        params_form.addRow("live_update_every", self.live_update_every)
        params_form.addRow("max_distance_m", self.max_distance_m)
        params_form.addRow("point_size", self.point_size)
        params_form.addRow("live_keep_frames", self.live_keep_frames)
        params_form.addRow("zed_confidence_threshold", self.zed_confidence_threshold)
        params_form.addRow("zed_texture_conf", self.zed_texture_confidence_threshold)
        params_form.addRow("depth_minimum_distance_m", self.depth_minimum_distance_m)
        params_form.addRow("depth_maximum_distance_m", self.depth_maximum_distance_m)
        params_form.addRow("zed_depth_stabilization", self.zed_depth_stabilization)
        params_form.addRow("enable_radius_outlier_filter", self.enable_radius_outlier_filter)
        params_form.addRow("radius_filter_radius_m", self.radius_filter_radius_m)
        params_form.addRow("radius_filter_min_neighbors", self.radius_filter_min_neighbors)
        params_form.addRow("enable_voxel_downsampling", self.enable_voxel_downsampling)
        params_form.addRow("voxel_size_m", self.voxel_size_m)
        params_form.addRow("log_run_metrics", self.log_run_metrics)
        params_form.addRow("run_metrics_dir", self.run_metrics_dir)
        params_form.addRow("run_label", self.run_label)
        params_form.addRow("auto_export_run_bundle", self.auto_export_run_bundle)

        playback_group = QGroupBox("Playback / Sampling")
        playback_form = QFormLayout(playback_group)
        self.playback_speed = QComboBox()
        self.playback_speed.addItems(["0.25x", "0.5x", "1x", "2x", "4x"])
        self.playback_speed.setCurrentText("1x")
        self.live_downsample = QComboBox()
        self.live_downsample.addItems(["RANDOM", "VOXEL", "NONE"])
        self.offline_downsample = QComboBox()
        self.offline_downsample.addItems(["RANDOM", "VOXEL", "NONE"])
        self.live_voxel_size = FloatParamWidget(0.01, 0.50, 0.06, 0.01)
        self.offline_voxel_size = FloatParamWidget(0.01, 0.50, 0.08, 0.01)
        self.color_mode = QComboBox()
        self.color_mode.addItems(["RGB", "HEIGHT", "DISTANCE"])
        self.seek_slider = QtCore.QObject()
        self.seek_widget = IntParamWidget(0, 1000, 0)
        self.seek_widget.valueChanged.connect(self.emit_seek)

        playback_form.addRow("playback_speed", self.playback_speed)
        playback_form.addRow("live_downsample", self.live_downsample)
        playback_form.addRow("offline_downsample", self.offline_downsample)
        playback_form.addRow("live_voxel_size", self.live_voxel_size)
        playback_form.addRow("offline_voxel_size", self.offline_voxel_size)
        playback_form.addRow("color_mode", self.color_mode)
        playback_form.addRow("seek", self.seek_widget)

        underwater_group = QGroupBox("Underwater Enhancement")
        underwater_form = QFormLayout(underwater_group)
        self.uw_enable = QCheckBox("enable")
        self.uw_color = QCheckBox("color")
        self.uw_clahe = QCheckBox("clahe")
        self.uw_denoise = QCheckBox("denoise")
        self.uw_backscatter = QCheckBox("backscatter")
        self.uw_confidence = QCheckBox("confidence hook")
        self.uw_metrics = QCheckBox("metrics")
        self.uw_color.setChecked(True)
        self.uw_clahe.setChecked(True)
        self.uw_denoise.setChecked(True)
        self.uw_backscatter.setChecked(True)
        self.uw_metrics.setChecked(True)
        self.uw_strength = FloatParamWidget(0.5, 2.0, 1.0, 0.1)
        self.uw_clahe_clip = FloatParamWidget(1.0, 4.0, 2.0, 0.1)
        self.uw_conf_threshold = FloatParamWidget(0.0, 100.0, 50.0, 1.0)

        underwater_flags = QWidget()
        underwater_flags_layout = QVBoxLayout(underwater_flags)
        underwater_flags_layout.setContentsMargins(0, 0, 0, 0)
        for widget in [
            self.uw_enable,
            self.uw_color,
            self.uw_clahe,
            self.uw_denoise,
            self.uw_backscatter,
            self.uw_confidence,
            self.uw_metrics,
        ]:
            underwater_flags_layout.addWidget(widget)
        underwater_form.addRow("features", underwater_flags)
        underwater_form.addRow("strength", self.uw_strength)
        underwater_form.addRow("clahe_clip", self.uw_clahe_clip)
        underwater_form.addRow("conf_thresh", self.uw_conf_threshold)

        view_group = QGroupBox("View Toggles")
        view_layout = QVBoxLayout(view_group)
        self.show_grid = QCheckBox("show_grid")
        self.show_axes = QCheckBox("show_axes")
        self.show_trajectory = QCheckBox("show_trajectory")
        self.show_camera_marker = QCheckBox("show_camera_marker")
        self.freeze_live = QCheckBox("freeze_live_view")

        self.show_grid.setChecked(True)
        self.show_axes.setChecked(True)
        self.show_trajectory.setChecked(True)
        self.show_camera_marker.setChecked(True)

        for widget in [
            self.show_grid,
            self.show_axes,
            self.show_trajectory,
            self.show_camera_marker,
            self.freeze_live,
        ]:
            view_layout.addWidget(widget)

        root.addWidget(source_group)
        root.addWidget(run_group)
        root.addWidget(params_group)
        root.addWidget(playback_group)
        root.addWidget(underwater_group)
        root.addWidget(view_group)
        root.addStretch(1)

        scroller = QScrollArea()
        scroller.setWidgetResizable(True)
        scroller.setWidget(container)
        dock.setWidget(scroller)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self._connect_param_updates()
        self._connect_view_toggles()

        if self._default_svo_path.exists():
            self.source_combo.setCurrentText(SOURCE_SVO)
            self.source_path.setText(str(self._default_svo_path))
        else:
            self.source_combo.setCurrentText(SOURCE_DEMO)
        self.on_source_mode_changed(self.source_combo.currentText())

    def _build_right_dock(self) -> None:
        dock = QDockWidget("Metrics / Status", self)
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        panel = QWidget()
        form = QFormLayout(panel)
        self.metric_labels: Dict[str, QLabel] = {}
        keys = [
            "backend",
            "frame_count",
            "processed_count",
            "tracking_state",
            "live_rendered_points",
            "effective_fps",
            "source_pos",
            "source_len",
            "source_time_s",
            "last_export_path",
            "last_traj_path",
            "last_summary_path",
            "last_frames_path",
            "last_error",
            "underwater_status",
            "underwater_metrics",
            "tracking_ok_count",
            "tracking_lost_count",
            "input_points",
            "post_underwater_points",
            "post_radius_points",
            "post_voxel_points",
        ]
        for key in keys:
            label = QLabel("-")
            label.setWordWrap(True)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metric_labels[key] = label
            form.addRow(key, label)
        dock.setWidget(panel)
        self.addDockWidget(Qt.RightDockWidgetArea, dock)

    def _build_bottom_dock(self) -> None:
        dock = QDockWidget("Log Output", self)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        dock.setFloating(False)
        self.log_box = QTextEdit()
        self.log_box.setReadOnly(True)
        dock.setWidget(self.log_box)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.log_dock = dock

    def _init_log_panel_controls(self) -> None:
        self.btn_toggle_log = QPushButton("Log Panel")
        self.btn_toggle_log.setCheckable(True)
        self.btn_toggle_log.setChecked(True)
        self.btn_toggle_log.clicked.connect(self.toggle_log_panel)
        self.statusBar().addPermanentWidget(self.btn_toggle_log)

        toggle_action = self.log_dock.toggleViewAction()
        toggle_action.setShortcut("Ctrl+J")
        self.addAction(toggle_action)
        self.log_dock.visibilityChanged.connect(self.on_log_visibility_changed)

    def _build_depth_combo(self) -> None:
        modes = ["PERFORMANCE", "QUALITY", "ULTRA", "NEURAL"]
        has_neural = False
        if has_pyzed():
            try:
                import pyzed.sl as sl

                has_neural = hasattr(sl.DEPTH_MODE, "NEURAL")
            except Exception:
                has_neural = False
        for mode in modes:
            self.depth_combo.addItem(mode)
        if not has_neural:
            idx = self.depth_combo.findText("NEURAL")
            if idx >= 0:
                item = self.depth_combo.model().item(idx)
                if item is not None:
                    item.setEnabled(False)
                self.depth_combo.setItemText(idx, "NEURAL (unavailable)")

    def _apply_backend_availability(self) -> None:
        if not has_pyzed():
            idx = self.source_combo.findText(SOURCE_LIVE_ZED)
            if idx >= 0:
                item = self.source_combo.model().item(idx)
                if item is not None:
                    item.setEnabled(False)
            self.append_log(pyzed_status_message())

    def _connect_param_updates(self) -> None:
        self.process_every.valueChanged.connect(lambda v: self.config_store.update(process_every=int(v)))
        self.live_frame_points.valueChanged.connect(lambda v: self.config_store.update(live_frame_points=int(v)))
        self.offline_frame_points.valueChanged.connect(lambda v: self.config_store.update(offline_frame_points=int(v)))
        self.live_global_cap.valueChanged.connect(lambda v: self.config_store.update(live_global_cap=int(v)))
        self.offline_global_cap.valueChanged.connect(lambda v: self.config_store.update(offline_global_cap=int(v)))
        self.live_update_every.valueChanged.connect(lambda v: self.config_store.update(live_update_every=int(v)))
        self.max_distance_m.valueChanged.connect(lambda v: self.config_store.update(max_distance_m=float(v)))
        self.point_size.valueChanged.connect(self.on_point_size_changed)
        self.live_keep_frames.valueChanged.connect(lambda v: self.config_store.update(live_keep_frames=int(v)))
        self.zed_confidence_threshold.valueChanged.connect(lambda v: self.config_store.update(zed_confidence_threshold=int(v)))
        self.zed_texture_confidence_threshold.valueChanged.connect(lambda v: self.config_store.update(zed_texture_confidence_threshold=int(v)))
        self.depth_minimum_distance_m.valueChanged.connect(lambda v: self.config_store.update(depth_minimum_distance_m=float(v)))
        self.depth_maximum_distance_m.valueChanged.connect(lambda v: self.config_store.update(depth_maximum_distance_m=float(v)))
        self.zed_depth_stabilization.valueChanged.connect(lambda v: self.config_store.update(zed_depth_stabilization=int(v)))
        self.enable_radius_outlier_filter.toggled.connect(lambda v: self.config_store.update(enable_radius_outlier_filter=bool(v)))
        self.radius_filter_radius_m.valueChanged.connect(lambda v: self.config_store.update(radius_filter_radius_m=float(v)))
        self.radius_filter_min_neighbors.valueChanged.connect(lambda v: self.config_store.update(radius_filter_min_neighbors=int(v)))
        self.enable_voxel_downsampling.toggled.connect(lambda v: self.config_store.update(enable_voxel_downsampling=bool(v)))
        self.voxel_size_m.valueChanged.connect(lambda v: self.config_store.update(voxel_size_m=float(v)))
        self.log_run_metrics.toggled.connect(lambda v: self.config_store.update(log_run_metrics=bool(v)))
        self.run_metrics_dir.textChanged.connect(lambda v: self.config_store.update(run_metrics_dir=str(v).strip()))
        self.run_label.textChanged.connect(lambda v: self.config_store.update(run_label=str(v)))
        self.auto_export_run_bundle.toggled.connect(lambda v: self.config_store.update(auto_export_run_bundle=bool(v)))
        self.playback_speed.currentTextChanged.connect(self.on_playback_speed_changed)
        self.live_downsample.currentTextChanged.connect(lambda v: self.config_store.update(live_downsample_method=str(v)))
        self.offline_downsample.currentTextChanged.connect(lambda v: self.config_store.update(offline_downsample_method=str(v)))
        self.live_voxel_size.valueChanged.connect(lambda v: self.config_store.update(live_voxel_size=float(v)))
        self.offline_voxel_size.valueChanged.connect(lambda v: self.config_store.update(offline_voxel_size=float(v)))
        self.color_mode.currentTextChanged.connect(lambda v: self.config_store.update(color_mode=str(v)))
        self.depth_combo.currentTextChanged.connect(self.on_depth_mode_changed)
        self.freeze_live.toggled.connect(self.on_freeze_live_toggled)
        for widget in [
            self.uw_enable,
            self.uw_color,
            self.uw_clahe,
            self.uw_denoise,
            self.uw_backscatter,
            self.uw_confidence,
            self.uw_metrics,
        ]:
            widget.toggled.connect(self.on_underwater_config_changed)
        self.uw_strength.valueChanged.connect(self.on_underwater_config_changed)
        self.uw_clahe_clip.valueChanged.connect(self.on_underwater_config_changed)
        self.uw_conf_threshold.valueChanged.connect(self.on_underwater_config_changed)

    def _connect_view_toggles(self) -> None:
        self.show_grid.toggled.connect(self.on_show_grid)
        self.show_axes.toggled.connect(self.on_show_axes)
        self.show_trajectory.toggled.connect(self.on_show_trajectory)
        self.show_camera_marker.toggled.connect(self.on_show_camera_marker)

        self.show_grid.toggled.connect(lambda v: self.config_store.update(show_grid=bool(v)))
        self.show_axes.toggled.connect(lambda v: self.config_store.update(show_axes=bool(v)))
        self.show_trajectory.toggled.connect(lambda v: self.config_store.update(show_trajectory=bool(v)))
        self.show_camera_marker.toggled.connect(lambda v: self.config_store.update(show_camera_marker=bool(v)))

    def _current_depth_mode_name(self) -> str:
        text = self.depth_combo.currentText().strip()
        return "NEURAL" if text.startswith("NEURAL") else text

    def _ensure_plotter(self) -> bool:
        if self.viewer is not None:
            return True
        try:
            self.append_log("Initializing PyVista viewport...")
            self.viewer = QtInteractor(self._viewport_container)
            if self._viewport_placeholder is not None:
                self._viewport_layout.removeWidget(self._viewport_placeholder)
                self._viewport_placeholder.deleteLater()
                self._viewport_placeholder = None
            self._viewport_layout.addWidget(self.viewer.interactor)
            try:
                self.viewer.disable_parallel_projection()
            except Exception:
                try:
                    self.viewer.camera.parallel_projection = False
                except Exception:
                    pass
            self.viewer.set_background("#1b1f24")
            self.clear_scene_data()
            self.append_log("PyVista viewport ready.")
            return True
        except Exception as exc:
            QMessageBox.critical(self, "Viewport Error", f"Failed to initialize PyVista viewport:\n{exc}")
            self.append_log(f"Viewport init failed: {exc}")
            return False

    def create_source(self):
        mode = self.source_combo.currentText().strip()
        config = self.config_store.snapshot()
        if mode == SOURCE_SVO:
            if not has_pyzed():
                QMessageBox.warning(self, "ZED SDK Required", pyzed_status_message())
                return None, "", ""
            path = self.source_path.text().strip()
            if not path:
                QMessageBox.warning(self, "Missing SVO", "Please select an SVO file first.")
                return None, "", ""
            if not Path(path).exists():
                QMessageBox.warning(self, "Invalid SVO", f"Path does not exist:\n{path}")
                return None, "", ""
            return ZEDSVOFrameSource(path, config=config), "ZED", path
        if mode == SOURCE_LIVE_ZED:
            if not has_pyzed():
                QMessageBox.warning(self, "ZED SDK Required", pyzed_status_message())
                return None, "", ""
            return ZEDLiveFrameSource(config=config), "ZED LIVE", "live"
        return DemoFrameSource(total_frames=8000, base_fps=30.0), "DEMO", "demo"

    def gather_initial_config(self) -> None:
        self.config_store.update(
            process_every=self.process_every.value(),
            live_frame_points=self.live_frame_points.value(),
            offline_frame_points=self.offline_frame_points.value(),
            live_global_cap=self.live_global_cap.value(),
            offline_global_cap=self.offline_global_cap.value(),
            live_update_every=self.live_update_every.value(),
            max_distance_m=self.max_distance_m.value(),
            point_size=self.point_size.value(),
            show_grid=self.show_grid.isChecked(),
            show_axes=self.show_axes.isChecked(),
            show_trajectory=self.show_trajectory.isChecked(),
            show_camera_marker=self.show_camera_marker.isChecked(),
            depth_mode_name=self._current_depth_mode_name(),
            playback_speed=self._parse_playback_speed(self.playback_speed.currentText()),
            freeze_live_view=self.freeze_live.isChecked(),
            live_downsample_method=self.live_downsample.currentText(),
            offline_downsample_method=self.offline_downsample.currentText(),
            live_voxel_size=self.live_voxel_size.value(),
            offline_voxel_size=self.offline_voxel_size.value(),
            live_keep_frames=self.live_keep_frames.value(),
            color_mode=self.color_mode.currentText(),
            zed_confidence_threshold=self.zed_confidence_threshold.value(),
            zed_texture_confidence_threshold=self.zed_texture_confidence_threshold.value(),
            depth_minimum_distance_m=self.depth_minimum_distance_m.value(),
            depth_maximum_distance_m=self.depth_maximum_distance_m.value(),
            zed_depth_stabilization=self.zed_depth_stabilization.value(),
            enable_radius_outlier_filter=self.enable_radius_outlier_filter.isChecked(),
            radius_filter_radius_m=self.radius_filter_radius_m.value(),
            radius_filter_min_neighbors=self.radius_filter_min_neighbors.value(),
            enable_voxel_downsampling=self.enable_voxel_downsampling.isChecked(),
            voxel_size_m=self.voxel_size_m.value(),
            log_run_metrics=self.log_run_metrics.isChecked(),
            run_metrics_dir=self.run_metrics_dir.text().strip() or str(PROJECT_ROOT / "exports" / "metrics"),
            run_label=self.run_label.text().strip(),
            auto_export_run_bundle=self.auto_export_run_bundle.isChecked(),
            underwater=self.build_underwater_config(),
        )

    def build_underwater_config(self) -> UnderwaterConfig:
        return UnderwaterConfig(
            enabled=self.uw_enable.isChecked(),
            enable_color_correction=self.uw_color.isChecked(),
            enable_clahe=self.uw_clahe.isChecked(),
            enable_denoise=self.uw_denoise.isChecked(),
            enable_backscatter_suppression=self.uw_backscatter.isChecked(),
            enable_depth_confidence_filter=self.uw_confidence.isChecked(),
            compute_quality_metrics=self.uw_metrics.isChecked(),
            color_gain=self.uw_strength.value(),
            clahe_clip_limit=self.uw_clahe_clip.value(),
            confidence_threshold=self.uw_conf_threshold.value(),
        )

    def start_processing(self) -> None:
        if self.worker_thread is not None:
            self.append_log("Worker already running.")
            return
        if not self._ensure_plotter():
            return
        self.gather_initial_config()

        source, source_name, path_hint = self.create_source()
        if source is None:
            return

        self.clear_scene_data()
        self._clear_completed_exports()
        self._fps_times.clear()

        self.worker_thread = QtCore.QThread(self)
        self.worker = MappingWorker(source=source, source_name=source_name, config_store=self.config_store, path_hint=path_hint)
        self.worker.moveToThread(self.worker_thread)

        self.worker_thread.started.connect(self.worker.run)
        self.worker.stats_updated.connect(self.on_stats_updated)
        self.worker.cloud_updated.connect(self.on_cloud_updated)
        self.worker.traj_updated.connect(self.on_traj_updated)
        self.worker.camera_updated.connect(self.on_camera_updated)
        self.worker.log.connect(self.append_log)
        self.worker.finished.connect(self.on_worker_finished)

        self.worker.finished.connect(self.worker_thread.quit)
        self.worker_thread.finished.connect(self.worker_thread.deleteLater)
        self.worker_thread.finished.connect(self.cleanup_worker_refs)

        self.btn_start.setEnabled(False)
        self.btn_pause.setEnabled(True)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(True)
        self.btn_export.setEnabled(True)
        self.btn_export_traj.setEnabled(True)
        self.btn_clear_map.setEnabled(True)

        self.worker_thread.start()
        self.append_log("Start requested.")

    def on_pause_resume(self) -> None:
        if self.worker is None:
            self.append_log("Pause/Resume ignored: worker not running.")
            return
        paused = self.btn_pause.text() == "Pause"
        self.worker.set_paused(paused)
        self.btn_pause.setText("Resume" if paused else "Pause")

    def stop_processing(self) -> None:
        if self.worker is None:
            self.append_log("Stop ignored: worker not running.")
            return
        self.worker.request_stop()
        self.append_log("Stop requested.")

    def reset_mapping(self) -> None:
        if self.worker is not None:
            self.worker.request_reset_mapping()
        self.clear_scene_data()

    def export_now(self) -> None:
        hint = self.source_path.text().strip() or "demo"
        default_path = default_export_path(hint, suffix="map", ext=".ply")
        out_path, _ = QFileDialog.getSaveFileName(self, "Export current point cloud", default_path, "PLY files (*.ply)")
        if not out_path:
            return
        if self.worker is not None:
            self.worker.request_export(out_path)
            self.append_log(f"Queued export: {out_path}")
            return
        if self._completed_map_points.shape[0] == 0:
            self.append_log("Export requested, but no completed map is available.")
            return
        self._write_saved_ply(out_path, self._completed_map_points, self._completed_map_colors)
        self.metric_labels["last_export_path"].setText(out_path)
        self.append_log(f"Exported completed map: {out_path}")

    def export_traj_now(self) -> None:
        hint = self.source_path.text().strip() or "demo"
        default_path = default_export_path(hint, suffix="traj", ext=".csv")
        out_path, _ = QFileDialog.getSaveFileName(self, "Export trajectory CSV", default_path, "CSV files (*.csv)")
        if not out_path:
            return
        if self.worker is not None:
            self.worker.request_export_traj(out_path)
            self.append_log(f"Queued trajectory export: {out_path}")
            return
        if self._completed_traj.shape[0] == 0:
            self.append_log("Trajectory export requested, but no completed trajectory is available.")
            return
        self._write_saved_traj_csv(out_path, self._completed_traj)
        self.append_log(f"Exported completed trajectory: {out_path}")

    def analyze_runs_now(self) -> None:
        runs_dir = PROJECT_ROOT / "exports" / "runs"
        output_dir = PROJECT_ROOT / "exports" / "analysis"
        try:
            result = analyze_runs(runs_dir, output_dir)
        except Exception as exc:
            QMessageBox.critical(self, "Analysis Error", f"Failed to analyze runs:\n{exc}")
            self.append_log(f"Analysis failed: {exc}")
            return

        self.append_log(f"Analysis complete: {result['runs_analyzed']} runs")
        self.append_log(f"Analysis CSV: {result['csv_path']}")
        self.append_log(f"Analysis report: {result['markdown_path']}")
        self.metric_labels["last_summary_path"].setText(str(result["markdown_path"]))
        self.metric_labels["last_frames_path"].setText(str(result["csv_path"]))
        if not result["runs_dir_exists"]:
            QMessageBox.information(
                self,
                "No Runs Yet",
                f"No run folders were found in:\n{runs_dir}\n\nThe analysis outputs were still created.",
            )

    def emit_seek(self, value: int) -> None:
        if self.worker is None or self._suppress_seek_emit:
            return
        self.worker.request_seek(float(value) / 1000.0)

    def reset_camera(self) -> None:
        if self.viewer is None:
            return
        try:
            self.viewer.reset_camera()
            self.viewer.render()
            self.append_log("Viewport reset.")
        except Exception as exc:
            self.append_log(f"Reset camera failed: {exc}")

    def on_browse(self) -> None:
        file_filter = "ZED SVO Files (*.svo *.svo2);;All Files (*)" if self.source_combo.currentText() == SOURCE_SVO else "All Files (*)"
        start_dir = str(self._default_svo_path.parent if self._default_svo_path.exists() else PROJECT_ROOT)
        path, _ = QFileDialog.getOpenFileName(self, "Select Source File", start_dir, file_filter)
        if path:
            self.source_path.setText(path)
            self.append_log(f"Selected source path: {path}")

    def on_source_mode_changed(self, mode: str) -> None:
        is_svo = mode == SOURCE_SVO
        self.source_path.setEnabled(is_svo)
        self.btn_browse.setEnabled(is_svo)
        if is_svo:
            self.source_path.setPlaceholderText("Required .svo or .svo2 path")
            if not self.source_path.text().strip() and self._default_svo_path.exists():
                self.source_path.setText(str(self._default_svo_path))
        elif mode == SOURCE_DEMO:
            self.source_path.setPlaceholderText("Demo mode does not require a path")
        else:
            self.source_path.setPlaceholderText("Live ZED camera does not require a file path")

    def on_playback_speed_changed(self, text: str) -> None:
        self.config_store.update(playback_speed=self._parse_playback_speed(text))

    def on_depth_mode_changed(self, text: str) -> None:
        self.config_store.update(depth_mode_name="NEURAL" if text.startswith("NEURAL") else text)
        if self.worker is not None:
            self.append_log("Depth mode changes apply on next Start.")

    def on_freeze_live_toggled(self, checked: bool) -> None:
        self._freeze_live_view = bool(checked)
        self.config_store.update(freeze_live_view=self._freeze_live_view)

    def on_underwater_config_changed(self, *_args) -> None:
        self.config_store.update(underwater=self.build_underwater_config())
        if self.worker is not None:
            self.append_log("Underwater config updated.")

    def on_point_size_changed(self, value: float) -> None:
        self.config_store.update(point_size=float(value))
        if self.live_cloud_actor is not None:
            try:
                self.live_cloud_actor.GetProperty().SetPointSize(float(value))
                self.viewer.render()
            except Exception:
                pass

    def on_show_grid(self, show: bool) -> None:
        if self.viewer is None:
            return
        try:
            if show:
                if self._grid_actor is None:
                    self._grid_actor = self.viewer.show_grid(color="#3b4252")
                else:
                    self._grid_actor.SetVisibility(1)
            elif self._grid_actor is not None:
                self._grid_actor.SetVisibility(0)
            self.viewer.render()
        except Exception:
            pass

    def on_show_axes(self, show: bool) -> None:
        if self.viewer is None:
            return
        try:
            if show:
                self.viewer.show_axes()
            else:
                self.viewer.hide_axes()
            self.viewer.render()
        except Exception:
            pass

    def on_show_trajectory(self, show: bool) -> None:
        if self.traj_actor is not None:
            try:
                self.traj_actor.SetVisibility(1 if show else 0)
                self.viewer.render()
            except Exception:
                pass

    def on_show_camera_marker(self, show: bool) -> None:
        if self.cam_actor is not None:
            try:
                self.cam_actor.SetVisibility(1 if show else 0)
                self.viewer.render()
            except Exception:
                pass

    def on_stats_updated(self, stats: Dict) -> None:
        self.metric_labels["backend"].setText(str(stats.get("backend", "-")))
        self.metric_labels["frame_count"].setText(str(stats.get("frame_count", "-")))
        self.metric_labels["processed_count"].setText(str(stats.get("processed_count", "-")))
        self.metric_labels["tracking_state"].setText(str(stats.get("tracking_state", "-")))
        self.metric_labels["live_rendered_points"].setText(str(stats.get("live_rendered_points", "-")))
        self.metric_labels["source_pos"].setText(str(stats.get("source_pos", "-")))
        self.metric_labels["source_len"].setText(str(stats.get("source_len", "-")))
        self.metric_labels["source_time_s"].setText(f"{float(stats.get('source_time_s', 0.0)):.2f}")
        self.metric_labels["underwater_status"].setText(str(stats.get("underwater_status", "-")))
        self.metric_labels["underwater_metrics"].setText(str(stats.get("underwater_metrics", "-")))
        if "last_export_path" in stats:
            self.metric_labels["last_export_path"].setText(str(stats["last_export_path"]))
        if "last_traj_path" in stats:
            self.metric_labels["last_traj_path"].setText(str(stats["last_traj_path"]))
        if "last_summary_path" in stats:
            self.metric_labels["last_summary_path"].setText(str(stats["last_summary_path"]))
        if "last_frames_path" in stats:
            self.metric_labels["last_frames_path"].setText(str(stats["last_frames_path"]))
        if "last_error" in stats:
            self.metric_labels["last_error"].setText(str(stats["last_error"]))
        self.metric_labels["tracking_ok_count"].setText(str(stats.get("tracking_ok_count", "-")))
        self.metric_labels["tracking_lost_count"].setText(str(stats.get("tracking_lost_count", "-")))
        self.metric_labels["input_points"].setText(str(stats.get("input_points", "-")))
        self.metric_labels["post_underwater_points"].setText(str(stats.get("post_underwater_points", "-")))
        self.metric_labels["post_radius_points"].setText(str(stats.get("post_radius_points", "-")))
        self.metric_labels["post_voxel_points"].setText(str(stats.get("post_voxel_points", "-")))

        now = time.perf_counter()
        self._fps_times.append(now)
        if len(self._fps_times) > 20:
            self._fps_times = self._fps_times[-20:]
        if len(self._fps_times) >= 2:
            dt_s = self._fps_times[-1] - self._fps_times[0]
            fps = (len(self._fps_times) - 1) / dt_s if dt_s > 0 else 0.0
            self.metric_labels["effective_fps"].setText(f"{fps:.2f}")

        src_len = int(stats.get("source_len", -1) or -1)
        src_pos = int(stats.get("source_pos", -1) or -1)
        if src_len > 0:
            self._suppress_seek_emit = True
            self.seek_widget.setValue(int(np.clip(src_pos / max(1, src_len - 1), 0.0, 1.0) * 1000))
            self._suppress_seek_emit = False

        self.status_line.setText(
            f"backend={stats.get('backend', '-')} "
            f"frame={stats.get('source_pos', 0)}/{stats.get('source_len', 0)} "
            f"t={float(stats.get('source_time_s', 0.0)):.1f}s "
            f"processed={stats.get('processed_count', 0)} "
            f"live={stats.get('live_rendered_points', 0)} "
            f"fps={self.metric_labels['effective_fps'].text()} "
            f"tracking={stats.get('tracking_state', 'OFF')} "
            f"uw={stats.get('underwater_status', 'OFF')}"
        )

    def on_cloud_updated(self, points: np.ndarray, colors_u8: np.ndarray) -> None:
        if self._freeze_live_view or self._render_failed or self.viewer is None:
            return
        try:
            if points.shape[0] == 0:
                if self.live_cloud_actor is not None:
                    self.viewer.remove_actor(self.live_cloud_actor)
                    self.live_cloud_actor = None
                    self.live_cloud = None
                return
            if self.live_cloud is None:
                self.live_cloud = pv.PolyData(points)
                self.live_cloud["colors"] = colors_u8
                self.live_cloud_actor = self.viewer.add_points(
                    self.live_cloud,
                    scalars="colors",
                    rgb=True,
                    point_size=self.point_size.value(),
                    render_points_as_spheres=False,
                    name="live_cloud",
                )
            else:
                self.live_cloud.points = points
                self.live_cloud["colors"] = colors_u8
            if self._first_data:
                self.viewer.reset_camera()
                self._first_data = False
            self.viewer.render()
        except Exception as exc:
            self._render_failed = True
            self.append_log(f"Viewer update failed: {exc}")

    def on_traj_updated(self, positions: np.ndarray) -> None:
        if self._freeze_live_view or self._render_failed or self.viewer is None:
            return
        try:
            if positions.shape[0] < 2:
                if self.traj_actor is not None:
                    self.viewer.remove_actor(self.traj_actor)
                    self.traj_actor = None
                    self.traj_poly = None
                return
            new_lines = pv.lines_from_points(positions)
            if self.traj_poly is None:
                self.traj_poly = new_lines
                self.traj_actor = self.viewer.add_mesh(self.traj_poly, color="#f8d66d", line_width=2, name="trajectory")
            else:
                self.traj_poly.points = new_lines.points
                self.traj_poly.lines = new_lines.lines
            if not self.show_trajectory.isChecked() and self.traj_actor is not None:
                self.traj_actor.SetVisibility(0)
            self.viewer.render()
        except Exception as exc:
            self._render_failed = True
            self.append_log(f"Trajectory update failed: {exc}")

    def on_camera_updated(self, pos: np.ndarray) -> None:
        if self._freeze_live_view or self._render_failed or self.viewer is None or not self.show_camera_marker.isChecked():
            return
        try:
            point = np.asarray(pos, dtype=np.float32).reshape(1, 3)
            if self.cam_poly is None:
                self.cam_poly = pv.PolyData(point)
                self.cam_actor = self.viewer.add_points(
                    self.cam_poly,
                    color="#46d07a",
                    point_size=10.0,
                    render_points_as_spheres=True,
                    name="camera_marker",
                )
            else:
                self.cam_poly.points = point
            self.viewer.render()
        except Exception as exc:
            self._render_failed = True
            self.append_log(f"Camera marker update failed: {exc}")

    def on_worker_finished(self, success: bool, message: str) -> None:
        if self.worker is not None:
            self._completed_map_points, self._completed_map_colors = self.worker.snapshot_map()
            self._completed_traj = self.worker.snapshot_traj()
        self.append_log(message)
        self.btn_start.setEnabled(True)
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(self._completed_map_points.shape[0] > 0)
        self.btn_export_traj.setEnabled(self._completed_traj.shape[0] > 0)
        self.btn_clear_map.setEnabled(True)
        if not success:
            self.metric_labels["last_error"].setText(message)

    def cleanup_worker_refs(self) -> None:
        if self.worker is not None:
            self.worker.deleteLater()
        self.worker = None
        self.worker_thread = None

    def _clear_completed_exports(self) -> None:
        self._completed_map_points = np.empty((0, 3), dtype=np.float32)
        self._completed_map_colors = np.empty((0, 3), dtype=np.uint8)
        self._completed_traj = np.empty((0, 3), dtype=np.float32)
        self.btn_export.setEnabled(False)
        self.btn_export_traj.setEnabled(False)

    @staticmethod
    def _write_saved_ply(path: str, points: np.ndarray, colors_u8: np.ndarray) -> None:
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

    @staticmethod
    def _write_saved_traj_csv(path: str, traj: np.ndarray) -> None:
        out = Path(path)
        out.parent.mkdir(parents=True, exist_ok=True)
        with out.open("w", encoding="utf-8", newline="") as f:
            f.write("x,y,z\n")
            for p in traj:
                f.write(f"{p[0]:.6f},{p[1]:.6f},{p[2]:.6f}\n")

    def clear_scene_data(self) -> None:
        self._first_data = True
        self._render_failed = False
        self.live_cloud = None
        self.live_cloud_actor = None
        self.traj_poly = None
        self.traj_actor = None
        self.cam_poly = None
        self.cam_actor = None
        self._grid_actor = None
        if self.viewer is None:
            return
        try:
            self.viewer.clear()
            self.viewer.set_background("#1b1f24")
            if self.show_axes.isChecked():
                self.viewer.show_axes()
            else:
                self.viewer.hide_axes()
            if self.show_grid.isChecked():
                self._grid_actor = self.viewer.show_grid(color="#3b4252")
            self.viewer.render()
        except Exception:
            pass

    def append_log(self, text: str) -> None:
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        self.log_box.append(f"[{timestamp}] {text}")

    def toggle_log_panel(self) -> None:
        self.log_dock.setVisible(not self.log_dock.isVisible())

    def on_log_visibility_changed(self, visible: bool) -> None:
        self.btn_toggle_log.blockSignals(True)
        self.btn_toggle_log.setChecked(visible)
        self.btn_toggle_log.blockSignals(False)

    @staticmethod
    def _parse_playback_speed(text: str) -> float:
        try:
            return float(text.lower().replace("x", "").strip())
        except Exception:
            return 1.0

    def _apply_dark_style(self) -> None:
        app = QApplication.instance()
        if app is None:
            return
        palette = QPalette()
        palette.setColor(QPalette.Window, QColor("#1f232a"))
        palette.setColor(QPalette.WindowText, QColor("#d8dee9"))
        palette.setColor(QPalette.Base, QColor("#171a20"))
        palette.setColor(QPalette.AlternateBase, QColor("#20242c"))
        palette.setColor(QPalette.Text, QColor("#d8dee9"))
        palette.setColor(QPalette.Button, QColor("#2b303b"))
        palette.setColor(QPalette.ButtonText, QColor("#e5e9f0"))
        palette.setColor(QPalette.Highlight, QColor("#4c7899"))
        palette.setColor(QPalette.HighlightedText, QColor("#ffffff"))
        app.setPalette(palette)
        app.setStyleSheet(
            """
            QMainWindow, QWidget { background-color: #1f232a; color: #d8dee9; }
            QGroupBox { border: 1px solid #3b4252; border-radius: 6px; margin-top: 10px; padding: 8px; font-weight: 600; }
            QGroupBox::title { subcontrol-origin: margin; left: 8px; padding: 0 4px; }
            QPushButton { background-color: #334155; border: 1px solid #4b5563; border-radius: 4px; padding: 6px 8px; }
            QPushButton:hover { background-color: #3d4f66; }
            QPushButton:disabled { color: #6b7280; background-color: #27303d; }
            QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QTextEdit { border: 1px solid #475569; border-radius: 4px; padding: 3px; }
            QDockWidget::title { background: #2a2f38; padding: 6px; text-align: left; }
            """
        )

    def closeEvent(self, event) -> None:
        if self.worker is not None:
            self.worker.request_stop()
        if self.worker_thread is not None:
            self.worker_thread.quit()
            self.worker_thread.wait(5000)
        super().closeEvent(event)


def main() -> int:
    if hasattr(pv, "set_plot_theme"):
        pv.set_plot_theme("document")
    try:
        pv.global_theme.allow_empty_mesh = True
    except Exception:
        pass
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
