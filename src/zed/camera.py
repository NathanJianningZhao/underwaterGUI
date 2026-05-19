"""ZED SDK frame sources for SVO playback and live camera streaming."""

from __future__ import annotations

import os
import sys
from dataclasses import replace
from pathlib import Path
from typing import Any, List, Optional

import numpy as np

SOURCE_SVO = "SVO Playback"
SOURCE_LIVE_ZED = "Live ZED Camera"


def configure_zed_runtime_paths() -> None:
    """Add optional Python and Windows DLL paths used by the ZED SDK."""
    extra_site_packages = os.environ.get("ZED_MAPPING_EXTRA_SITE_PACKAGES", "").strip()
    if extra_site_packages:
        for entry in extra_site_packages.split(os.pathsep):
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


configure_zed_runtime_paths()


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


class ZEDSVOFrameSource:
    def __init__(self, path: str, config: Any):
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

    def read(self, config: Any):
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
    def __init__(self, config: Any):
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

    def read(self, config: Any):
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
