"""Synthetic frame source used when the ZED SDK or input data is unavailable."""

from __future__ import annotations

import math
from typing import Any

import numpy as np

SOURCE_DEMO = "Demo"


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

    def read(self, config: Any):
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
