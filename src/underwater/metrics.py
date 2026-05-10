from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


@dataclass(frozen=True)
class QualityMetrics:
    brightness_mean: float = 0.0
    contrast_std: float = 0.0
    saturation_mean: float = 0.0
    edge_density: float = 0.0
    red_blue_ratio: float = 1.0
    valid_points: int = 0
    source: str = "none"

    def to_summary(self) -> str:
        return (
            f"src={self.source} "
            f"Y={self.brightness_mean:.1f} "
            f"C={self.contrast_std:.1f} "
            f"S={self.saturation_mean:.1f} "
            f"E={self.edge_density:.3f} "
            f"R/B={self.red_blue_ratio:.2f} "
            f"pts={self.valid_points}"
        )


def _edge_density_from_gray(gray: np.ndarray) -> float:
    if gray.size == 0 or gray.ndim != 2:
        return 0.0
    gy, gx = np.gradient(gray.astype(np.float32))
    magnitude = np.hypot(gx, gy)
    # A per-frame adaptive threshold is enough for relative run comparisons.
    threshold = float(np.mean(magnitude) + np.std(magnitude))
    if threshold <= 0:
        return 0.0
    return float(np.mean(magnitude > threshold))


def compute_quality_metrics(
    image_bgr: Optional[np.ndarray] = None,
    point_colors_u8: Optional[np.ndarray] = None,
    points_world: Optional[np.ndarray] = None,
) -> QualityMetrics:
    if image_bgr is not None and image_bgr.size:
        image = np.asarray(image_bgr, dtype=np.uint8)
        # Images arrive in BGR because OpenCV filters use that convention.
        rgb = image[:, :, ::-1].astype(np.float32)
        gray = 0.114 * image[:, :, 0] + 0.587 * image[:, :, 1] + 0.299 * image[:, :, 2]
        channel_max = rgb.max(axis=2)
        channel_min = rgb.min(axis=2)
        saturation = channel_max - channel_min
        red = np.mean(rgb[:, :, 0]) + 1e-6
        blue = np.mean(rgb[:, :, 2]) + 1e-6
        valid_points = int(points_world.shape[0]) if points_world is not None else 0
        return QualityMetrics(
            brightness_mean=float(np.mean(gray)),
            contrast_std=float(np.std(gray)),
            saturation_mean=float(np.mean(saturation)),
            edge_density=_edge_density_from_gray(gray),
            red_blue_ratio=float(red / blue),
            valid_points=valid_points,
            source="image",
        )

    if point_colors_u8 is not None and point_colors_u8.size:
        colors = np.asarray(point_colors_u8, dtype=np.float32)
        # Point colors are already RGB, so use standard luma weights directly.
        luma = 0.2126 * colors[:, 0] + 0.7152 * colors[:, 1] + 0.0722 * colors[:, 2]
        saturation = colors.max(axis=1) - colors.min(axis=1)
        red = np.mean(colors[:, 0]) + 1e-6
        blue = np.mean(colors[:, 2]) + 1e-6
        valid_points = int(points_world.shape[0]) if points_world is not None else int(colors.shape[0])
        return QualityMetrics(
            brightness_mean=float(np.mean(luma)),
            contrast_std=float(np.std(luma)),
            saturation_mean=float(np.mean(saturation)),
            edge_density=0.0,
            red_blue_ratio=float(red / blue),
            valid_points=valid_points,
            source="point_colors",
        )

    return QualityMetrics(valid_points=int(points_world.shape[0]) if points_world is not None else 0)
