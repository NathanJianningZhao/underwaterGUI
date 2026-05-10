from __future__ import annotations

from typing import Optional, Tuple

import numpy as np

try:
    import cv2  # type: ignore
except Exception:
    cv2 = None


def cv2_available() -> bool:
    return cv2 is not None


def gray_world_white_balance(data_u8: np.ndarray, gain: float = 1.0) -> np.ndarray:
    # Underwater imagery often drifts blue/green; gray-world balancing gives a
    # cheap correction that works for both image pixels and point-cloud colors.
    data = np.asarray(data_u8, dtype=np.float32)
    if data.size == 0:
        return np.asarray(data_u8, dtype=np.uint8)
    means = np.mean(data.reshape(-1, 3), axis=0)
    mean_gray = float(np.mean(means)) + 1e-6
    scales = np.clip((mean_gray / (means + 1e-6)) * float(gain), 0.5, 2.5)
    balanced = np.clip(data * scales, 0, 255)
    return balanced.astype(np.uint8)


def percentile_color_stretch(data_u8: np.ndarray, gain: float = 1.0) -> np.ndarray:
    data = np.asarray(data_u8, dtype=np.float32)
    if data.size == 0:
        return np.asarray(data_u8, dtype=np.uint8)
    reshaped = data.reshape(-1, 3)
    low = np.percentile(reshaped, 1.0, axis=0)
    high = np.percentile(reshaped, 99.0, axis=0)
    scale = 255.0 / np.maximum(high - low, 1.0)
    stretched = np.clip((data - low) * scale * float(gain), 0, 255)
    return stretched.astype(np.uint8)


def apply_clahe_bgr(image_bgr: np.ndarray, clip_limit: float, tile_grid_size: int) -> Tuple[np.ndarray, str]:
    if cv2 is None:
        return image_bgr, "cv2 unavailable; CLAHE skipped"
    lab = cv2.cvtColor(image_bgr, cv2.COLOR_BGR2LAB)
    l_channel, a_channel, b_channel = cv2.split(lab)
    clahe = cv2.createCLAHE(clipLimit=float(clip_limit), tileGridSize=(int(tile_grid_size), int(tile_grid_size)))
    l_channel = clahe.apply(l_channel)
    merged = cv2.merge((l_channel, a_channel, b_channel))
    return cv2.cvtColor(merged, cv2.COLOR_LAB2BGR), "CLAHE applied"


def edge_preserving_denoise_bgr(image_bgr: np.ndarray, sigma_color: float, sigma_space: float) -> Tuple[np.ndarray, str]:
    if cv2 is None:
        return image_bgr, "cv2 unavailable; denoise skipped"
    denoised = cv2.bilateralFilter(
        image_bgr,
        d=7,
        sigmaColor=float(max(1.0, sigma_color)),
        sigmaSpace=float(max(1.0, sigma_space)),
    )
    return denoised, "edge-preserving denoise applied"


def suppress_backscatter_bgr(image_bgr: np.ndarray, percentile: float, floor: int) -> np.ndarray:
    image = np.asarray(image_bgr, dtype=np.float32)
    if image.size == 0:
        return image_bgr
    per_channel_floor = np.percentile(image.reshape(-1, 3), float(percentile), axis=0)
    per_channel_floor = np.maximum(per_channel_floor, float(floor))
    image = np.clip(image - per_channel_floor + float(floor), 0, 255)
    return image.astype(np.uint8)


def suppress_backscatter_points(colors_u8: np.ndarray, percentile: float, floor: int) -> np.ndarray:
    if colors_u8.size == 0:
        return colors_u8
    colors = np.asarray(colors_u8, dtype=np.float32)
    low = np.percentile(colors, float(percentile), axis=0)
    adjusted = np.clip(colors - np.maximum(low, floor) + floor, 0, 255)
    return adjusted.astype(np.uint8)


def statistical_outlier_filter(
    points_world: np.ndarray,
    colors_u8: np.ndarray,
    zscore_threshold: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    if points_world.shape[0] < 8:
        mask = np.ones(points_world.shape[0], dtype=bool)
        return points_world, colors_u8, mask
    # Median/MAD is stable enough for frame-by-frame filtering without adding
    # a SciPy dependency to the shareable project.
    centroid = np.median(points_world, axis=0)
    dist = np.linalg.norm(points_world - centroid, axis=1)
    median = np.median(dist)
    mad = np.median(np.abs(dist - median)) + 1e-6
    robust_z = 0.6745 * (dist - median) / mad
    mask = np.abs(robust_z) <= float(zscore_threshold)
    return points_world[mask], colors_u8[mask], mask


def radius_outlier_filter(
    points_world: np.ndarray,
    colors_u8: np.ndarray,
    radius_m: float,
    min_neighbors: int,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32)
    colors = np.asarray(colors_u8, dtype=np.uint8)
    if points.shape[0] == 0:
        return points, colors, np.zeros(0, dtype=bool)
    if points.shape[0] == 1:
        mask = np.array([int(min_neighbors) <= 0], dtype=bool)
        return points[mask], colors[mask], mask
    # This O(n^2) pass is guarded by the GUI's point caps and is intended as a
    # simple cleanup filter for exported maps, not a high-density global solver.
    radius_sq = float(max(radius_m, 1e-6)) ** 2
    deltas = points[:, None, :] - points[None, :, :]
    dist_sq = np.sum(deltas * deltas, axis=2)
    neighbor_counts = np.sum(dist_sq <= radius_sq, axis=1) - 1
    mask = neighbor_counts >= int(min_neighbors)
    return points[mask], colors[mask], mask


def voxel_downsample(
    points_world: np.ndarray,
    colors_u8: np.ndarray,
    voxel_size_m: float,
) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    points = np.asarray(points_world, dtype=np.float32)
    colors = np.asarray(colors_u8, dtype=np.uint8)
    if points.shape[0] == 0:
        return points, colors, np.zeros(0, dtype=bool)
    voxel_size = float(max(voxel_size_m, 1e-6))
    voxel_keys = np.floor(points / voxel_size).astype(np.int64)
    _, keep_idx = np.unique(voxel_keys, axis=0, return_index=True)
    keep_idx = np.sort(keep_idx)
    mask = np.zeros(points.shape[0], dtype=bool)
    mask[keep_idx] = True
    return points[mask], colors[mask], mask


def depth_confidence_filter_hook(
    points_world: np.ndarray,
    colors_u8: np.ndarray,
    depth_confidence: Optional[np.ndarray],
    threshold: float,
) -> Tuple[np.ndarray, np.ndarray, Optional[np.ndarray], str]:
    # Confidence maps come from the ZED SDK and may be unavailable or misaligned
    # after point sampling. In those cases the frame is left unchanged.
    if depth_confidence is None:
        return points_world, colors_u8, None, "no depth confidence data available"
    confidence = np.asarray(depth_confidence)
    if confidence.size == 0:
        return points_world, colors_u8, None, "empty depth confidence input"
    flat = confidence.reshape(-1)
    if flat.shape[0] != points_world.shape[0]:
        return points_world, colors_u8, None, "confidence shape does not match filtered points"
    mask = flat >= float(threshold)
    return points_world[mask], colors_u8[mask], mask, "depth confidence filter applied"
