from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional

import numpy as np

from .config import UnderwaterConfig
from .metrics import QualityMetrics, compute_quality_metrics
from .ops import (
    apply_clahe_bgr,
    cv2_available,
    depth_confidence_filter_hook,
    edge_preserving_denoise_bgr,
    gray_world_white_balance,
    percentile_color_stretch,
    statistical_outlier_filter,
    suppress_backscatter_bgr,
    suppress_backscatter_points,
)


@dataclass
class UnderwaterPipelineResult:
    points_world: np.ndarray
    colors_u8: np.ndarray
    image_bgr: Optional[np.ndarray]
    metrics: QualityMetrics
    status: str
    diagnostics: List[str] = field(default_factory=list)


class UnderwaterPipeline:
    def __init__(self, config: UnderwaterConfig):
        self._config = config

    def update_config(self, config: UnderwaterConfig) -> None:
        self._config = config

    def process_frame(
        self,
        points_world: np.ndarray,
        colors_u8: np.ndarray,
        image_bgr: Optional[np.ndarray] = None,
        depth_confidence: Optional[np.ndarray] = None,
    ) -> UnderwaterPipelineResult:
        cfg = self._config
        diagnostics: List[str] = []
        points = np.asarray(points_world, dtype=np.float32)
        colors = np.asarray(colors_u8, dtype=np.uint8)
        image = None if image_bgr is None else np.asarray(image_bgr, dtype=np.uint8)

        if not cfg.enabled:
            metrics = compute_quality_metrics(image_bgr=image, point_colors_u8=colors, points_world=points)
            return UnderwaterPipelineResult(points, colors, image, metrics, status="OFF", diagnostics=diagnostics)

        # Apply the same enhancement choices to point colors and optional camera
        # images so metrics and map colors tell the same story.
        if cfg.enable_color_correction and colors.size:
            colors = gray_world_white_balance(colors, gain=cfg.color_gain)
            colors = percentile_color_stretch(colors, gain=cfg.color_gain)
            diagnostics.append("color correction applied to point colors")

        if cfg.enable_backscatter_suppression and colors.size:
            colors = suppress_backscatter_points(colors, cfg.backscatter_percentile, cfg.backscatter_floor)
            diagnostics.append("backscatter suppression applied to point colors")

        if points.size:
            points, colors, _ = statistical_outlier_filter(points, colors, cfg.point_outlier_zscore)
            diagnostics.append("point outlier filter applied")

        if cfg.enable_depth_confidence_filter:
            points, colors, _, depth_note = depth_confidence_filter_hook(
                points,
                colors,
                depth_confidence,
                cfg.confidence_threshold,
            )
            diagnostics.append(depth_note)

        # Image filters are optional because demo mode and some SDK builds only
        # provide point-cloud colors.
        if image is not None:
            if cfg.enable_color_correction:
                image = gray_world_white_balance(image, gain=cfg.color_gain)
                image = percentile_color_stretch(image, gain=cfg.color_gain)
                diagnostics.append("color correction applied to image")
            if cfg.enable_clahe:
                image, clahe_note = apply_clahe_bgr(image, cfg.clahe_clip_limit, cfg.clahe_tile_grid_size)
                diagnostics.append(clahe_note)
            if cfg.enable_denoise:
                image, denoise_note = edge_preserving_denoise_bgr(image, cfg.denoise_sigma_color, cfg.denoise_sigma_space)
                diagnostics.append(denoise_note)
            if cfg.enable_backscatter_suppression:
                image = suppress_backscatter_bgr(image, cfg.backscatter_percentile, cfg.backscatter_floor)
                diagnostics.append("backscatter suppression applied to image")

        metrics = (
            compute_quality_metrics(image_bgr=image, point_colors_u8=colors, points_world=points)
            if cfg.compute_quality_metrics
            else QualityMetrics(valid_points=int(points.shape[0]), source="disabled")
        )

        # Keep the status short enough for the GUI status bar and detailed notes
        # in diagnostics, where the log panel has room to wrap.
        status_parts = ["ON"]
        status_parts.append("cv2" if cv2_available() else "no-cv2")
        if cfg.enable_depth_confidence_filter and depth_confidence is None:
            status_parts.append("confidence-hook")
        return UnderwaterPipelineResult(
            points_world=points,
            colors_u8=colors,
            image_bgr=image,
            metrics=metrics,
            status=" ".join(status_parts),
            diagnostics=diagnostics,
        )
