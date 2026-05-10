from __future__ import annotations

from dataclasses import dataclass


@dataclass(frozen=True)
class UnderwaterConfig:
    enabled: bool = False
    enable_color_correction: bool = True
    enable_clahe: bool = True
    enable_denoise: bool = True
    enable_backscatter_suppression: bool = True
    enable_depth_confidence_filter: bool = False
    compute_quality_metrics: bool = True
    color_gain: float = 1.0
    clahe_clip_limit: float = 2.0
    clahe_tile_grid_size: int = 8
    denoise_sigma_color: float = 35.0
    denoise_sigma_space: float = 7.0
    backscatter_percentile: float = 2.5
    backscatter_floor: int = 10
    confidence_threshold: float = 50.0
    point_outlier_zscore: float = 3.5
