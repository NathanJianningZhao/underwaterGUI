from __future__ import annotations

import importlib

import numpy as np
import pytest


def test_underwater_pipeline_smoke() -> None:
    from src.underwater import UnderwaterConfig, UnderwaterPipeline

    points = np.zeros((4, 3), dtype=np.float32)
    colors = np.zeros((4, 3), dtype=np.uint8)
    result = UnderwaterPipeline(UnderwaterConfig(enabled=True)).process_frame(points, colors)

    assert result.points_world.shape == (4, 3)
    assert result.colors_u8.shape == (4, 3)
    assert result.metrics.valid_points == 4


def test_non_gui_modules_import() -> None:
    modules = [
        "src.mapping.export",
        "src.zed.camera",
        "src.zed.demo_source",
        "src.underwater.config",
        "src.underwater.metrics",
        "src.underwater.ops",
        "src.underwater.pipeline",
    ]
    for module_name in modules:
        assert importlib.import_module(module_name)


def test_app_modules_import_when_gui_deps_exist() -> None:
    pytest.importorskip("PySide6")
    pytest.importorskip("pyvista")
    pytest.importorskip("pyvistaqt")

    app = importlib.import_module("src.app")
    main_window = importlib.import_module("src.gui.main_window")
    worker = importlib.import_module("src.mapping.worker")

    assert callable(app.main)
    assert hasattr(main_window, "MainWindow")
    assert hasattr(worker, "MappingWorker")
