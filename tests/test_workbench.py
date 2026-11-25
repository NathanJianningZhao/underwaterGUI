"""Smoke test the workbench with the built-in source on a real Qt display."""

from __future__ import annotations

import os
import time

import numpy as np
import pytest

pytestmark = pytest.mark.skipif(
    os.environ.get("ZED_GUI_TEST") != "1",
    reason="Set ZED_GUI_TEST=1 in a desktop session or on a virtual GL display",
)


def test_demo_workbench_and_export(tmp_path, monkeypatch):
    import pyvista as pv
    from PySide6.QtCore import QSettings
    from PySide6.QtWidgets import QApplication, QFileDialog, QInputDialog

    from src.gui.main_window import MainWindow

    pv.global_theme.allow_empty_mesh = True
    app = QApplication.instance() or QApplication([])
    settings = QSettings(str(tmp_path / "capture.ini"), QSettings.Format.IniFormat)
    window = MainWindow(settings)
    window.source_combo.setCurrentText("Demo")
    window.log_run_metrics.setChecked(False)
    window.auto_export_run_bundle.setChecked(False)
    window.show()

    def until(condition, timeout=10):
        deadline = time.monotonic() + timeout
        while time.monotonic() < deadline:
            app.processEvents()
            if condition():
                return
            time.sleep(0.02)
        pytest.fail(window.log_box.toPlainText())

    def point_count():
        cloud = window.viewer_controller.live_cloud
        return cloud.n_points if cloud is not None else 0

    try:
        assert window.settings_tabs.count() == 4
        window.max_distance_m.setValue(3.5)
        window.uw_enable.setChecked(True)
        monkeypatch.setattr(QInputDialog, "getText", lambda *args: ("Close range", True))
        window.capture_presets.save_button.click()
        second_window = MainWindow(QSettings(settings.fileName(), settings.format()))
        try:
            second_window.capture_presets.selection.setCurrentText("Close range")
            second_window.capture_presets.load_button.click()
            assert second_window.max_distance_m.value() == 3.5
            assert second_window.uw_enable.isChecked()
        finally:
            second_window.close()
        window.uw_enable.setChecked(False)
        window.run_label.setText("Synthetic capture")

        assert window.log_dock.isHidden()
        assert window.details_dock.isHidden()
        next(b for b in window.statusBar().findChildren(type(window.btn_start)) if b.text() == "Details").click()
        assert window.details_dock.isVisible()

        window.btn_start.click()
        until(lambda: point_count() >= 800)
        assert not window.source_combo.isEnabled()
        assert not window.capture_presets.isEnabled()
        window.btn_pause.click()
        assert window.btn_pause.text() == "Resume"
        until(lambda: "Paused." in window.log_box.toPlainText())
        assert window.status_line.text() == "Paused"
        queued_path = tmp_path / "during-run.ply"
        monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(queued_path), ""))
        window.btn_export.click()
        until(lambda: queued_path.is_file() and window.recent_exports.entries.count() == 1)
        assert window.recent_exports.records[0]["path"] == str(queued_path)
        assert window.recent_exports.records[0]["label"] == "Synthetic capture"

        window.btn_pause.click()
        until(lambda: window.status_line.text().startswith("DEMO"))
        assert int(window.metric_labels["live_rendered_points"].text()) > 0

        window.btn_stop.click()
        until(lambda: window.worker_thread is None)
        assert window.btn_start.isEnabled()
        assert window.btn_export.isEnabled()
        assert window.btn_export_traj.isEnabled()
        for button, filename, header in (
            (window.btn_export, "demo.ply", "ply"),
            (window.btn_export_traj, "demo.csv", "x,y,z"),
        ):
            destination = tmp_path / filename
            monkeypatch.setattr(QFileDialog, "getSaveFileName", lambda *args: (str(destination), ""))
            button.click()
            assert destination.read_text().splitlines()[0] == header

        assert window.capture_presets.isEnabled()
        from src.gui.capture_library import RecentExports
        opened = []
        monkeypatch.setattr("src.gui.capture_library.QDesktopServices.openUrl",
                            lambda url: opened.append(url.toLocalFile()) or True)
        history = RecentExports(QSettings(settings.fileName(), settings.format()), window.append_log)
        assert [row["path"] for row in history.records] == [str(tmp_path / "demo.ply"), str(queued_path)]
        history.open_button.click()
        assert opened == [str(tmp_path)]
        (tmp_path / "demo.ply").unlink()
        history.open_button.click()
        assert opened == [str(tmp_path)]
        assert "moved or removed" in window.log_box.toPlainText()

        controller = window.viewer_controller
        for size in (50, 4, 17):
            rng = np.random.default_rng(31)
            points = rng.normal(size=(size, 3)).astype(np.float32)
            colors = np.full((size, 3), 40, dtype=np.uint8)
            controller.update_cloud(points, colors, 2.0, window.append_log)
            assert controller.live_cloud.n_points == size
            assert controller.live_cloud.n_verts == size
            assert controller.live_cloud["colors"].shape == (size, 3)
            assert not controller._render_failed

        window.btn_clear_map.click()
        assert not window.btn_export.isEnabled()
        assert controller.live_cloud is None

        bundle = tmp_path / "run-bundle"
        monkeypatch.setattr("src.mapping.worker.build_run_bundle_dir", lambda *_: bundle)
        window.auto_export_run_bundle.setChecked(True)
        window.btn_start.click()
        until(lambda: point_count() >= 400)
        window.close()
        until(lambda: window.worker_thread is None and not window.isVisible())
        assert (bundle / "map.ply").is_file()
        assert window.recent_exports.records[0]["path"] == str(bundle / "map.ply")
    finally:
        window.close()
        until(lambda: window.worker_thread is None and not window.isVisible())
