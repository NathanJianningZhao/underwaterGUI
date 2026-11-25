"""Main Qt window, docking layout, and signal wiring for the GUI."""

from __future__ import annotations

import datetime as dt
import time
from pathlib import Path
from typing import Dict, Optional

import numpy as np
from PySide6 import QtCore
from PySide6.QtCore import Qt
from PySide6.QtWidgets import (
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
    QPlainTextEdit,
    QSizePolicy,
    QTabWidget,
    QVBoxLayout,
    QWidget,
)

from src.gui.appearance import apply_workbench_style, label_dock, scroll_panel
from src.gui.controls import FloatParamWidget, IntParamWidget
from src.gui.viewer import ViewerController
from src.mapping.export import PROJECT_ROOT, default_export_path, write_ply, write_traj_csv
from src.mapping.worker import ConfigStore, MappingWorker, WorkerConfig
from src.underwater import UnderwaterConfig
from src.zed.camera import (
    SOURCE_LIVE_ZED,
    SOURCE_SVO,
    ZEDLiveFrameSource,
    ZEDSVOFrameSource,
    has_pyzed,
    pyzed_status_message,
)
from src.zed.demo_source import SOURCE_DEMO, DemoFrameSource
from tools.analyze_metrics import analyze_runs


class MainWindow(QMainWindow):
    sig_pause = QtCore.Signal(bool)
    sig_stop = QtCore.Signal()
    sig_export = QtCore.Signal(str)
    sig_export_traj = QtCore.Signal(str)
    sig_seek = QtCore.Signal(float)
    sig_reset_map = QtCore.Signal()

    def __init__(self):
        super().__init__()
        self.setWindowTitle("ZED Mapping Workbench")
        self.resize(1350, 850)
        self._closing_after_run = False

        self.config_store = ConfigStore(WorkerConfig())
        self.worker_thread: Optional[QtCore.QThread] = None
        self.worker: Optional[MappingWorker] = None

        self.viewer_controller = ViewerController(self)
        self._suppress_seek_emit = False
        self._fps_times: list[float] = []
        self._completed_map_points = np.empty((0, 3), dtype=np.float32)
        self._completed_map_colors = np.empty((0, 3), dtype=np.uint8)
        self._completed_traj = np.empty((0, 3), dtype=np.float32)
        self._default_svo_path = PROJECT_ROOT / "data" / "svo" / "sofa.svo"

        self._build_ui()
        apply_workbench_style(self)
        self._init_log_panel_controls()
        self._apply_backend_availability()

    def _build_ui(self) -> None:
        self.setCentralWidget(self.viewer_controller.widget)

        self._build_left_dock()
        self._build_right_dock()
        self._build_bottom_dock()

        self.status_line = QLabel("Ready · Choose a source and start mapping")
        self.status_line.setObjectName("runStatus")
        self.statusBar().addPermanentWidget(self.status_line, 1)

    def _build_left_dock(self) -> None:
        dock = QDockWidget("Workspace", self)
        label_dock(dock)
        dock.setAllowedAreas(Qt.LeftDockWidgetArea)
        container = QWidget()
        root = QVBoxLayout(container)

        container.setMinimumWidth(320)
        root.setContentsMargins(10, 10, 10, 10)
        root.setSpacing(14)

        source_group = QGroupBox("Input")
        source_form = QFormLayout(source_group)
        self.source_combo = QComboBox()
        self.source_combo.addItems([SOURCE_DEMO, SOURCE_SVO, SOURCE_LIVE_ZED])
        self.source_combo.currentTextChanged.connect(self.on_source_mode_changed)

        self.source_path = QLineEdit()
        self.source_path.setPlaceholderText("Optional source path")
        self.btn_browse = QPushButton("Browse")
        self.btn_browse.clicked.connect(self.on_browse)
        source_row = QWidget()
        source_row_layout = QHBoxLayout(source_row)
        source_row_layout.setContentsMargins(0, 0, 0, 0)
        source_row_layout.addWidget(self.source_path)
        source_row_layout.addWidget(self.btn_browse)

        self.source_hint = QLabel()
        self.source_hint.setWordWrap(True)
        self.source_hint.setObjectName("mutedText")
        self.depth_combo = QComboBox()
        self._build_depth_combo()

        source_form.addRow("Data source", self.source_combo)
        source_form.addRow("Path", source_row)
        source_form.addRow("Depth mode", self.depth_combo)
        source_form.addRow(self.source_hint)

        run_group = QGroupBox("Run Controls")
        run_layout = QVBoxLayout(run_group)
        self.btn_start = QPushButton("Start mapping")
        self.btn_start.setObjectName("primaryAction")
        self.btn_pause = QPushButton("Pause")
        self.btn_stop = QPushButton("Stop")
        self.btn_export = QPushButton("Export point cloud…")
        self.btn_export_traj = QPushButton("Export trajectory…")
        self.btn_reset_cam = QPushButton("Reset view")
        self.btn_clear_map = QPushButton("Clear map")
        self.btn_analyze_runs = QPushButton("Analyze run folders")

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

        transport = QHBoxLayout()
        for button in (self.btn_start, self.btn_pause, self.btn_stop):
            transport.addWidget(button, 1 if button is self.btn_start else 0)
        run_layout.addLayout(transport)

        params_group = QGroupBox("Reconstruction")
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

        params_form.addRow("Process every N frames", self.process_every)
        params_form.addRow("Preview points / frame", self.live_frame_points)
        params_form.addRow("Export points / frame", self.offline_frame_points)
        params_form.addRow("Preview point limit", self.live_global_cap)
        params_form.addRow("Export point limit", self.offline_global_cap)
        params_form.addRow("Preview refresh (frames)", self.live_update_every)
        params_form.addRow("Map range (m)", self.max_distance_m)
        params_form.addRow("Display point size", self.point_size)
        params_form.addRow("Preview history (frames)", self.live_keep_frames)
        params_form.addRow("Depth confidence", self.zed_confidence_threshold)
        params_form.addRow("Texture confidence", self.zed_texture_confidence_threshold)
        params_form.addRow("Minimum SDK depth (m)", self.depth_minimum_distance_m)
        params_form.addRow("Maximum SDK depth (m)", self.depth_maximum_distance_m)
        params_form.addRow("Depth stabilization", self.zed_depth_stabilization)
        params_form.addRow("Radius outlier filter", self.enable_radius_outlier_filter)
        params_form.addRow("Neighbor radius (m)", self.radius_filter_radius_m)
        params_form.addRow("Minimum neighbors", self.radius_filter_min_neighbors)
        params_form.addRow("Voxel downsampling", self.enable_voxel_downsampling)
        params_form.addRow("Voxel size (m)", self.voxel_size_m)

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
        self.seek_widget = IntParamWidget(0, 1000, 0)
        self.seek_widget.valueChanged.connect(self.emit_seek)

        playback_form.addRow("Speed", self.playback_speed)
        playback_form.addRow("Preview sampling", self.live_downsample)
        playback_form.addRow("Export sampling", self.offline_downsample)
        playback_form.addRow("Preview voxel (m)", self.live_voxel_size)
        playback_form.addRow("Export voxel (m)", self.offline_voxel_size)
        playback_form.addRow("Color by", self.color_mode)
        playback_form.addRow("Seek (0–1000)", self.seek_widget)

        underwater_group = QGroupBox("Underwater Enhancement")
        underwater_form = QFormLayout(underwater_group)
        self.uw_enable = QCheckBox("Enable underwater processing")
        self.uw_color = QCheckBox("White balance and color stretch")
        self.uw_clahe = QCheckBox("Local contrast (CLAHE)")
        self.uw_denoise = QCheckBox("Bilateral denoising")
        self.uw_denoise.setToolTip("Requires camera imagery; synthetic demo frames do not have it.")
        self.uw_backscatter = QCheckBox("Backscatter suppression")
        self.uw_confidence = QCheckBox("Depth-confidence filter")
        self.uw_metrics = QCheckBox("Record color metrics")
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
        underwater_hint = QLabel("Adjust how the point cloud is colored. Camera-image filters need a ZED image.")
        underwater_hint.setWordWrap(True)
        underwater_hint.setObjectName("mutedText")
        underwater_form.addRow(underwater_hint)
        underwater_form.addRow(underwater_flags)
        underwater_form.addRow("Color strength", self.uw_strength)
        underwater_form.addRow("Contrast limit", self.uw_clahe_clip)
        underwater_form.addRow("Confidence threshold", self.uw_conf_threshold)

        view_group = QGroupBox("View Toggles")
        view_layout = QVBoxLayout(view_group)
        self.show_grid = QCheckBox("Grid")
        self.show_axes = QCheckBox("Orientation axes")
        self.show_trajectory = QCheckBox("Camera trajectory")
        self.show_camera_marker = QCheckBox("Camera marker")
        self.freeze_live = QCheckBox("Freeze preview")

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
        output_group = QGroupBox("Run outputs")
        output_layout = QVBoxLayout(output_group)
        for button in (self.btn_export, self.btn_export_traj, self.btn_analyze_runs):
            output_layout.addWidget(button)

        run_settings = QGroupBox("Experiment files")
        files_form = QFormLayout(run_settings)
        for label, widget in (
            ("Record frame metrics", self.log_run_metrics),
            ("Metrics folder", self.run_metrics_dir),
            ("Run label", self.run_label),
            ("Auto-export run bundle", self.auto_export_run_bundle),
        ):
            files_form.addRow(label, widget)

        self.settings_tabs = QTabWidget()
        self.settings_tabs.addTab(scroll_panel(params_group), "Map")
        self.settings_tabs.addTab(scroll_panel(playback_group), "Playback")
        self.settings_tabs.addTab(scroll_panel(underwater_group, view_group), "Color / view")
        self.settings_tabs.addTab(scroll_panel(output_group, run_settings), "Outputs")
        view_layout.addWidget(self.btn_reset_cam)
        view_layout.addWidget(self.btn_clear_map)
        root.addWidget(self.settings_tabs, 1)
        dock.setWidget(container)
        self.addDockWidget(Qt.LeftDockWidgetArea, dock)

        self._connect_param_updates()
        self._connect_view_toggles()

        if has_pyzed() and self._default_svo_path.is_file():
            self.source_combo.setCurrentText(SOURCE_SVO)
            self.source_path.setText(str(self._default_svo_path))
        else:
            self.source_combo.setCurrentText(SOURCE_DEMO)
        self.on_source_mode_changed(self.source_combo.currentText())

    def _build_right_dock(self) -> None:
        dock = QDockWidget("Run details", self)
        label_dock(dock)
        dock.setMinimumWidth(244)
        dock.setAllowedAreas(Qt.RightDockWidgetArea)
        panel = QWidget()
        form = QFormLayout(panel)
        form.setRowWrapPolicy(QFormLayout.RowWrapPolicy.WrapLongRows)
        form.setFormAlignment(Qt.AlignTop)
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
        display_names = {
            "backend": "Source", "live_rendered_points": "Preview points",
            "effective_fps": "UI updates / sec", "source_pos": "Source position",
            "source_len": "Source frames", "source_time_s": "Source time (s)",
            "last_traj_path": "Last trajectory", "last_frames_path": "Last frame metrics",
            "underwater_metrics": "Color metrics",
        }
        for key in keys:
            label = QLabel("—")
            label.setTextFormat(Qt.PlainText)
            label.setWordWrap(True)
            label.setSizePolicy(QSizePolicy.Policy.Ignored, QSizePolicy.Policy.Preferred)
            label.setTextInteractionFlags(Qt.TextSelectableByMouse)
            self.metric_labels[key] = label
            form.addRow(display_names.get(key, key.replace("_", " ").capitalize()), label)
        dock.setWidget(scroll_panel(panel))
        self.addDockWidget(Qt.RightDockWidgetArea, dock)
        self.details_dock = dock
        dock.hide()

    def _build_bottom_dock(self) -> None:
        dock = QDockWidget("Log", self)
        label_dock(dock)
        dock.setAllowedAreas(Qt.BottomDockWidgetArea)
        dock.setFeatures(QDockWidget.DockWidgetClosable | QDockWidget.DockWidgetMovable)
        dock.setFloating(False)
        self.log_box = QPlainTextEdit()
        self.log_box.setMaximumBlockCount(2000)
        self.log_box.setReadOnly(True)
        dock.setWidget(self.log_box)
        self.addDockWidget(Qt.BottomDockWidgetArea, dock)
        self.log_dock = dock
        dock.hide()

    def _init_log_panel_controls(self) -> None:
        for dock, label, shortcut in (
            (self.details_dock, "Details", "Ctrl+D"),
            (self.log_dock, "Log", "Ctrl+J"),
        ):
            button = QPushButton(label)
            button.setCheckable(True)
            button.setChecked(not dock.isHidden())
            button.clicked.connect(lambda checked, panel=dock: panel.setVisible(checked))
            dock.visibilityChanged.connect(lambda visible, control=button: control.setChecked(visible))
            self.statusBar().addPermanentWidget(button)
            action = dock.toggleViewAction()
            action.setShortcut(shortcut)
            self.addAction(action)

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
            for mode in (SOURCE_SVO, SOURCE_LIVE_ZED):
                item = self.source_combo.model().item(self.source_combo.findText(mode))
                item.setEnabled(False)
                item.setToolTip("Requires the Stereolabs ZED SDK and Python bindings.")
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
        return self.viewer_controller.ensure_plotter(self.append_log)

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
            if not Path(path).expanduser().is_file() or Path(path).suffix.lower() not in {".svo", ".svo2"}:
                QMessageBox.warning(self, "Invalid SVO", f"Choose an existing .svo or .svo2 file:\n{path}")
                return None, "", ""
            path = str(Path(path).expanduser())
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
        self.gather_initial_config()
        source, source_name, path_hint = self.create_source()
        if source is None or not self._ensure_plotter():
            return

        self.clear_scene_data()
        self._clear_completed_exports()
        self._fps_times.clear()
        for label in self.metric_labels.values():
            label.setText("—")
        self.status_line.setText(f"{source_name} · Starting…")
        self.source_combo.setEnabled(False)
        self._sync_source_controls()

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
        self.worker_thread.finished.connect(self.worker.deleteLater)
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
        self.status_line.setText("Paused" if paused else "Resuming…")

    def stop_processing(self) -> None:
        if self.worker is None:
            self.append_log("Stop ignored: worker not running.")
            return
        self.worker.request_stop()
        self.btn_pause.setEnabled(False)
        self.btn_stop.setEnabled(False)
        self.status_line.setText("Stopping…")
        self.append_log("Stop requested.")

    def reset_mapping(self) -> None:
        if self.worker is not None:
            self.worker.request_reset_mapping()
        self.clear_scene_data()
        if self.worker is None:
            self._clear_completed_exports()
            self.btn_clear_map.setEnabled(False)

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
        write_ply(out_path, self._completed_map_points, self._completed_map_colors)
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
        write_traj_csv(out_path, self._completed_traj)
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
        self.viewer_controller.reset_camera(self.append_log)

    def on_browse(self) -> None:
        file_filter = "ZED SVO Files (*.svo *.svo2);;All Files (*)" if self.source_combo.currentText() == SOURCE_SVO else "All Files (*)"
        start_dir = str(self._default_svo_path.parent if self._default_svo_path.exists() else PROJECT_ROOT)
        path, _ = QFileDialog.getOpenFileName(self, "Select Source File", start_dir, file_filter)
        if path:
            self.source_path.setText(path)
            self.append_log(f"Selected source path: {path}")

    def on_source_mode_changed(self, mode: str) -> None:
        is_svo = mode == SOURCE_SVO
        self._sync_source_controls()
        if is_svo:
            self.source_path.setPlaceholderText("Required .svo or .svo2 path")
            if not self.source_path.text().strip() and self._default_svo_path.exists():
                self.source_path.setText(str(self._default_svo_path))
            self.source_hint.setText("Replay a recorded ZED session.")
        elif mode == SOURCE_DEMO:
            self.source_path.setPlaceholderText("No file needed")
            self.source_hint.setText("Synthetic scene · no camera or SDK needed")
        else:
            self.source_path.setPlaceholderText("No file needed")
            self.source_hint.setText("Stream from a connected ZED camera.")

    def _sync_source_controls(self) -> None:
        mode = self.source_combo.currentText()
        idle = self.source_combo.isEnabled()
        self.source_path.setEnabled(idle and mode == SOURCE_SVO)
        self.btn_browse.setEnabled(idle and mode == SOURCE_SVO)
        self.depth_combo.setEnabled(mode != SOURCE_DEMO)

    def on_playback_speed_changed(self, text: str) -> None:
        self.config_store.update(playback_speed=self._parse_playback_speed(text))

    def on_depth_mode_changed(self, text: str) -> None:
        self.config_store.update(depth_mode_name="NEURAL" if text.startswith("NEURAL") else text)
        if self.worker is not None:
            self.append_log("Depth mode changes apply on next Start.")

    def on_freeze_live_toggled(self, checked: bool) -> None:
        freeze = bool(checked)
        self.viewer_controller.set_freeze_live_view(freeze)
        self.config_store.update(freeze_live_view=freeze)

    def on_underwater_config_changed(self, *_args) -> None:
        self.config_store.update(underwater=self.build_underwater_config())
        if self.worker is not None:
            self.append_log("Underwater config updated.")

    def on_point_size_changed(self, value: float) -> None:
        self.config_store.update(point_size=float(value))
        self.viewer_controller.set_point_size(float(value))

    def on_show_grid(self, show: bool) -> None:
        self.viewer_controller.show_grid(show)

    def on_show_axes(self, show: bool) -> None:
        self.viewer_controller.show_axes(show)

    def on_show_trajectory(self, show: bool) -> None:
        self.viewer_controller.show_trajectory(show)

    def on_show_camera_marker(self, show: bool) -> None:
        self.viewer_controller.show_camera_marker(show)

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

        if self.btn_stop.isEnabled() and self.btn_pause.text() != "Resume":
            self.status_line.setText(
                f"{stats.get('backend', '—')}  ·  "
                f"{float(stats.get('source_time_s', 0.0)):.1f} s  ·  "
                f"{stats.get('live_rendered_points', 0):,} preview points  ·  "
                f"Tracking: {stats.get('tracking_state', '—')}"
            )

    def on_cloud_updated(self, points: np.ndarray, colors_u8: np.ndarray) -> None:
        self.viewer_controller.update_cloud(points, colors_u8, self.point_size.value(), self.append_log)

    def on_traj_updated(self, positions: np.ndarray) -> None:
        self.viewer_controller.update_trajectory(positions, self.show_trajectory.isChecked(), self.append_log)

    def on_camera_updated(self, pos: np.ndarray) -> None:
        self.viewer_controller.update_camera(pos, self.show_camera_marker.isChecked(), self.append_log)

    def on_worker_finished(self, success: bool, message: str) -> None:
        if self.worker is not None:
            self._completed_map_points, self._completed_map_colors = self.worker.snapshot_map()
            self._completed_traj = self.worker.snapshot_traj()
        self.append_log(message)
        self.status_line.setText("Run finished" if success else "Run failed · See details or log")
        self.btn_pause.setEnabled(False)
        self.btn_pause.setText("Pause")
        self.btn_stop.setEnabled(False)
        self.btn_export.setEnabled(self._completed_map_points.shape[0] > 0)
        self.btn_export_traj.setEnabled(self._completed_traj.shape[0] > 0)
        self.btn_clear_map.setEnabled(True)
        if not success:
            self.metric_labels["last_error"].setText(message)

    def cleanup_worker_refs(self) -> None:
        self.worker = None
        self.worker_thread = None
        self.btn_start.setEnabled(True)
        self.source_combo.setEnabled(True)
        self._sync_source_controls()
        if self._closing_after_run:
            QtCore.QTimer.singleShot(0, self.close)

    def _clear_completed_exports(self) -> None:
        self._completed_map_points = np.empty((0, 3), dtype=np.float32)
        self._completed_map_colors = np.empty((0, 3), dtype=np.uint8)
        self._completed_traj = np.empty((0, 3), dtype=np.float32)
        self.btn_export.setEnabled(False)
        self.btn_export_traj.setEnabled(False)

    def clear_scene_data(self) -> None:
        self.viewer_controller.clear_scene_data(
            show_axes=self.show_axes.isChecked(),
            show_grid=self.show_grid.isChecked(),
        )

    def append_log(self, text: str) -> None:
        timestamp = dt.datetime.now().strftime("%H:%M:%S")
        self.log_box.appendPlainText(f"[{timestamp}] {text}")

    @staticmethod
    def _parse_playback_speed(text: str) -> float:
        try:
            return float(text.lower().replace("x", "").strip())
        except Exception:
            return 1.0

    def closeEvent(self, event) -> None:
        if self.worker_thread is not None:
            event.ignore()
            if not self._closing_after_run:
                self._closing_after_run = True
                self.stop_processing()
            return
        self.viewer_controller.close()
        super().closeEvent(event)
