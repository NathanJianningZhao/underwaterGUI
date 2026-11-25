from __future__ import annotations

from typing import Callable, Optional

import numpy as np
import pyvista as pv
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QLabel, QMessageBox, QVBoxLayout, QWidget
from pyvistaqt import QtInteractor


class ViewerController:
    """Owns the lazy PyVista viewport and all 3D scene actors."""

    def __init__(self, parent: QWidget):
        self.widget = QWidget(parent)
        self._layout = QVBoxLayout(self.widget)
        self._layout.setContentsMargins(0, 0, 0, 0)
        self.widget.setObjectName("mapViewport")
        self._placeholder: Optional[QWidget] = QWidget(self.widget)
        placeholder_layout = QVBoxLayout(self._placeholder)
        placeholder_layout.setSpacing(6)
        placeholder_layout.addStretch()
        for message, name in (
            ("3D mapping workspace", "viewportTitle"),
            ("Choose an input and select Start mapping to view the point cloud.", "viewportHint"),
        ):
            label = QLabel(message)
            label.setObjectName(name)
            label.setAlignment(Qt.AlignCenter)
            label.setWordWrap(True)
            placeholder_layout.addWidget(label)
        placeholder_layout.addStretch()
        self._layout.addWidget(self._placeholder)

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

    def set_freeze_live_view(self, freeze: bool) -> None:
        self._freeze_live_view = bool(freeze)

    def ensure_plotter(self, append_log: Callable[[str], None]) -> bool:
        if self.viewer is not None:
            return True
        try:
            append_log("Initializing PyVista viewport...")
            self.viewer = QtInteractor(self.widget)
            try:
                self.viewer.disable_parallel_projection()
            except Exception:
                try:
                    self.viewer.camera.parallel_projection = False
                except Exception:
                    pass
            self.viewer.set_background("#0b1620")
            self.clear_scene_data(show_axes=True, show_grid=True)
            self._layout.addWidget(self.viewer.interactor)
            if self._placeholder is not None:
                self._layout.removeWidget(self._placeholder)
                self._placeholder.deleteLater()
                self._placeholder = None
            append_log("PyVista viewport ready.")
            return True
        except Exception as exc:
            self.close()
            QMessageBox.critical(self.widget, "Viewport Error", f"Failed to initialize PyVista viewport:\n{exc}")
            append_log(f"Viewport init failed: {exc}")
            return False

    def close(self) -> None:
        if self.viewer is not None:
            viewer, self.viewer = self.viewer, None
            viewer.close()
            viewer.deleteLater()

    def reset_camera(self, append_log: Callable[[str], None]) -> None:
        if self.viewer is None:
            return
        try:
            self.viewer.reset_camera()
            self.viewer.render()
            append_log("Viewport reset.")
        except Exception as exc:
            append_log(f"Reset camera failed: {exc}")

    def set_point_size(self, value: float) -> None:
        if self.live_cloud_actor is not None and self.viewer is not None:
            try:
                self.live_cloud_actor.GetProperty().SetPointSize(float(value))
                self.viewer.render()
            except Exception:
                pass

    def show_grid(self, show: bool) -> None:
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

    def show_axes(self, show: bool) -> None:
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

    def show_trajectory(self, show: bool) -> None:
        if self.traj_actor is not None and self.viewer is not None:
            try:
                self.traj_actor.SetVisibility(1 if show else 0)
                self.viewer.render()
            except Exception:
                pass

    def show_camera_marker(self, show: bool) -> None:
        if self.cam_actor is not None and self.viewer is not None:
            try:
                self.cam_actor.SetVisibility(1 if show else 0)
                self.viewer.render()
            except Exception:
                pass

    def update_cloud(
        self,
        points: np.ndarray,
        colors_u8: np.ndarray,
        point_size: float,
        append_log: Callable[[str], None],
    ) -> None:
        if self._freeze_live_view or self._render_failed or self.viewer is None:
            return
        try:
            if points.shape[0] == 0:
                if self.live_cloud_actor is not None:
                    self.viewer.remove_actor(self.live_cloud_actor)
                    self.live_cloud_actor = None
                    self.live_cloud = None
                    self.viewer.render()
                return
            if self.live_cloud is None:
                self.live_cloud = pv.PolyData(points)
                self.live_cloud["colors"] = colors_u8
                self.live_cloud_actor = self.viewer.add_points(
                    self.live_cloud,
                    scalars="colors",
                    rgb=True,
                    point_size=point_size,
                    render_points_as_spheres=False,
                    name="live_cloud",
                    copy_mesh=False,
                )
            elif self.live_cloud.n_points != len(points):
                # Rebuild vertex cells as well as coordinates when the point count changes.
                replacement = pv.PolyData(points)
                replacement["colors"] = colors_u8
                self.live_cloud.shallow_copy(replacement)
            else:
                self.live_cloud.points = points
                self.live_cloud["colors"] = colors_u8
            if self._first_data:
                self.viewer.reset_camera()
                self._first_data = False
            self.viewer.render()
        except Exception as exc:
            self._render_failed = True
            append_log(f"Viewer update failed: {exc}")

    def update_trajectory(
        self,
        positions: np.ndarray,
        show_trajectory: bool,
        append_log: Callable[[str], None],
    ) -> None:
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
            self.traj_actor.SetVisibility(int(show_trajectory))
            self.viewer.render()
        except Exception as exc:
            self._render_failed = True
            append_log(f"Trajectory update failed: {exc}")

    def update_camera(
        self,
        pos: np.ndarray,
        show_camera_marker: bool,
        append_log: Callable[[str], None],
    ) -> None:
        if self._freeze_live_view or self._render_failed or self.viewer is None or not show_camera_marker:
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
            append_log(f"Camera marker update failed: {exc}")

    def clear_scene_data(self, show_axes: bool, show_grid: bool) -> None:
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
            self.viewer.set_background("#0b1620")
            if show_axes:
                self.viewer.show_axes()
            else:
                self.viewer.hide_axes()
            if show_grid:
                self._grid_actor = self.viewer.show_grid(color="#3b4252")
            self.viewer.render()
        except Exception:
            pass
