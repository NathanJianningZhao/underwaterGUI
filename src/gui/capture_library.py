"""Local capture presets and shortcuts to successfully written PLY files."""

from __future__ import annotations

import datetime as dt
import json
import math
from pathlib import Path
from typing import Callable

from PySide6.QtCore import QSettings, QSize, Qt, QUrl
from PySide6.QtGui import QDesktopServices
from PySide6.QtWidgets import (
    QCheckBox, QComboBox, QDoubleSpinBox, QGroupBox, QHBoxLayout, QInputDialog,
    QLabel, QListWidget, QListWidgetItem, QMessageBox, QPushButton, QSpinBox, QVBoxLayout,
)

Setting = bool | int | float | str
Control = QCheckBox | QComboBox | QDoubleSpinBox | QSpinBox


def read_json(settings: QSettings, key: str, fallback: list | dict) -> list | dict:
    try:
        data = json.loads(settings.value(key, "", type=str))
    except (TypeError, ValueError):
        return fallback
    return data if isinstance(data, type(fallback)) else fallback


def write_json(settings: QSettings, key: str, data: list | dict) -> bool:
    settings.setValue(key, json.dumps(data, ensure_ascii=False))
    settings.sync()
    return settings.status() == QSettings.Status.NoError


def control_value(control: Control) -> Setting:
    if isinstance(control, QCheckBox):
        return control.isChecked()
    if isinstance(control, QComboBox):
        return control.currentText()
    return control.value()


def restore_control(control: Control, value: Setting) -> bool:
    """Restore compatible values without selecting unavailable camera options."""
    if isinstance(control, QCheckBox):
        if not isinstance(value, bool):
            return False
        control.setChecked(value)
    elif isinstance(control, QComboBox):
        index = control.findText(value) if isinstance(value, str) else -1
        if index < 0 or not control.model().flags(control.model().index(index, 0)) & Qt.ItemIsEnabled:
            return False
        control.setCurrentIndex(index)
    else:
        if type(value) not in (int, float) or not control.minimum() <= value <= control.maximum():
            return False
        if isinstance(control, QSpinBox):
            if value != int(value):
                return False
            control.setValue(int(value))
        elif math.isfinite(value):
            control.setValue(value)
        else:
            return False
    return True


class CapturePresets(QGroupBox):
    def __init__(
        self, settings: QSettings, controls: dict[str, Control], notify: Callable[[str], None]
    ):
        super().__init__("Capture presets")
        self.settings = settings
        self.controls = controls
        self.notify = notify
        self.presets = {
            name: values for name, values in read_json(settings, "capture/presets-v1", {}).items()
            if isinstance(values, dict)
        }
        layout = QVBoxLayout(self)
        self.selection = QComboBox()
        self.selection.setPlaceholderText("Choose a preset")
        self.selection.setMinimumContentsLength(12)
        self.selection.setSizeAdjustPolicy(QComboBox.SizeAdjustPolicy.AdjustToMinimumContentsLengthWithIcon)
        layout.addWidget(self.selection)
        actions = QHBoxLayout()
        self.load_button = QPushButton("Load")
        self.save_button = QPushButton("Save as…")
        actions.addWidget(self.load_button)
        actions.addWidget(self.save_button)
        layout.addLayout(actions)
        self.selection.currentIndexChanged.connect(lambda index: self.load_button.setEnabled(index >= 0))
        self.load_button.clicked.connect(self.load)
        self.save_button.clicked.connect(self.save)
        self._refresh()

    def _refresh(self, selected: str = "") -> None:
        self.selection.clear()
        self.selection.addItems(sorted(self.presets, key=str.casefold))
        self.selection.setCurrentIndex(self.selection.findText(selected))
        self.load_button.setEnabled(self.selection.currentIndex() >= 0)

    def save(self) -> None:
        name, accepted = QInputDialog.getText(self, "Save capture preset", "Preset name")
        if not accepted:
            return
        name = name.strip()
        if not name or len(name) > 60 or not name.isprintable():
            self.notify("Enter a preset name of 1–60 printable characters.")
            return
        existing = next((old for old in self.presets if old.casefold() == name.casefold()), None)
        if existing:
            answer = QMessageBox.question(
                self, "Replace preset?", f"Replace the saved settings for “{existing}”?",
                QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
                QMessageBox.StandardButton.No,
            )
            if answer != QMessageBox.StandardButton.Yes:
                return
            name = existing
        values = self.presets | {name: {key: control_value(control) for key, control in self.controls.items()}}
        if not write_json(self.settings, "capture/presets-v1", values):
            self.notify("Could not save the preset. Check local settings permissions.")
            return
        self.presets = values
        self._refresh(name)
        self.notify(f"Saved preset: {name}")

    def load(self) -> None:
        name = self.selection.currentText()
        if name not in self.presets:
            return
        skipped = [
            key for key, value in self.presets[name].items()
            if key in self.controls and not restore_control(self.controls[key], value)
        ]
        suffix = " · Unavailable settings skipped" if skipped else ""
        self.notify(f"Loaded preset: {name}{suffix}")


class RecentExports(QGroupBox):
    LIMIT = 40

    def __init__(self, settings: QSettings, notify: Callable[[str], None]):
        super().__init__("Recent PLY exports")
        self.settings = settings
        self.notify = notify
        self.records = [
            row for row in read_json(settings, "capture/recent-ply-v1", [])
            if isinstance(row, dict)
            and all(isinstance(row.get(k), str) for k in ("path", "source", "label", "time"))
            and Path(row["path"]).is_absolute()
        ][:self.LIMIT]
        layout = QVBoxLayout(self)
        self.empty = QLabel("Saved PLY files will appear here.")
        self.empty.setObjectName("mutedText")
        self.empty.setWordWrap(True)
        layout.addWidget(self.empty)
        self.entries = QListWidget()
        self.entries.setMinimumHeight(92)
        self.entries.setMaximumHeight(140)
        layout.addWidget(self.entries)
        self.open_button = QPushButton("Open export folder")
        self.open_button.clicked.connect(self.open_folder)
        self.entries.currentItemChanged.connect(lambda item, _: self.open_button.setEnabled(item is not None))
        self.entries.itemActivated.connect(lambda _: self.open_folder())
        layout.addWidget(self.open_button)
        self._refresh()

    def _refresh(self) -> None:
        self.entries.clear()
        for row in self.records:
            path = Path(row["path"])
            title = row["label"] or path.name
            item = QListWidgetItem(f"{title}\n{row['source']} · {row['time']}")
            item.setSizeHint(QSize(0, 45))
            item.setToolTip(str(path))
            item.setData(Qt.UserRole, str(path))
            self.entries.addItem(item)
        self.empty.setVisible(not self.records)
        self.entries.setVisible(bool(self.records))
        self.open_button.setVisible(bool(self.records))
        self.open_button.setEnabled(False)
        if self.records:
            self.entries.setCurrentRow(0)

    def record(self, path: str, source: str, label: str) -> None:
        full_path = str(Path(path).absolute())
        row = {
            "path": full_path, "source": source, "label": label,
            "time": dt.datetime.now().astimezone().strftime("%Y-%m-%d %H:%M"),
        }
        self.records = [row, *(old for old in self.records if old["path"] != full_path)][:self.LIMIT]
        if not write_json(self.settings, "capture/recent-ply-v1", self.records):
            self.notify("PLY saved, but its history could not be saved locally.")
        self._refresh()

    def open_folder(self) -> None:
        item = self.entries.currentItem()
        if item is None:
            return
        path = Path(item.data(Qt.UserRole))
        if not path.is_file():
            self.notify("This PLY was moved or removed.")
            return
        if not QDesktopServices.openUrl(QUrl.fromLocalFile(str(path.parent))):
            self.notify(f"Could not open the export folder: {path.parent}")
