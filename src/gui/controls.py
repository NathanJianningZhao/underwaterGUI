from __future__ import annotations

from PySide6.QtWidgets import QDoubleSpinBox, QSpinBox


class IntParamWidget(QSpinBox):
    """Integer spin box with project-standard range initialization."""

    def __init__(self, minimum: int, maximum: int, value: int):
        super().__init__()
        self.setRange(int(minimum), int(maximum))
        self.setValue(int(value))


class FloatParamWidget(QDoubleSpinBox):
    """Float spin box with consistent precision and step handling."""

    def __init__(self, minimum: float, maximum: float, value: float, step: float):
        super().__init__()
        self.setRange(float(minimum), float(maximum))
        self.setSingleStep(float(step))
        self.setDecimals(3)
        self.setValue(float(value))

