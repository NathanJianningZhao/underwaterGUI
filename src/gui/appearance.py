"""Shared Qt styling and scrollable settings pages for the mapping workbench."""

from PySide6.QtGui import QColor, QPalette
from PySide6.QtCore import Qt
from PySide6.QtWidgets import QDockWidget, QLabel, QScrollArea, QVBoxLayout, QWidget


def scroll_panel(*widgets: QWidget) -> QScrollArea:
    panel = QWidget()
    layout = QVBoxLayout(panel)
    layout.setContentsMargins(4, 6, 4, 4)
    layout.setSpacing(12)
    for widget in widgets:
        layout.addWidget(widget)
    layout.addStretch()
    scroll = QScrollArea()
    scroll.setFrameShape(QScrollArea.Shape.NoFrame)
    scroll.setWidgetResizable(True)
    scroll.setHorizontalScrollBarPolicy(Qt.ScrollBarPolicy.ScrollBarAlwaysOff)
    scroll.setWidget(panel)
    return scroll


def label_dock(dock: QDockWidget) -> None:
    heading = QLabel(dock.windowTitle())
    heading.setObjectName("dockHeading")
    dock.setTitleBarWidget(heading)


def apply_workbench_style(window: QWidget) -> None:
    palette = window.palette()
    for role, color in (
        (QPalette.Window, "#111d27"),
        (QPalette.WindowText, "#e2edf5"),
        (QPalette.Base, "#0d1822"),
        (QPalette.Text, "#e2edf5"),
        (QPalette.Button, "#1d3040"),
        (QPalette.ButtonText, "#e2edf5"),
        (QPalette.Highlight, "#419387"),
        (QPalette.HighlightedText, "#ffffff"),
    ):
        palette.setColor(role, QColor(color))
    palette.setColor(QPalette.ColorGroup.Disabled, QPalette.Text, QColor("#8193a2"))
    window.setPalette(palette)
    window.setStyleSheet("""
        QMainWindow, QDockWidget, QScrollArea, QScrollArea > QWidget > QWidget {
            background: #111d27;
        }
        QMainWindow::separator { background: #2a4152; width: 2px; height: 2px; }
        QLabel#dockHeading { color: #d7e9f4; background: #172836; padding: 9px 12px; font-weight: 650; }
        QGroupBox {
            color: #c7dce8; font-weight: 600; border: 1px solid #304657;
            border-radius: 8px; margin-top: 12px; padding: 9px 8px 7px;
        }
        QGroupBox::title { subcontrol-origin: margin; left: 9px; padding: 0 5px; }
        QLabel, QCheckBox { color: #e2edf5; }
        QLabel#mutedText, QLabel#runStatus { color: #a7bfce; }
        QPushButton, QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
            border: 1px solid #3a5266; border-radius: 5px; padding: 5px;
            min-height: 16px;
        }
        QPushButton { color: #e2edf5; background: #203546; padding: 6px 10px; }
        QPushButton:hover, QPushButton:checked { background: #304f61; }
        QPushButton#primaryAction {
            color: #092823; background: #80ddcc; border-color: #80ddcc; font-weight: 700;
        }
        QPushButton#primaryAction:hover { background: #a0ede0; }
        QPushButton:disabled, QPushButton#primaryAction:disabled {
            color: #8193a2; background: #192936; border-color: #304657;
        }
        QPushButton:focus, QLineEdit:focus, QComboBox:focus, QSpinBox:focus,
        QDoubleSpinBox:focus { border-color: #80ddcc; }
        QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QPlainTextEdit {
            color: #e2edf5; background: #0d1822;
        }
        QLineEdit:disabled, QComboBox:disabled, QSpinBox:disabled, QDoubleSpinBox:disabled {
            color: #849bab; border-color: #304657;
        }
        QComboBox QAbstractItemView { color: #e2edf5; background: #152431; selection-background-color: #304f61; }
        QScrollBar:vertical { background: #142331; width: 8px; margin: 0px; }
        QScrollBar::handle:vertical { background: #516b7d; min-height: 28px; border-radius: 4px; }
        QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical { height: 0px; }
        QTabWidget::pane { border: 1px solid #304657; border-radius: 6px; }
        QTabBar::tab { color: #bdcfdd; padding: 8px; border-bottom: 2px solid transparent; }
        QTabBar::tab:selected { color: #9be8db; border-bottom-color: #80ddcc; }
        QTabBar::tab:hover { background: #203546; }
        QWidget#mapViewport { background: #0b1620; }
        QLabel#viewportTitle { font-size: 24px; font-weight: 650; }
        QLabel#viewportHint { color: #a7bfce; }
        QStatusBar { background: #0d1822; }
    """)
