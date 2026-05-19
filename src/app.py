"""Application entry point for the modular ZED underwater mapping GUI."""

from __future__ import annotations

import sys
from pathlib import Path

PROJECT_ROOT = Path(__file__).resolve().parent.parent
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

import pyvista as pv
from PySide6.QtWidgets import QApplication

from src.gui.main_window import MainWindow


def main() -> int:
    """Start the ZED underwater mapping GUI."""
    if hasattr(pv, "set_plot_theme"):
        pv.set_plot_theme("document")
    try:
        pv.global_theme.allow_empty_mesh = True
    except Exception:
        pass
    app = QApplication(sys.argv)
    window = MainWindow()
    window.show()
    return app.exec()


if __name__ == "__main__":
    sys.exit(main())
