#!/usr/bin/env python3
"""
30Bots Desktop Trading Application
===================================
A professional PyQt6 desktop trading terminal with AI-powered predictions.

Usage:
    python main.py
"""

import sys
import os

# Add project root to path
sys.path.insert(0, os.path.dirname(__file__))

from PyQt6.QtWidgets import QApplication
from PyQt6.QtCore import Qt
from PyQt6.QtGui import QFont

from config import APP_NAME
from ui.main_window import MainWindow


def main():
    # High DPI support
    os.environ["QT_ENABLE_HIGHDPI_SCALING"] = "1"

    # Create application
    app = QApplication(sys.argv)
    app.setApplicationName(APP_NAME)

    # Set default font
    font = QFont("Segoe UI", 10)
    app.setFont(font)

    # Create and show main window
    window = MainWindow()
    window.show()

    # Start event loop
    sys.exit(app.exec())


if __name__ == "__main__":
    main()
