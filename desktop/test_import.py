#!/usr/bin/env python3
"""Test imports for the desktop application"""

import sys
print(f"Python: {sys.version}")

try:
    print("Testing PyQt6...")
    from PyQt6.QtWidgets import QApplication
    print("  OK: PyQt6.QtWidgets")

    print("Testing config...")
    from config import APP_NAME
    print(f"  OK: config (APP_NAME={APP_NAME})")

    print("Testing styles...")
    from ui.styles import DARK_THEME, COLORS
    print(f"  OK: styles ({len(COLORS)} colors defined)")

    print("Testing portfolio_widget...")
    from ui.widgets.portfolio_widget import PortfolioWidget
    print("  OK: portfolio_widget")

    print("Testing positions_table...")
    from ui.widgets.positions_table import PositionsTableWidget
    print("  OK: positions_table")

    print("Testing dashboard_tab...")
    from ui.tabs.dashboard_tab import DashboardTab
    print("  OK: dashboard_tab")

    print("Testing main_window...")
    from ui.main_window import MainWindow
    print("  OK: main_window")

    print("\n" + "=" * 50)
    print("ALL IMPORTS SUCCESSFUL!")
    print("=" * 50)

    # Quick app test
    print("\nCreating QApplication...")
    app = QApplication([])
    print("Creating MainWindow...")
    window = MainWindow()
    print(f"Window title: {window.windowTitle()}")
    print(f"Window size: {window.minimumWidth()}x{window.minimumHeight()}")

    print("\n" + "=" * 50)
    print("APPLICATION READY TO RUN!")
    print("Run: python main.py")
    print("=" * 50)

except Exception as e:
    print(f"\nERROR: {e}")
    import traceback
    traceback.print_exc()
    sys.exit(1)
