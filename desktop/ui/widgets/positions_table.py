"""
Positions Table Widget
Displays open positions with real-time P&L updates
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QTableWidget,
    QTableWidgetItem, QHeaderView, QPushButton, QMenu, QFrame
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QColor, QAction

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.styles import COLORS, get_profit_color, get_risk_color


class PositionsTableWidget(QWidget):
    """Widget displaying open trading positions"""

    # Signals
    close_position_requested = pyqtSignal(str)  # symbol
    modify_stop_requested = pyqtSignal(str)  # symbol
    view_chart_requested = pyqtSignal(str)  # symbol

    COLUMNS = [
        ("Symbol", 80),
        ("Shares", 70),
        ("Entry", 90),
        ("Current", 90),
        ("P&L $", 100),
        ("P&L %", 80),
        ("Stop Loss", 90),
        ("Target", 90),
        ("Risk", 70),
        ("Model", 120),
    ]

    def __init__(self, parent=None):
        super().__init__(parent)
        self.positions = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Header
        header_layout = QHBoxLayout()

        header = QLabel("Open Positions")
        header.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        header_layout.addWidget(header)

        header_layout.addStretch()

        # Close all button
        close_all_btn = QPushButton("Close All")
        close_all_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['loss']};
                border: 1px solid {COLORS['loss']};
                padding: 6px 12px;
                border-radius: 4px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['loss']};
                color: white;
            }}
        """)
        close_all_btn.clicked.connect(self.on_close_all)
        header_layout.addWidget(close_all_btn)

        layout.addLayout(header_layout)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(len(self.COLUMNS))
        self.table.setHorizontalHeaderLabels([col[0] for col in self.COLUMNS])

        # Set column widths
        for i, (_, width) in enumerate(self.COLUMNS):
            self.table.setColumnWidth(i, width)

        # Table styling
        self.table.setAlternatingRowColors(True)
        self.table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.table.setSelectionMode(QTableWidget.SelectionMode.SingleSelection)
        self.table.setShowGrid(False)
        self.table.verticalHeader().setVisible(False)
        self.table.setContextMenuPolicy(Qt.ContextMenuPolicy.CustomContextMenu)
        self.table.customContextMenuRequested.connect(self.show_context_menu)

        # Header styling
        header = self.table.horizontalHeader()
        header.setStretchLastSection(True)
        header.setSectionResizeMode(QHeaderView.ResizeMode.Fixed)
        header.setSectionResizeMode(len(self.COLUMNS) - 1, QHeaderView.ResizeMode.Stretch)

        layout.addWidget(self.table)

        # Empty state
        self.empty_label = QLabel("No open positions")
        self.empty_label.setStyleSheet(f"""
            color: {COLORS['text_muted']};
            font-size: 14px;
            padding: 40px;
        """)
        self.empty_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(self.empty_label)

        self.update_empty_state()

    def update_empty_state(self):
        """Show/hide empty state based on positions"""
        has_positions = self.table.rowCount() > 0
        self.table.setVisible(has_positions)
        self.empty_label.setVisible(not has_positions)

    def update_positions(self, positions: dict):
        """Update all positions in the table"""
        self.positions = positions
        self.table.setRowCount(len(positions))

        for row, (symbol, pos) in enumerate(positions.items()):
            self.update_position_row(row, symbol, pos)

        self.update_empty_state()

    def update_position_row(self, row: int, symbol: str, pos: dict):
        """Update a single position row"""
        entry_price = pos.get('entry_price', 0)
        current_price = pos.get('current_price', entry_price)
        shares = pos.get('shares', 0)
        stop_loss = pos.get('stop_loss', 0)
        profit_target = pos.get('profit_target', 0)
        risk_score = pos.get('risk_score', 50)
        model_name = pos.get('ai_model_name', 'N/A')

        # Calculate P&L
        pnl_dollars = (current_price - entry_price) * shares
        pnl_pct = ((current_price - entry_price) / entry_price * 100) if entry_price > 0 else 0

        # Symbol
        self.set_cell(row, 0, symbol, bold=True)

        # Shares
        self.set_cell(row, 1, str(shares))

        # Entry price
        self.set_cell(row, 2, f"${entry_price:.2f}")

        # Current price
        self.set_cell(row, 3, f"${current_price:.2f}")

        # P&L $
        pnl_color = get_profit_color(pnl_dollars)
        sign = "+" if pnl_dollars >= 0 else ""
        self.set_cell(row, 4, f"{sign}${abs(pnl_dollars):.2f}", color=pnl_color)

        # P&L %
        self.set_cell(row, 5, f"{sign}{abs(pnl_pct):.2f}%", color=pnl_color)

        # Stop loss
        self.set_cell(row, 6, f"${stop_loss:.2f}")

        # Target
        self.set_cell(row, 7, f"${profit_target:.2f}")

        # Risk score
        risk_color = get_risk_color(risk_score)
        self.set_cell(row, 8, f"{risk_score:.0f}", color=risk_color)

        # Model
        self.set_cell(row, 9, model_name or "N/A")

    def set_cell(self, row: int, col: int, text: str, color: str = None, bold: bool = False):
        """Set a cell value with optional styling"""
        item = QTableWidgetItem(text)
        item.setFlags(item.flags() & ~Qt.ItemFlag.ItemIsEditable)

        if color:
            item.setForeground(QColor(color))

        if bold:
            font = item.font()
            font.setBold(True)
            item.setFont(font)

        # Right-align numeric columns
        if col in [1, 2, 3, 4, 5, 6, 7, 8]:
            item.setTextAlignment(Qt.AlignmentFlag.AlignRight | Qt.AlignmentFlag.AlignVCenter)
        else:
            item.setTextAlignment(Qt.AlignmentFlag.AlignLeft | Qt.AlignmentFlag.AlignVCenter)

        self.table.setItem(row, col, item)

    def show_context_menu(self, pos):
        """Show right-click context menu"""
        row = self.table.rowAt(pos.y())
        if row < 0:
            return

        symbol_item = self.table.item(row, 0)
        if not symbol_item:
            return

        symbol = symbol_item.text()

        menu = QMenu(self)

        # View chart
        chart_action = QAction("View Chart", self)
        chart_action.triggered.connect(lambda: self.view_chart_requested.emit(symbol))
        menu.addAction(chart_action)

        menu.addSeparator()

        # Modify stop loss
        modify_action = QAction("Modify Stop Loss", self)
        modify_action.triggered.connect(lambda: self.modify_stop_requested.emit(symbol))
        menu.addAction(modify_action)

        menu.addSeparator()

        # Close position
        close_action = QAction("Close Position", self)
        close_action.setStyleSheet(f"color: {COLORS['loss']};")
        close_action.triggered.connect(lambda: self.close_position_requested.emit(symbol))
        menu.addAction(close_action)

        menu.exec(self.table.mapToGlobal(pos))

    def on_close_all(self):
        """Handle close all positions button"""
        for symbol in list(self.positions.keys()):
            self.close_position_requested.emit(symbol)

    def update_price(self, symbol: str, price: float):
        """Update the current price for a position"""
        if symbol not in self.positions:
            return

        # Find the row for this symbol
        for row in range(self.table.rowCount()):
            item = self.table.item(row, 0)
            if item and item.text() == symbol:
                self.positions[symbol]['current_price'] = price
                self.update_position_row(row, symbol, self.positions[symbol])
                break
