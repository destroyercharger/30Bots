"""
AI Trading Tab
Automated trading with AI model signals
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QTableWidget, QTableWidgetItem, QHeaderView,
    QPushButton, QLineEdit, QComboBox
)
from PyQt6.QtCore import Qt, pyqtSignal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.styles import COLORS
from ui.widgets.auto_trade_panel import AutoTradingPanel, TradingStatsWidget


class WatchlistWidget(QFrame):
    """Watchlist management widget."""

    symbol_added = pyqtSignal(str)
    symbol_removed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.symbols = []
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(12)

        # Header
        header = QLabel("Watchlist")
        header.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600; font-size: 14px;")
        layout.addWidget(header)

        # Add symbol row
        add_layout = QHBoxLayout()

        self.symbol_input = QLineEdit()
        self.symbol_input.setPlaceholderText("Add symbol...")
        self.symbol_input.setMaxLength(5)
        self.symbol_input.setToolTip(
            "Add Symbol to Watchlist\n\n"
            "Enter a stock ticker (e.g., AAPL, MSFT, GOOGL) and press Enter.\n"
            "The AI will scan these symbols for trading opportunities."
        )
        self.symbol_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS['text_primary']};
            }}
        """)
        self.symbol_input.returnPressed.connect(self.add_symbol)
        add_layout.addWidget(self.symbol_input)

        add_btn = QPushButton("+")
        add_btn.setFixedWidth(36)
        add_btn.setToolTip("Add symbol to watchlist")
        add_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['profit']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 8px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #00c853;
            }}
        """)
        add_btn.clicked.connect(self.add_symbol)
        add_layout.addWidget(add_btn)

        layout.addLayout(add_layout)

        # Symbols table
        self.table = QTableWidget()
        self.table.setColumnCount(3)
        self.table.setHorizontalHeaderLabels(["Symbol", "Signal", ""])
        self.table.horizontalHeader().setSectionResizeMode(0, QHeaderView.ResizeMode.Stretch)
        self.table.horizontalHeader().setSectionResizeMode(1, QHeaderView.ResizeMode.Fixed)
        self.table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Fixed)
        self.table.setColumnWidth(1, 80)
        self.table.setColumnWidth(2, 40)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                border: none;
                gridline-color: {COLORS['border']};
            }}
            QTableWidget::item {{
                padding: 8px;
                color: {COLORS['text_primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_secondary']};
                padding: 8px;
                border: none;
                font-weight: 600;
            }}
        """)
        layout.addWidget(self.table)

    def add_symbol(self):
        """Add symbol to watchlist."""
        symbol = self.symbol_input.text().upper().strip()
        if symbol and symbol not in self.symbols:
            self.symbols.append(symbol)
            self._add_row(symbol)
            self.symbol_added.emit(symbol)
        self.symbol_input.clear()

    def _add_row(self, symbol: str):
        """Add a row to the table."""
        row = self.table.rowCount()
        self.table.insertRow(row)

        # Symbol
        self.table.setItem(row, 0, QTableWidgetItem(symbol))

        # Signal (placeholder)
        signal_item = QTableWidgetItem("--")
        signal_item.setTextAlignment(Qt.AlignmentFlag.AlignCenter)
        self.table.setItem(row, 1, signal_item)

        # Remove button
        remove_btn = QPushButton("×")
        remove_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['text_muted']};
                border: none;
                font-size: 16px;
            }}
            QPushButton:hover {{
                color: {COLORS['loss']};
            }}
        """)
        remove_btn.clicked.connect(lambda: self.remove_symbol(symbol))
        self.table.setCellWidget(row, 2, remove_btn)

    def remove_symbol(self, symbol: str):
        """Remove symbol from watchlist."""
        if symbol in self.symbols:
            self.symbols.remove(symbol)
            for row in range(self.table.rowCount()):
                if self.table.item(row, 0).text() == symbol:
                    self.table.removeRow(row)
                    break
            self.symbol_removed.emit(symbol)

    def set_symbols(self, symbols: list):
        """Set the entire watchlist."""
        self.table.setRowCount(0)
        self.symbols = []
        for symbol in symbols:
            self.symbols.append(symbol)
            self._add_row(symbol)

    def get_symbols(self) -> list:
        """Get current watchlist."""
        return list(self.symbols)

    def update_signal(self, symbol: str, signal: str, color: str = None):
        """Update signal display for a symbol."""
        for row in range(self.table.rowCount()):
            if self.table.item(row, 0).text() == symbol:
                item = self.table.item(row, 1)
                item.setText(signal)
                if color:
                    item.setForeground(Qt.GlobalColor.green if color == 'green' else Qt.GlobalColor.red)
                break


class RecentTradesWidget(QFrame):
    """Display recent trades."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel("Recent Trades")
        header.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600; font-size: 14px;")
        header.setToolTip(
            "Recent Trades\n\n"
            "Shows the last 20 trades executed by the AI.\n"
            "Includes time, symbol, side (buy/sell), P&L, and the AI model used."
        )
        layout.addWidget(header)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(5)
        self.table.setHorizontalHeaderLabels(["Time", "Symbol", "Side", "P&L", "Model"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                border: none;
                gridline-color: {COLORS['border']};
            }}
            QTableWidget::item {{
                padding: 6px;
                color: {COLORS['text_primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_secondary']};
                padding: 6px;
                border: none;
                font-size: 11px;
            }}
        """)
        layout.addWidget(self.table)

    def add_trade(self, trade: dict):
        """Add a trade to the table."""
        row = 0
        self.table.insertRow(row)

        # Time
        time_str = trade.get('exit_time', '')[:19].replace('T', ' ')[11:]  # HH:MM:SS
        self.table.setItem(row, 0, QTableWidgetItem(time_str))

        # Symbol
        self.table.setItem(row, 1, QTableWidgetItem(trade.get('symbol', '')))

        # Side
        self.table.setItem(row, 2, QTableWidgetItem(trade.get('side', '').upper()))

        # P&L
        pnl = trade.get('pnl', 0)
        pnl_item = QTableWidgetItem(f"${pnl:+.2f}")
        if pnl >= 0:
            pnl_item.setForeground(Qt.GlobalColor.green)
        else:
            pnl_item.setForeground(Qt.GlobalColor.red)
        self.table.setItem(row, 3, pnl_item)

        # Model
        self.table.setItem(row, 4, QTableWidgetItem(trade.get('ai_model', 'Manual')[:12]))

        # Keep only last 20 trades
        while self.table.rowCount() > 20:
            self.table.removeRow(self.table.rowCount() - 1)

    def load_trades(self, trades: list):
        """Load multiple trades."""
        self.table.setRowCount(0)
        for trade in trades:
            self.add_trade(trade)


class ModelPerformanceWidget(QFrame):
    """Display model performance stats."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(8)

        # Header
        header = QLabel("Model Performance (7 days)")
        header.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600; font-size: 14px;")
        header.setToolTip(
            "AI Model Performance\n\n"
            "Shows how each AI model has performed over the last 7 days.\n"
            "Win Rate: Percentage of profitable trades.\n"
            "P&L: Total profit/loss from that model's trades."
        )
        layout.addWidget(header)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Model", "Trades", "Win Rate", "P&L"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setStyleSheet(f"""
            QTableWidget {{
                background-color: {COLORS['bg_card']};
                border: none;
                gridline-color: {COLORS['border']};
            }}
            QTableWidget::item {{
                padding: 6px;
                color: {COLORS['text_primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_secondary']};
                padding: 6px;
                border: none;
                font-size: 11px;
            }}
        """)
        layout.addWidget(self.table)

    def update_performance(self, models: list):
        """Update model performance display."""
        self.table.setRowCount(0)

        for model in models:
            row = self.table.rowCount()
            self.table.insertRow(row)

            # Model name (truncated)
            name = model.get('model', 'Unknown')[:20]
            self.table.setItem(row, 0, QTableWidgetItem(name))

            # Trades
            self.table.setItem(row, 1, QTableWidgetItem(str(model.get('trades', 0))))

            # Win rate
            win_rate = model.get('win_rate', 0)
            wr_item = QTableWidgetItem(f"{win_rate:.1f}%")
            if win_rate >= 60:
                wr_item.setForeground(Qt.GlobalColor.green)
            elif win_rate < 40:
                wr_item.setForeground(Qt.GlobalColor.red)
            self.table.setItem(row, 2, wr_item)

            # P&L
            pnl = model.get('pnl', 0)
            pnl_item = QTableWidgetItem(f"${pnl:+.2f}")
            if pnl >= 0:
                pnl_item.setForeground(Qt.GlobalColor.green)
            else:
                pnl_item.setForeground(Qt.GlobalColor.red)
            self.table.setItem(row, 3, pnl_item)


class AITradingTab(QWidget):
    """AI Trading tab with auto-trading controls."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 16, 16, 16)
        layout.setSpacing(16)

        # Left side - Auto trading panel
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 0, 0)
        left_layout.setSpacing(16)

        self.auto_panel = AutoTradingPanel()
        left_layout.addWidget(self.auto_panel)

        left_widget.setFixedWidth(400)
        layout.addWidget(left_widget)

        # Right side - Watchlist, Recent Trades, Model Performance
        right_widget = QWidget()
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(0, 0, 0, 0)
        right_layout.setSpacing(16)

        # Watchlist
        self.watchlist = WatchlistWidget()
        self.watchlist.setMaximumHeight(300)
        right_layout.addWidget(self.watchlist)

        # Bottom splitter for trades and performance
        bottom_splitter = QSplitter(Qt.Orientation.Horizontal)

        self.recent_trades = RecentTradesWidget()
        bottom_splitter.addWidget(self.recent_trades)

        self.model_performance = ModelPerformanceWidget()
        bottom_splitter.addWidget(self.model_performance)

        right_layout.addWidget(bottom_splitter, stretch=1)

        layout.addWidget(right_widget, stretch=1)

    def load_default_watchlist(self, mode: str = 'day'):
        """Load default watchlist for mode."""
        from core.auto_trader import DEFAULT_DAY_WATCHLIST, DEFAULT_SWING_WATCHLIST
        watchlist = DEFAULT_DAY_WATCHLIST if mode == 'day' else DEFAULT_SWING_WATCHLIST
        self.watchlist.set_symbols(watchlist)
