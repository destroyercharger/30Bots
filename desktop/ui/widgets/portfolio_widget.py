"""
Portfolio Summary Widget
Displays total portfolio value, P&L, and key metrics
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QFrame, QGridLayout
)
from PyQt6.QtCore import Qt, pyqtSignal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.styles import COLORS, get_profit_color


class MetricCard(QFrame):
    """A single metric display card"""

    def __init__(self, title: str, value: str = "--", parent=None):
        super().__init__(parent)
        self.setup_ui(title, value)

    def setup_ui(self, title: str, value: str):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(16, 12, 16, 12)
        layout.setSpacing(4)

        # Title
        self.title_label = QLabel(title)
        self.title_label.setStyleSheet(f"""
            font-size: 11px;
            font-weight: 500;
            color: {COLORS['text_secondary']};
            text-transform: uppercase;
            letter-spacing: 1px;
        """)
        layout.addWidget(self.title_label)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(self.value_label)

    def set_value(self, value: str, color: str = None):
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"""
                font-size: 24px;
                font-weight: 600;
                color: {color};
            """)


class PortfolioWidget(QWidget):
    """Main portfolio summary widget"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Header
        header = QLabel("Portfolio Summary")
        header.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['text_primary']};
            padding-bottom: 8px;
        """)
        layout.addWidget(header)

        # Metrics grid
        metrics_layout = QGridLayout()
        metrics_layout.setSpacing(12)

        # Total Value
        self.total_value_card = MetricCard("Total Value", "$100,000.00")
        self.total_value_card.setToolTip(
            "Total Portfolio Value\n\n"
            "The combined value of all cash and open positions.\n"
            "This is your total account equity including unrealized gains/losses."
        )
        metrics_layout.addWidget(self.total_value_card, 0, 0)

        # Day P&L
        self.day_pnl_card = MetricCard("Day P&L", "$0.00")
        self.day_pnl_card.setToolTip(
            "Day Profit & Loss\n\n"
            "Your total profit or loss for today.\n"
            "Green = profit, Red = loss.\n"
            "Includes both realized (closed) and unrealized (open) positions."
        )
        metrics_layout.addWidget(self.day_pnl_card, 0, 1)

        # Available Cash
        self.cash_card = MetricCard("Available Cash", "$100,000.00")
        self.cash_card.setToolTip(
            "Available Cash (Buying Power)\n\n"
            "The amount of cash available for new trades.\n"
            "This excludes cash tied up in open positions."
        )
        metrics_layout.addWidget(self.cash_card, 1, 0)

        # Positions
        self.positions_card = MetricCard("Open Positions", "0 / 6")
        self.positions_card.setToolTip(
            "Open Positions\n\n"
            "Current / Maximum positions allowed.\n"
            "When at maximum, the AI won't open new trades until a position closes."
        )
        metrics_layout.addWidget(self.positions_card, 1, 1)

        layout.addLayout(metrics_layout)

        # Additional metrics row
        extra_layout = QHBoxLayout()
        extra_layout.setSpacing(12)

        # Win Rate
        self.winrate_card = MetricCard("Win Rate", "--%")
        self.winrate_card.setToolTip(
            "Win Rate\n\n"
            "Percentage of trades that were profitable.\n"
            "Calculated as: (Winning Trades / Total Trades) x 100\n"
            "A good win rate is 55%+ for day trading."
        )
        extra_layout.addWidget(self.winrate_card)

        # Total Trades
        self.trades_card = MetricCard("Total Trades", "0")
        self.trades_card.setToolTip(
            "Total Trades\n\n"
            "The total number of trades executed.\n"
            "Includes both winning and losing trades."
        )
        extra_layout.addWidget(self.trades_card)

        layout.addLayout(extra_layout)

    def update_portfolio(self, data: dict):
        """Update all portfolio metrics"""
        # Total value
        total_value = data.get('total_value', 0)
        self.total_value_card.set_value(f"${total_value:,.2f}")

        # Day P&L
        day_pnl = data.get('day_pnl', 0)
        pnl_color = get_profit_color(day_pnl)
        sign = "+" if day_pnl >= 0 else ""
        self.day_pnl_card.set_value(f"{sign}${abs(day_pnl):,.2f}", pnl_color)

        # Cash
        cash = data.get('cash', 0)
        self.cash_card.set_value(f"${cash:,.2f}")

        # Positions
        positions = data.get('positions', 0)
        max_positions = data.get('max_positions', 6)
        self.positions_card.set_value(f"{positions} / {max_positions}")

        # Win rate
        win_rate = data.get('win_rate', 0)
        self.winrate_card.set_value(f"{win_rate:.1f}%")

        # Total trades
        total_trades = data.get('total_trades', 0)
        self.trades_card.set_value(str(total_trades))
