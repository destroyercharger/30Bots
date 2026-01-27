"""
Analytics Tab
Trading performance analytics with charts and visualizations
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QPushButton, QScrollArea
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen

import sys
from pathlib import Path
from datetime import datetime, timedelta
from typing import List, Dict, Optional
import random

# Add parent for imports
PARENT_DIR = Path(__file__).parent.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from ui.styles import COLORS, get_profit_color


class MetricCard(QFrame):
    """Small card displaying a single metric."""

    def __init__(self, title: str, value: str = "-", subtitle: str = "", parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 10px;
            }}
        """)
        self.setup_ui(title, value, subtitle)

    def setup_ui(self, title: str, value: str, subtitle: str):
        layout = QVBoxLayout(self)
        layout.setSpacing(5)
        layout.setContentsMargins(15, 12, 15, 12)

        # Title
        title_label = QLabel(title)
        title_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 11px;")
        layout.addWidget(title_label)

        # Value
        self.value_label = QLabel(value)
        self.value_label.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        self.value_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(self.value_label)

        # Subtitle
        self.subtitle_label = QLabel(subtitle)
        self.subtitle_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
        layout.addWidget(self.subtitle_label)

    def set_value(self, value: str, color: str = None):
        self.value_label.setText(value)
        if color:
            self.value_label.setStyleSheet(f"color: {color};")

    def set_subtitle(self, subtitle: str):
        self.subtitle_label.setText(subtitle)


class EquityCurveWidget(QFrame):
    """Simple equity curve visualization using paint events."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        self.data_points: List[float] = []
        self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample equity curve data."""
        # Start at 100k, random walk with slight upward bias
        value = 100000
        self.data_points = [value]
        for _ in range(60):
            change = random.uniform(-0.01, 0.012) * value
            value += change
            self.data_points.append(value)

    def set_data(self, data: List[float]):
        self.data_points = data
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if len(self.data_points) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        # Calculate dimensions
        margin = 50
        width = self.width() - 2 * margin
        height = self.height() - 2 * margin

        min_val = min(self.data_points)
        max_val = max(self.data_points)
        val_range = max_val - min_val or 1

        # Draw title
        painter.setPen(QColor(COLORS['text_primary']))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(margin, 25, "Equity Curve")

        # Draw grid lines
        painter.setPen(QPen(QColor(COLORS['border']), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = margin + (height * i / 4)
            painter.drawLine(margin, int(y), self.width() - margin, int(y))

        # Draw Y-axis labels
        painter.setPen(QColor(COLORS['text_muted']))
        painter.setFont(QFont("Segoe UI", 9))
        for i in range(5):
            y = margin + (height * i / 4)
            val = max_val - (val_range * i / 4)
            painter.drawText(5, int(y + 4), f"${val/1000:.0f}K")

        # Draw the curve
        start_val = self.data_points[0]
        end_val = self.data_points[-1]
        color = QColor(COLORS['profit']) if end_val >= start_val else QColor(COLORS['loss'])
        painter.setPen(QPen(color, 2))

        points = []
        for i, val in enumerate(self.data_points):
            x = margin + (width * i / (len(self.data_points) - 1))
            y = margin + height - (height * (val - min_val) / val_range)
            points.append((int(x), int(y)))

        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

        # Fill under curve
        fill_color = QColor(color)
        fill_color.setAlpha(30)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.PenStyle.NoPen)

        from PyQt6.QtGui import QPolygon
        from PyQt6.QtCore import QPoint
        polygon = QPolygon()
        polygon.append(QPoint(points[0][0], margin + height))
        for x, y in points:
            polygon.append(QPoint(x, y))
        polygon.append(QPoint(points[-1][0], margin + height))
        painter.drawPolygon(polygon)

        painter.end()


class ModelPerformanceWidget(QFrame):
    """Bar chart showing model win rates."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(200)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        self.model_data: List[Dict] = []
        self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample model performance data."""
        self.model_data = [
            {'name': 'Momentum_Selective', 'win_rate': 94.8, 'trades': 15},
            {'name': 'MeanReversion_Sel', 'win_rate': 94.6, 'trades': 12},
            {'name': 'MeanReversion_Mod', 'win_rate': 90.9, 'trades': 8},
            {'name': 'Breakout_Moderate', 'win_rate': 85.2, 'trades': 10},
            {'name': 'Trend_Following', 'win_rate': 78.5, 'trades': 6},
        ]

    def set_data(self, data: List[Dict]):
        self.model_data = data[:5]  # Top 5 models
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if not self.model_data:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 20
        bar_height = 25
        spacing = 10
        max_bar_width = self.width() - margin - 180

        # Title
        painter.setPen(QColor(COLORS['text_primary']))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(margin, 25, "Model Performance (Win Rate)")

        y = 50
        for model in self.model_data:
            # Model name
            painter.setPen(QColor(COLORS['text_secondary']))
            painter.setFont(QFont("Segoe UI", 10))
            painter.drawText(margin, y + 17, model['name'][:18])

            # Bar background
            bar_x = 160
            painter.setBrush(QBrush(QColor(COLORS['bg_light'])))
            painter.setPen(Qt.PenStyle.NoPen)
            painter.drawRoundedRect(bar_x, y, max_bar_width, bar_height, 4, 4)

            # Bar fill
            win_rate = model['win_rate']
            bar_width = int(max_bar_width * win_rate / 100)
            if win_rate >= 90:
                color = COLORS['profit']
            elif win_rate >= 70:
                color = COLORS['accent_yellow']
            else:
                color = COLORS['loss']
            painter.setBrush(QBrush(QColor(color)))
            painter.drawRoundedRect(bar_x, y, bar_width, bar_height, 4, 4)

            # Percentage
            painter.setPen(QColor(COLORS['text_primary']))
            painter.setFont(QFont("Segoe UI", 10, QFont.Weight.Bold))
            painter.drawText(bar_x + max_bar_width + 10, y + 17, f"{win_rate:.1f}%")

            y += bar_height + spacing

        painter.end()


class TradeCalendarWidget(QFrame):
    """Calendar heatmap showing daily P&L."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(180)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        self.daily_pnl: Dict[str, float] = {}
        self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample daily P&L data for past 4 weeks."""
        today = datetime.now()
        for i in range(28):
            date = today - timedelta(days=i)
            if date.weekday() < 5:  # Weekdays only
                # Random P&L with slight positive bias
                pnl = random.uniform(-500, 700)
                self.daily_pnl[date.strftime("%Y-%m-%d")] = pnl

    def set_data(self, daily_pnl: Dict[str, float]):
        self.daily_pnl = daily_pnl
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 20
        cell_size = 30
        cell_spacing = 4

        # Title
        painter.setPen(QColor(COLORS['text_primary']))
        painter.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        painter.drawText(margin, 25, "Trade Calendar (Daily P&L)")

        # Day headers
        days = ['Mon', 'Tue', 'Wed', 'Thu', 'Fri']
        painter.setFont(QFont("Segoe UI", 9))
        painter.setPen(QColor(COLORS['text_muted']))
        for i, day in enumerate(days):
            x = margin + 50 + i * (cell_size + cell_spacing)
            painter.drawText(x, 50, day)

        # Draw 4 weeks
        today = datetime.now()
        start_of_week = today - timedelta(days=today.weekday())

        for week in range(4):
            week_start = start_of_week - timedelta(weeks=3-week)
            y = 65 + week * (cell_size + cell_spacing)

            # Week label
            painter.setPen(QColor(COLORS['text_muted']))
            painter.drawText(margin, y + 20, f"W{week+1}")

            for day in range(5):
                date = week_start + timedelta(days=day)
                date_str = date.strftime("%Y-%m-%d")
                x = margin + 50 + day * (cell_size + cell_spacing)

                pnl = self.daily_pnl.get(date_str, 0)

                # Determine color based on P&L
                if pnl > 300:
                    color = QColor("#00c853")  # Strong green
                elif pnl > 0:
                    color = QColor("#69f0ae")  # Light green
                elif pnl > -300:
                    color = QColor("#ff8a80")  # Light red
                elif pnl != 0:
                    color = QColor("#ff5252")  # Strong red
                else:
                    color = QColor(COLORS['bg_light'])  # No trades

                painter.setBrush(QBrush(color))
                painter.setPen(Qt.PenStyle.NoPen)
                painter.drawRoundedRect(x, y, cell_size, cell_size, 4, 4)

                # P&L text inside cell
                if pnl != 0:
                    painter.setPen(QColor("#ffffff" if abs(pnl) > 100 else COLORS['text_primary']))
                    painter.setFont(QFont("Segoe UI", 7))
                    text = f"{pnl/1000:.1f}K" if abs(pnl) >= 1000 else f"{pnl:.0f}"
                    painter.drawText(x + 2, y + 20, text[:4])

        painter.end()


class TimeAnalysisWidget(QFrame):
    """Analysis of best/worst trading hours."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 15px;
            }}
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Time Analysis")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(title)

        # Best Hour
        best_layout = QHBoxLayout()
        best_label = QLabel("Best Hour:")
        best_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        best_layout.addWidget(best_label)
        self.best_hour = QLabel("10:00 AM")
        self.best_hour.setStyleSheet(f"color: {COLORS['profit']}; font-weight: bold;")
        self.best_hour.setToolTip("Hour with highest average profit")
        best_layout.addWidget(self.best_hour)
        best_layout.addStretch()
        layout.addLayout(best_layout)

        # Worst Hour
        worst_layout = QHBoxLayout()
        worst_label = QLabel("Worst Hour:")
        worst_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        worst_layout.addWidget(worst_label)
        self.worst_hour = QLabel("3:00 PM")
        self.worst_hour.setStyleSheet(f"color: {COLORS['loss']}; font-weight: bold;")
        self.worst_hour.setToolTip("Hour with lowest average profit")
        worst_layout.addWidget(self.worst_hour)
        worst_layout.addStretch()
        layout.addLayout(worst_layout)

        # Average Hold Time
        hold_layout = QHBoxLayout()
        hold_label = QLabel("Avg Hold Time:")
        hold_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        hold_layout.addWidget(hold_label)
        self.avg_hold = QLabel("2.3 hours")
        self.avg_hold.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: bold;")
        self.avg_hold.setToolTip("Average time positions are held")
        hold_layout.addWidget(self.avg_hold)
        hold_layout.addStretch()
        layout.addLayout(hold_layout)

        # Most Active Day
        active_layout = QHBoxLayout()
        active_label = QLabel("Most Active:")
        active_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        active_layout.addWidget(active_label)
        self.most_active = QLabel("Tuesday")
        self.most_active.setStyleSheet(f"color: {COLORS['accent_blue']}; font-weight: bold;")
        self.most_active.setToolTip("Day with most trades")
        active_layout.addWidget(self.most_active)
        active_layout.addStretch()
        layout.addLayout(active_layout)

    def update_data(self, best_hour: str, worst_hour: str, avg_hold: str, most_active: str):
        self.best_hour.setText(best_hour)
        self.worst_hour.setText(worst_hour)
        self.avg_hold.setText(avg_hold)
        self.most_active.setText(most_active)


class SymbolPerformanceTable(QFrame):
    """Table showing P&L breakdown by symbol."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        self.setup_ui()
        self.load_sample_data()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Title
        title = QLabel("Symbol Performance")
        title.setFont(QFont("Segoe UI", 12, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(title)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(4)
        self.table.setHorizontalHeaderLabels(["Symbol", "Trades", "Win Rate", "P&L"])
        self.table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.table.verticalHeader().setVisible(False)
        self.table.setAlternatingRowColors(True)
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
                color: {COLORS['text_muted']};
                border: none;
                padding: 8px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.table)

    def load_sample_data(self):
        """Load sample symbol performance data."""
        data = [
            ("NVDA", 12, 83.3, 1234.56),
            ("AAPL", 15, 73.3, 890.12),
            ("MSFT", 8, 75.0, 456.78),
            ("AMD", 10, 60.0, -123.45),
            ("GOOGL", 6, 66.7, 234.00),
        ]
        self.set_data(data)

    def set_data(self, data: List[tuple]):
        """Set table data: [(symbol, trades, win_rate, pnl), ...]"""
        self.table.setRowCount(len(data))
        for row, (symbol, trades, win_rate, pnl) in enumerate(data):
            self.table.setItem(row, 0, QTableWidgetItem(symbol))
            self.table.setItem(row, 1, QTableWidgetItem(str(trades)))

            win_item = QTableWidgetItem(f"{win_rate:.1f}%")
            win_item.setForeground(QColor(get_profit_color(win_rate - 50)))
            self.table.setItem(row, 2, win_item)

            pnl_item = QTableWidgetItem(f"${pnl:+,.2f}")
            pnl_item.setForeground(QColor(get_profit_color(pnl)))
            self.table.setItem(row, 3, pnl_item)


class AnalyticsTab(QWidget):
    """Analytics tab with performance visualizations."""

    def __init__(self, trade_logger=None, parent=None):
        super().__init__(parent)
        self.trade_logger = trade_logger
        self.setup_ui()

    def setup_ui(self):
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {COLORS['bg_dark']};
            }}
        """)

        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(15)
        layout.setContentsMargins(20, 20, 20, 20)

        # Header with title and time range selector
        header = QHBoxLayout()
        title = QLabel("Trading Analytics")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        header.addWidget(title)

        header.addStretch()

        self.time_range = QComboBox()
        self.time_range.addItems(["Last 7 Days", "Last 30 Days", "Last 90 Days", "All Time"])
        self.time_range.setToolTip("Select time range for analytics")
        self.time_range.currentTextChanged.connect(self.refresh_data)
        header.addWidget(self.time_range)

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Refresh analytics data")
        refresh_btn.clicked.connect(self.refresh_data)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Top row: Metrics cards
        metrics_layout = QHBoxLayout()
        metrics_layout.setSpacing(15)

        self.win_rate_card = MetricCard("Win Rate", "67.5%", "Based on closed trades")
        self.win_rate_card.setToolTip("Percentage of profitable trades")
        metrics_layout.addWidget(self.win_rate_card)

        self.sharpe_card = MetricCard("Sharpe Ratio", "1.85", "Risk-adjusted return")
        self.sharpe_card.setToolTip("Risk-adjusted return metric (>1 is good, >2 is excellent)")
        metrics_layout.addWidget(self.sharpe_card)

        self.max_dd_card = MetricCard("Max Drawdown", "-8.2%", "Largest peak-to-trough")
        self.max_dd_card.setToolTip("Maximum percentage decline from peak equity")
        metrics_layout.addWidget(self.max_dd_card)

        self.profit_factor_card = MetricCard("Profit Factor", "2.1", "Gross profit / gross loss")
        self.profit_factor_card.setToolTip("Ratio of gross profit to gross loss (>1 is profitable)")
        metrics_layout.addWidget(self.profit_factor_card)

        self.total_trades_card = MetricCard("Total Trades", "156", "Closed positions")
        self.total_trades_card.setToolTip("Total number of completed trades")
        metrics_layout.addWidget(self.total_trades_card)

        layout.addLayout(metrics_layout)

        # Middle row: Charts
        charts_layout = QHBoxLayout()
        charts_layout.setSpacing(15)

        # Equity curve (larger)
        self.equity_curve = EquityCurveWidget()
        self.equity_curve.setToolTip("Account equity over time")
        charts_layout.addWidget(self.equity_curve, stretch=2)

        # Model performance
        self.model_performance = ModelPerformanceWidget()
        self.model_performance.setToolTip("AI model win rates comparison")
        charts_layout.addWidget(self.model_performance, stretch=1)

        layout.addLayout(charts_layout)

        # Bottom row: Calendar and analysis
        bottom_layout = QHBoxLayout()
        bottom_layout.setSpacing(15)

        # Trade calendar
        self.trade_calendar = TradeCalendarWidget()
        self.trade_calendar.setToolTip("Daily P&L heatmap - green is profit, red is loss")
        bottom_layout.addWidget(self.trade_calendar, stretch=2)

        # Right column: Time analysis and symbol table
        right_column = QVBoxLayout()
        right_column.setSpacing(15)

        self.time_analysis = TimeAnalysisWidget()
        right_column.addWidget(self.time_analysis)

        self.symbol_table = SymbolPerformanceTable()
        self.symbol_table.setToolTip("P&L breakdown by trading symbol")
        right_column.addWidget(self.symbol_table)

        bottom_layout.addLayout(right_column, stretch=1)

        layout.addLayout(bottom_layout)

        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def refresh_data(self):
        """Refresh all analytics data."""
        # In a real implementation, this would pull from trade_logger
        # For now, regenerate sample data
        self.equity_curve.generate_sample_data()
        self.equity_curve.update()

        self.model_performance.generate_sample_data()
        self.model_performance.update()

        self.trade_calendar.generate_sample_data()
        self.trade_calendar.update()

    def update_metrics(self, win_rate: float, sharpe: float, max_dd: float,
                       profit_factor: float, total_trades: int):
        """Update the metric cards with real data."""
        self.win_rate_card.set_value(f"{win_rate:.1f}%", get_profit_color(win_rate - 50))
        self.sharpe_card.set_value(f"{sharpe:.2f}", get_profit_color(sharpe - 1))
        self.max_dd_card.set_value(f"{max_dd:.1f}%", COLORS['loss'])
        self.profit_factor_card.set_value(f"{profit_factor:.2f}", get_profit_color(profit_factor - 1))
        self.total_trades_card.set_value(str(total_trades))
