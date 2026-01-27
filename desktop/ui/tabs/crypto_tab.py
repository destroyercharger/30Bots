"""
Crypto Tab
Cryptocurrency positions, volatility scanner, and market overview
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QTableWidget, QTableWidgetItem,
    QHeaderView, QComboBox, QPushButton, QScrollArea,
    QProgressBar
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QFont, QColor, QPainter, QBrush, QPen

import sys
from pathlib import Path
from datetime import datetime
from typing import List, Dict, Optional
import random

# Add parent for imports
PARENT_DIR = Path(__file__).parent.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from ui.styles import COLORS, get_profit_color


class CryptoPortfolioWidget(QFrame):
    """Summary of crypto holdings."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Crypto Portfolio")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(title)

        # Total Value
        self.total_label = QLabel("$0.00")
        self.total_label.setFont(QFont("Segoe UI", 28, QFont.Weight.Bold))
        self.total_label.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.total_label.setToolTip("Total value of all crypto holdings")
        layout.addWidget(self.total_label)

        # 24h Change
        change_layout = QHBoxLayout()
        self.change_label = QLabel("+$0.00 (0.00%)")
        self.change_label.setFont(QFont("Segoe UI", 12))
        self.change_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        self.change_label.setToolTip("24-hour change in portfolio value")
        change_layout.addWidget(self.change_label)
        change_layout.addStretch()
        layout.addLayout(change_layout)

        # Divider
        divider = QFrame()
        divider.setFixedHeight(1)
        divider.setStyleSheet(f"background-color: {COLORS['border']};")
        layout.addWidget(divider)

        # Individual holdings
        self.holdings_layout = QVBoxLayout()
        self.holdings_layout.setSpacing(8)
        layout.addLayout(self.holdings_layout)

        # Sample data
        self.update_holdings([
            {"symbol": "BTC", "value": 45234.56, "change_pct": 2.34},
            {"symbol": "ETH", "value": 12456.78, "change_pct": -1.23},
            {"symbol": "SOL", "value": 3456.00, "change_pct": 5.67},
        ])

    def update_holdings(self, holdings: List[Dict]):
        """Update portfolio with holdings data."""
        # Clear existing
        while self.holdings_layout.count():
            item = self.holdings_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        total = 0
        total_change = 0

        for holding in holdings:
            row = QHBoxLayout()

            symbol = QLabel(holding['symbol'])
            symbol.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            symbol.setStyleSheet(f"color: {COLORS['accent_blue']};")
            row.addWidget(symbol)

            row.addStretch()

            value = holding['value']
            total += value
            change_pct = holding['change_pct']
            total_change += value * (change_pct / 100)

            value_label = QLabel(f"${value:,.2f}")
            value_label.setFont(QFont("Segoe UI", 11))
            value_label.setStyleSheet(f"color: {COLORS['text_primary']};")
            row.addWidget(value_label)

            change_label = QLabel(f"{change_pct:+.2f}%")
            change_label.setFont(QFont("Segoe UI", 10))
            change_label.setStyleSheet(f"color: {get_profit_color(change_pct)};")
            change_label.setFixedWidth(60)
            row.addWidget(change_label)

            container = QWidget()
            container.setLayout(row)
            self.holdings_layout.addWidget(container)

        # Update totals
        self.total_label.setText(f"${total:,.2f}")
        change_pct_total = (total_change / total * 100) if total > 0 else 0
        self.change_label.setText(f"{'+' if total_change >= 0 else ''}{total_change:,.2f} ({change_pct_total:+.2f}%) 24h")
        self.change_label.setStyleSheet(f"color: {get_profit_color(total_change)};")


class CryptoPositionsTable(QFrame):
    """Table showing crypto positions with details."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
            }}
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(15, 15, 15, 15)

        # Header
        header = QHBoxLayout()
        title = QLabel("Crypto Positions")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        header.addWidget(title)
        header.addStretch()

        refresh_btn = QPushButton("Refresh")
        refresh_btn.setToolTip("Refresh position data")
        refresh_btn.clicked.connect(self.refresh_positions)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Table
        self.table = QTableWidget()
        self.table.setColumnCount(7)
        self.table.setHorizontalHeaderLabels([
            "Symbol", "Quantity", "Entry Price", "Current", "P&L", "P&L %", "24h Change"
        ])
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
                padding: 10px;
                color: {COLORS['text_primary']};
            }}
            QHeaderView::section {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_muted']};
                border: none;
                padding: 10px;
                font-weight: bold;
            }}
        """)
        layout.addWidget(self.table)

        # Load sample data
        self.load_sample_data()

    def load_sample_data(self):
        """Load sample position data."""
        positions = [
            ("BTC/USD", 0.5, 42000.00, 45234.56, 1617.28, 7.70, 2.34),
            ("ETH/USD", 5.0, 2400.00, 2491.36, 456.80, 3.80, -1.23),
            ("SOL/USD", 50.0, 65.00, 69.12, 206.00, 6.34, 5.67),
        ]
        self.set_positions(positions)

    def set_positions(self, positions: List[tuple]):
        """Set position data."""
        self.table.setRowCount(len(positions))
        for row, (symbol, qty, entry, current, pnl, pnl_pct, change_24h) in enumerate(positions):
            self.table.setItem(row, 0, QTableWidgetItem(symbol))
            self.table.setItem(row, 1, QTableWidgetItem(f"{qty:.4f}"))
            self.table.setItem(row, 2, QTableWidgetItem(f"${entry:,.2f}"))
            self.table.setItem(row, 3, QTableWidgetItem(f"${current:,.2f}"))

            pnl_item = QTableWidgetItem(f"${pnl:+,.2f}")
            pnl_item.setForeground(QColor(get_profit_color(pnl)))
            self.table.setItem(row, 4, pnl_item)

            pnl_pct_item = QTableWidgetItem(f"{pnl_pct:+.2f}%")
            pnl_pct_item.setForeground(QColor(get_profit_color(pnl_pct)))
            self.table.setItem(row, 5, pnl_pct_item)

            change_item = QTableWidgetItem(f"{change_24h:+.2f}%")
            change_item.setForeground(QColor(get_profit_color(change_24h)))
            self.table.setItem(row, 6, change_item)

    def refresh_positions(self):
        """Refresh position data from broker."""
        # In real implementation, fetch from Alpaca crypto API
        self.load_sample_data()


class VolatilityScannerWidget(QFrame):
    """Real-time volatility scanner for crypto assets."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(12)

        # Title
        title = QLabel("Volatility Scanner")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        title.setToolTip("Real-time 24h volatility for major cryptocurrencies")
        layout.addWidget(title)

        # Scanner items
        self.items_layout = QVBoxLayout()
        self.items_layout.setSpacing(10)
        layout.addLayout(self.items_layout)

        # Load sample data
        self.update_volatility([
            {"symbol": "BTC", "volatility": 45, "trend": "up"},
            {"symbol": "ETH", "volatility": 52, "trend": "up"},
            {"symbol": "SOL", "volatility": 78, "trend": "down"},
            {"symbol": "DOGE", "volatility": 92, "trend": "up"},
            {"symbol": "AVAX", "volatility": 65, "trend": "down"},
        ])

        layout.addStretch()

    def update_volatility(self, data: List[Dict]):
        """Update volatility display."""
        # Clear existing
        while self.items_layout.count():
            item = self.items_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        for item in data:
            row = QHBoxLayout()

            # Symbol
            symbol = QLabel(item['symbol'])
            symbol.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
            symbol.setStyleSheet(f"color: {COLORS['text_primary']};")
            symbol.setFixedWidth(50)
            row.addWidget(symbol)

            # Progress bar for volatility
            progress = QProgressBar()
            progress.setRange(0, 100)
            progress.setValue(item['volatility'])
            progress.setTextVisible(False)
            progress.setFixedHeight(12)

            # Color based on volatility level
            if item['volatility'] >= 70:
                color = COLORS['loss']  # High volatility
                level = "High"
            elif item['volatility'] >= 40:
                color = COLORS['accent_yellow']  # Medium
                level = "Med"
            else:
                color = COLORS['profit']  # Low
                level = "Low"

            progress.setStyleSheet(f"""
                QProgressBar {{
                    background-color: {COLORS['bg_light']};
                    border-radius: 6px;
                }}
                QProgressBar::chunk {{
                    background-color: {color};
                    border-radius: 6px;
                }}
            """)
            progress.setToolTip(f"{item['volatility']}% volatility ({level})")
            row.addWidget(progress)

            # Percentage and level
            vol_label = QLabel(f"{item['volatility']}%")
            vol_label.setStyleSheet(f"color: {color}; font-weight: bold;")
            vol_label.setFixedWidth(45)
            row.addWidget(vol_label)

            level_label = QLabel(f"({level})")
            level_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
            level_label.setFixedWidth(35)
            row.addWidget(level_label)

            container = QWidget()
            container.setLayout(row)
            self.items_layout.addWidget(container)


class MarketOverviewWidget(QFrame):
    """Crypto market overview with key metrics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
                padding: 15px;
            }}
        """)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setSpacing(15)

        # Title
        title = QLabel("Market Overview")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(title)

        # Fear & Greed Index
        fg_layout = QHBoxLayout()
        fg_label = QLabel("Fear & Greed:")
        fg_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        fg_layout.addWidget(fg_label)
        fg_layout.addStretch()
        self.fear_greed = QLabel("62 (Greed)")
        self.fear_greed.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.fear_greed.setStyleSheet(f"color: {COLORS['profit']};")
        self.fear_greed.setToolTip("Market sentiment indicator (0=Extreme Fear, 100=Extreme Greed)")
        fg_layout.addWidget(self.fear_greed)
        layout.addLayout(fg_layout)

        # BTC Dominance
        dom_layout = QHBoxLayout()
        dom_label = QLabel("BTC Dominance:")
        dom_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        dom_layout.addWidget(dom_label)
        dom_layout.addStretch()
        self.btc_dominance = QLabel("52.3%")
        self.btc_dominance.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.btc_dominance.setStyleSheet(f"color: {COLORS['accent_blue']};")
        self.btc_dominance.setToolTip("Bitcoin's share of total crypto market cap")
        dom_layout.addWidget(self.btc_dominance)
        layout.addLayout(dom_layout)

        # 24h Volume
        vol_layout = QHBoxLayout()
        vol_label = QLabel("24h Volume:")
        vol_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        vol_layout.addWidget(vol_label)
        vol_layout.addStretch()
        self.volume_24h = QLabel("$45.2B")
        self.volume_24h.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.volume_24h.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.volume_24h.setToolTip("Total crypto trading volume in last 24 hours")
        vol_layout.addWidget(self.volume_24h)
        layout.addLayout(vol_layout)

        # Total Market Cap
        cap_layout = QHBoxLayout()
        cap_label = QLabel("Total Market Cap:")
        cap_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        cap_layout.addWidget(cap_label)
        cap_layout.addStretch()
        self.market_cap = QLabel("$1.8T")
        self.market_cap.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.market_cap.setStyleSheet(f"color: {COLORS['text_primary']};")
        self.market_cap.setToolTip("Total cryptocurrency market capitalization")
        cap_layout.addWidget(self.market_cap)
        layout.addLayout(cap_layout)

        # Active Cryptos
        active_layout = QHBoxLayout()
        active_label = QLabel("Active Cryptos:")
        active_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        active_layout.addWidget(active_label)
        active_layout.addStretch()
        self.active_cryptos = QLabel("12,500+")
        self.active_cryptos.setFont(QFont("Segoe UI", 11, QFont.Weight.Bold))
        self.active_cryptos.setStyleSheet(f"color: {COLORS['text_muted']};")
        active_layout.addWidget(self.active_cryptos)
        layout.addLayout(active_layout)

        layout.addStretch()

    def update_data(self, fear_greed: int, btc_dom: float, volume: str, market_cap: str):
        """Update market overview data."""
        # Fear & Greed with color
        if fear_greed >= 75:
            fg_text = f"{fear_greed} (Extreme Greed)"
            fg_color = COLORS['loss']
        elif fear_greed >= 55:
            fg_text = f"{fear_greed} (Greed)"
            fg_color = COLORS['profit']
        elif fear_greed >= 45:
            fg_text = f"{fear_greed} (Neutral)"
            fg_color = COLORS['text_muted']
        elif fear_greed >= 25:
            fg_text = f"{fear_greed} (Fear)"
            fg_color = COLORS['accent_yellow']
        else:
            fg_text = f"{fear_greed} (Extreme Fear)"
            fg_color = COLORS['loss']

        self.fear_greed.setText(fg_text)
        self.fear_greed.setStyleSheet(f"color: {fg_color}; font-weight: bold;")

        self.btc_dominance.setText(f"{btc_dom:.1f}%")
        self.volume_24h.setText(volume)
        self.market_cap.setText(market_cap)


class CryptoPriceChart(QFrame):
    """Simple price chart for selected crypto."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setMinimumHeight(250)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
            }}
        """)
        self.symbol = "BTC/USD"
        self.data_points: List[float] = []
        self.generate_sample_data()

    def generate_sample_data(self):
        """Generate sample price data."""
        price = 45000
        self.data_points = [price]
        for _ in range(48):  # 48 hours of data
            change = random.uniform(-0.02, 0.022) * price
            price += change
            self.data_points.append(price)

    def set_symbol(self, symbol: str):
        self.symbol = symbol
        self.generate_sample_data()
        self.update()

    def paintEvent(self, event):
        super().paintEvent(event)
        if len(self.data_points) < 2:
            return

        painter = QPainter(self)
        painter.setRenderHint(QPainter.RenderHint.Antialiasing)

        margin = 60
        top_margin = 50
        width = self.width() - 2 * margin
        height = self.height() - top_margin - 30

        min_val = min(self.data_points)
        max_val = max(self.data_points)
        val_range = max_val - min_val or 1

        # Title
        painter.setPen(QColor(COLORS['text_primary']))
        painter.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        painter.drawText(20, 30, f"{self.symbol} - 48h Chart")

        # Current price
        current = self.data_points[-1]
        change = ((current - self.data_points[0]) / self.data_points[0]) * 100
        color = COLORS['profit'] if change >= 0 else COLORS['loss']
        painter.setPen(QColor(color))
        painter.setFont(QFont("Segoe UI", 12))
        painter.drawText(self.width() - 180, 30, f"${current:,.2f} ({change:+.2f}%)")

        # Grid lines
        painter.setPen(QPen(QColor(COLORS['border']), 1, Qt.PenStyle.DotLine))
        for i in range(5):
            y = top_margin + (height * i / 4)
            painter.drawLine(margin, int(y), self.width() - margin, int(y))

        # Y-axis labels
        painter.setPen(QColor(COLORS['text_muted']))
        painter.setFont(QFont("Segoe UI", 9))
        for i in range(5):
            y = top_margin + (height * i / 4)
            val = max_val - (val_range * i / 4)
            painter.drawText(5, int(y + 4), f"${val:,.0f}")

        # Draw the curve
        start_val = self.data_points[0]
        end_val = self.data_points[-1]
        line_color = QColor(COLORS['profit']) if end_val >= start_val else QColor(COLORS['loss'])
        painter.setPen(QPen(line_color, 2))

        points = []
        for i, val in enumerate(self.data_points):
            x = margin + (width * i / (len(self.data_points) - 1))
            y = top_margin + height - (height * (val - min_val) / val_range)
            points.append((int(x), int(y)))

        for i in range(len(points) - 1):
            painter.drawLine(points[i][0], points[i][1], points[i+1][0], points[i+1][1])

        # Fill under curve
        fill_color = QColor(line_color)
        fill_color.setAlpha(30)
        painter.setBrush(QBrush(fill_color))
        painter.setPen(Qt.PenStyle.NoPen)

        from PyQt6.QtGui import QPolygon
        from PyQt6.QtCore import QPoint
        polygon = QPolygon()
        polygon.append(QPoint(points[0][0], top_margin + height))
        for x, y in points:
            polygon.append(QPoint(x, y))
        polygon.append(QPoint(points[-1][0], top_margin + height))
        painter.drawPolygon(polygon)

        painter.end()


class CryptoTab(QWidget):
    """Crypto trading tab with positions and market data."""

    trade_requested = pyqtSignal(str, str, float)  # symbol, side, quantity

    def __init__(self, broker=None, parent=None):
        super().__init__(parent)
        self.broker = broker
        self.setup_ui()

        # Auto-refresh timer
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_data)
        self.refresh_timer.start(30000)  # Refresh every 30 seconds

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

        # Header
        header = QHBoxLayout()
        title = QLabel("Cryptocurrency Trading")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        header.addWidget(title)

        header.addStretch()

        # Symbol selector for chart
        header.addWidget(QLabel("Chart:"))
        self.symbol_combo = QComboBox()
        self.symbol_combo.addItems(["BTC/USD", "ETH/USD", "SOL/USD", "DOGE/USD", "AVAX/USD"])
        self.symbol_combo.setToolTip("Select cryptocurrency to display in chart")
        self.symbol_combo.currentTextChanged.connect(self.on_symbol_changed)
        header.addWidget(self.symbol_combo)

        refresh_btn = QPushButton("Refresh All")
        refresh_btn.setToolTip("Refresh all crypto data")
        refresh_btn.clicked.connect(self.refresh_data)
        header.addWidget(refresh_btn)

        layout.addLayout(header)

        # Top row: Portfolio and Chart
        top_row = QHBoxLayout()
        top_row.setSpacing(15)

        # Portfolio widget
        self.portfolio = CryptoPortfolioWidget()
        self.portfolio.setFixedWidth(280)
        top_row.addWidget(self.portfolio)

        # Price chart
        self.price_chart = CryptoPriceChart()
        top_row.addWidget(self.price_chart)

        layout.addLayout(top_row)

        # Middle: Positions table
        self.positions_table = CryptoPositionsTable()
        layout.addWidget(self.positions_table)

        # Bottom row: Volatility scanner and Market overview
        bottom_row = QHBoxLayout()
        bottom_row.setSpacing(15)

        self.volatility_scanner = VolatilityScannerWidget()
        bottom_row.addWidget(self.volatility_scanner)

        self.market_overview = MarketOverviewWidget()
        bottom_row.addWidget(self.market_overview)

        layout.addLayout(bottom_row)

        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

    def on_symbol_changed(self, symbol: str):
        """Handle symbol selection change."""
        self.price_chart.set_symbol(symbol)

    def refresh_data(self):
        """Refresh all crypto data."""
        # In real implementation, fetch from Alpaca crypto API
        self.price_chart.generate_sample_data()
        self.price_chart.update()

    def set_broker(self, broker):
        """Set broker for data fetching."""
        self.broker = broker
