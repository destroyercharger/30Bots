"""
Stock Chart Widget
Candlestick chart with technical indicators using pyqtgraph
"""

import numpy as np
from datetime import datetime, timedelta
from typing import Optional, Dict, List, Tuple

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QComboBox, QFrame, QMenu, QCheckBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QTimer
from PyQt6.QtGui import QColor, QPen, QBrush, QAction

import pyqtgraph as pg
from pyqtgraph import PlotWidget, DateAxisItem, ViewBox

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.styles import COLORS


class CandlestickItem(pg.GraphicsObject):
    """Custom candlestick chart item"""

    def __init__(self, data=None):
        super().__init__()
        self.data = data  # numpy array: [time, open, high, low, close]
        self.picture = None
        self.generatePicture()

    def setData(self, data):
        self.data = data
        self.generatePicture()
        self.update()

    def generatePicture(self):
        from PyQt6.QtGui import QPicture, QPainter

        self.picture = QPicture()
        painter = QPainter(self.picture)

        if self.data is None or len(self.data) == 0:
            painter.end()
            return

        # Colors
        bull_color = QColor(COLORS['profit'])
        bear_color = QColor(COLORS['loss'])

        # Width calculation
        if len(self.data) > 1:
            w = (self.data[1][0] - self.data[0][0]) * 0.6
        else:
            w = 60 * 0.6  # Default 1-minute width

        for row in self.data:
            t, o, h, l, c = row[:5]

            if c >= o:
                # Bullish - green
                painter.setPen(pg.mkPen(bull_color, width=1))
                painter.setBrush(pg.mkBrush(bull_color))
            else:
                # Bearish - red
                painter.setPen(pg.mkPen(bear_color, width=1))
                painter.setBrush(pg.mkBrush(bear_color))

            # Draw wick (high-low line)
            painter.drawLine(
                pg.QtCore.QPointF(t, l),
                pg.QtCore.QPointF(t, h)
            )

            # Draw body
            painter.drawRect(
                pg.QtCore.QRectF(t - w/2, o, w, c - o)
            )

        painter.end()

    def paint(self, p, *args):
        if self.picture:
            self.picture.play(p)

    def boundingRect(self):
        if self.data is None or len(self.data) == 0:
            return pg.QtCore.QRectF()

        min_x = self.data[0][0]
        max_x = self.data[-1][0]
        min_y = np.min(self.data[:, 3])  # lows
        max_y = np.max(self.data[:, 2])  # highs

        return pg.QtCore.QRectF(min_x, min_y, max_x - min_x, max_y - min_y)


class VolumeItem(pg.BarGraphItem):
    """Volume bars widget"""

    def __init__(self, data=None):
        super().__init__(x=[], height=[], width=0.6)
        if data is not None:
            self.setData(data)

    def setData(self, data):
        if data is None or len(data) == 0:
            return

        times = data[:, 0]
        opens = data[:, 1]
        closes = data[:, 4]
        volumes = data[:, 5] if data.shape[1] > 5 else np.ones(len(data)) * 1000

        # Width calculation
        if len(data) > 1:
            w = (data[1][0] - data[0][0]) * 0.6
        else:
            w = 60 * 0.6

        # Color based on price direction
        colors = []
        for i in range(len(data)):
            if closes[i] >= opens[i]:
                colors.append(QColor(COLORS['profit']))
            else:
                colors.append(QColor(COLORS['loss']))

        self.setOpts(
            x=times,
            height=volumes,
            width=w,
            brushes=colors
        )


class StockChartWidget(QWidget):
    """Main stock chart widget with candlesticks and indicators"""

    # Signals
    symbol_changed = pyqtSignal(str)
    timeframe_changed = pyqtSignal(str)

    TIMEFRAMES = ['1m', '5m', '15m', '30m', '1h', '4h', '1d']

    def __init__(self, parent=None):
        super().__init__(parent)
        self.symbol = "AAPL"
        self.timeframe = "5m"
        self.data = None
        self.indicators = {}
        self.indicator_items = {}
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        # Toolbar
        toolbar = self.create_toolbar()
        layout.addWidget(toolbar)

        # Chart container
        chart_container = QWidget()
        chart_layout = QVBoxLayout(chart_container)
        chart_layout.setContentsMargins(0, 0, 0, 0)
        chart_layout.setSpacing(0)

        # Main price chart
        self.price_chart = self.create_price_chart()
        chart_layout.addWidget(self.price_chart, stretch=3)

        # Volume chart
        self.volume_chart = self.create_volume_chart()
        chart_layout.addWidget(self.volume_chart, stretch=1)

        layout.addWidget(chart_container)

        # Link X axes
        self.volume_chart.setXLink(self.price_chart)

        # Crosshair
        self.setup_crosshair()

    def create_toolbar(self) -> QFrame:
        """Create the chart toolbar"""
        toolbar = QFrame()
        toolbar.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
            }}
        """)

        layout = QHBoxLayout(toolbar)
        layout.setContentsMargins(8, 4, 8, 4)
        layout.setSpacing(8)

        # Symbol label
        self.symbol_label = QLabel(self.symbol)
        self.symbol_label.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 700;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(self.symbol_label)

        # Price label
        self.price_label = QLabel("$0.00")
        self.price_label.setStyleSheet(f"""
            font-size: 14px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(self.price_label)

        # Change label
        self.change_label = QLabel("+0.00%")
        self.change_label.setStyleSheet(f"""
            font-size: 12px;
            color: {COLORS['text_secondary']};
        """)
        layout.addWidget(self.change_label)

        layout.addStretch()

        # Timeframe buttons
        self.timeframe_buttons = {}
        for tf in self.TIMEFRAMES:
            btn = QPushButton(tf)
            btn.setCheckable(True)
            btn.setChecked(tf == self.timeframe)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_light']};
                    color: {COLORS['text_secondary']};
                    border: 1px solid {COLORS['border']};
                    padding: 4px 8px;
                    border-radius: 3px;
                    font-size: 11px;
                }}
                QPushButton:checked {{
                    background-color: {COLORS['accent_blue']};
                    color: white;
                    border-color: {COLORS['accent_blue']};
                }}
                QPushButton:hover {{
                    border-color: {COLORS['accent_blue']};
                }}
            """)
            btn.clicked.connect(lambda checked, t=tf: self.on_timeframe_clicked(t))
            layout.addWidget(btn)
            self.timeframe_buttons[tf] = btn

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {COLORS['border']};")
        layout.addWidget(sep)

        # Indicators button
        indicators_btn = QPushButton("Indicators")
        indicators_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_secondary']};
                border: 1px solid {COLORS['border']};
                padding: 4px 12px;
                border-radius: 3px;
                font-size: 11px;
            }}
            QPushButton:hover {{
                border-color: {COLORS['accent_blue']};
            }}
        """)
        indicators_btn.clicked.connect(self.show_indicators_menu)
        layout.addWidget(indicators_btn)

        return toolbar

    def create_price_chart(self) -> PlotWidget:
        """Create the main price chart"""
        # Custom axis for time
        date_axis = DateAxisItem(orientation='bottom')

        chart = PlotWidget(
            axisItems={'bottom': date_axis},
            background=COLORS['bg_dark']
        )

        # Configure axes
        chart.showGrid(x=True, y=True, alpha=0.2)
        chart.setLabel('right', 'Price', color=COLORS['text_secondary'])
        chart.hideAxis('left')
        chart.showAxis('right')

        # Styling
        chart.getAxis('bottom').setPen(pg.mkPen(COLORS['border']))
        chart.getAxis('right').setPen(pg.mkPen(COLORS['border']))
        chart.getAxis('bottom').setTextPen(pg.mkPen(COLORS['text_secondary']))
        chart.getAxis('right').setTextPen(pg.mkPen(COLORS['text_secondary']))

        # Candlestick item
        self.candle_item = CandlestickItem()
        chart.addItem(self.candle_item)

        # Enable auto-range and mouse interaction
        chart.setMouseEnabled(x=True, y=True)
        chart.enableAutoRange(axis=ViewBox.XYAxes)

        return chart

    def create_volume_chart(self) -> PlotWidget:
        """Create the volume chart"""
        date_axis = DateAxisItem(orientation='bottom')

        chart = PlotWidget(
            axisItems={'bottom': date_axis},
            background=COLORS['bg_dark']
        )

        # Configure axes
        chart.showGrid(x=True, y=True, alpha=0.1)
        chart.setLabel('right', 'Volume', color=COLORS['text_secondary'])
        chart.hideAxis('left')
        chart.showAxis('right')
        chart.setMaximumHeight(100)

        # Styling
        chart.getAxis('bottom').setPen(pg.mkPen(COLORS['border']))
        chart.getAxis('right').setPen(pg.mkPen(COLORS['border']))
        chart.getAxis('bottom').setTextPen(pg.mkPen(COLORS['text_secondary']))
        chart.getAxis('right').setTextPen(pg.mkPen(COLORS['text_secondary']))

        # Volume bars
        self.volume_item = VolumeItem()
        chart.addItem(self.volume_item)

        return chart

    def setup_crosshair(self):
        """Setup crosshair for price tracking"""
        self.vLine = pg.InfiniteLine(angle=90, movable=False, pen=pg.mkPen(COLORS['text_muted'], style=Qt.PenStyle.DashLine))
        self.hLine = pg.InfiniteLine(angle=0, movable=False, pen=pg.mkPen(COLORS['text_muted'], style=Qt.PenStyle.DashLine))

        self.price_chart.addItem(self.vLine, ignoreBounds=True)
        self.price_chart.addItem(self.hLine, ignoreBounds=True)

        self.price_chart.scene().sigMouseMoved.connect(self.on_mouse_move)

    def on_mouse_move(self, pos):
        """Handle mouse movement for crosshair"""
        if self.price_chart.sceneBoundingRect().contains(pos):
            mouse_point = self.price_chart.plotItem.vb.mapSceneToView(pos)
            self.vLine.setPos(mouse_point.x())
            self.hLine.setPos(mouse_point.y())

    def on_timeframe_clicked(self, timeframe: str):
        """Handle timeframe button click"""
        self.timeframe = timeframe

        # Update button states
        for tf, btn in self.timeframe_buttons.items():
            btn.setChecked(tf == timeframe)

        self.timeframe_changed.emit(timeframe)

    def show_indicators_menu(self):
        """Show indicators selection menu"""
        menu = QMenu(self)
        menu.setStyleSheet(f"""
            QMenu {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                padding: 4px;
            }}
            QMenu::item {{
                padding: 6px 20px;
                color: {COLORS['text_primary']};
            }}
            QMenu::item:selected {{
                background-color: {COLORS['bg_light']};
            }}
            QMenu::item:checked {{
                color: {COLORS['accent_blue']};
            }}
        """)

        # Moving Averages
        ma_menu = menu.addMenu("Moving Averages")
        for ma in ['SMA 9', 'SMA 20', 'SMA 50', 'SMA 200', 'EMA 9', 'EMA 21']:
            action = QAction(ma, self)
            action.setCheckable(True)
            action.setChecked(ma in self.indicators)
            action.triggered.connect(lambda checked, m=ma: self.toggle_indicator(m, checked))
            ma_menu.addAction(action)

        # Momentum
        mom_menu = menu.addMenu("Momentum")
        for ind in ['RSI', 'MACD', 'Stochastic']:
            action = QAction(ind, self)
            action.setCheckable(True)
            action.setChecked(ind in self.indicators)
            action.triggered.connect(lambda checked, i=ind: self.toggle_indicator(i, checked))
            mom_menu.addAction(action)

        # Volatility
        vol_menu = menu.addMenu("Volatility")
        for ind in ['Bollinger Bands', 'ATR', 'Keltner Channels']:
            action = QAction(ind, self)
            action.setCheckable(True)
            action.setChecked(ind in self.indicators)
            action.triggered.connect(lambda checked, i=ind: self.toggle_indicator(i, checked))
            vol_menu.addAction(action)

        # Trend
        trend_menu = menu.addMenu("Trend")
        for ind in ['SuperTrend', 'Ichimoku Cloud', 'VWAP']:
            action = QAction(ind, self)
            action.setCheckable(True)
            action.setChecked(ind in self.indicators)
            action.triggered.connect(lambda checked, i=ind: self.toggle_indicator(i, checked))
            trend_menu.addAction(action)

        menu.addSeparator()

        # Clear all
        clear_action = QAction("Clear All Indicators", self)
        clear_action.triggered.connect(self.clear_indicators)
        menu.addAction(clear_action)

        # Show menu
        sender = self.sender()
        if sender:
            menu.exec(sender.mapToGlobal(sender.rect().bottomLeft()))

    def toggle_indicator(self, name: str, enabled: bool):
        """Toggle an indicator on/off"""
        if enabled:
            self.add_indicator(name)
        else:
            self.remove_indicator(name)

    def add_indicator(self, name: str):
        """Add an indicator to the chart"""
        if self.data is None or len(self.data) == 0:
            return

        self.indicators[name] = True

        closes = self.data[:, 4]
        times = self.data[:, 0]

        # Calculate indicator
        if name.startswith('SMA'):
            period = int(name.split()[1])
            values = self.calc_sma(closes, period)
            color = COLORS['accent_blue']
            self.plot_line_indicator(name, times, values, color)

        elif name.startswith('EMA'):
            period = int(name.split()[1])
            values = self.calc_ema(closes, period)
            color = COLORS['accent_purple']
            self.plot_line_indicator(name, times, values, color)

        elif name == 'Bollinger Bands':
            upper, middle, lower = self.calc_bollinger(closes, 20, 2)
            self.plot_line_indicator(f'{name}_upper', times, upper, COLORS['text_muted'])
            self.plot_line_indicator(f'{name}_middle', times, middle, COLORS['accent_yellow'])
            self.plot_line_indicator(f'{name}_lower', times, lower, COLORS['text_muted'])

        elif name == 'VWAP':
            highs = self.data[:, 2]
            lows = self.data[:, 3]
            volumes = self.data[:, 5] if self.data.shape[1] > 5 else np.ones(len(self.data)) * 1000000
            values = self.calc_vwap(highs, lows, closes, volumes)
            self.plot_line_indicator(name, times, values, COLORS['accent_orange'])

    def remove_indicator(self, name: str):
        """Remove an indicator from the chart"""
        if name in self.indicators:
            del self.indicators[name]

        # Remove plot items
        items_to_remove = [key for key in self.indicator_items if key.startswith(name)]
        for key in items_to_remove:
            item = self.indicator_items.pop(key)
            self.price_chart.removeItem(item)

    def clear_indicators(self):
        """Remove all indicators"""
        for item in self.indicator_items.values():
            self.price_chart.removeItem(item)
        self.indicator_items.clear()
        self.indicators.clear()

    def plot_line_indicator(self, name: str, times, values, color: str):
        """Plot a line indicator"""
        pen = pg.mkPen(color, width=1)
        item = self.price_chart.plot(times, values, pen=pen)
        self.indicator_items[name] = item

    # Technical Indicator Calculations
    def calc_sma(self, data, period: int):
        """Simple Moving Average"""
        result = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            result[i] = np.mean(data[i - period + 1:i + 1])
        return result

    def calc_ema(self, data, period: int):
        """Exponential Moving Average"""
        result = np.full(len(data), np.nan)
        multiplier = 2 / (period + 1)
        result[period - 1] = np.mean(data[:period])
        for i in range(period, len(data)):
            result[i] = (data[i] - result[i - 1]) * multiplier + result[i - 1]
        return result

    def calc_bollinger(self, data, period: int = 20, std_dev: float = 2.0):
        """Bollinger Bands"""
        middle = self.calc_sma(data, period)
        std = np.full(len(data), np.nan)
        for i in range(period - 1, len(data)):
            std[i] = np.std(data[i - period + 1:i + 1])
        upper = middle + (std * std_dev)
        lower = middle - (std * std_dev)
        return upper, middle, lower

    def calc_vwap(self, highs, lows, closes, volumes):
        """Volume Weighted Average Price"""
        typical_price = (highs + lows + closes) / 3
        cumulative_tpv = np.cumsum(typical_price * volumes)
        cumulative_volume = np.cumsum(volumes)
        return cumulative_tpv / cumulative_volume

    def calc_rsi(self, data, period: int = 14):
        """Relative Strength Index"""
        deltas = np.diff(data)
        gains = np.where(deltas > 0, deltas, 0)
        losses = np.where(deltas < 0, -deltas, 0)

        avg_gain = np.full(len(data), np.nan)
        avg_loss = np.full(len(data), np.nan)
        rsi = np.full(len(data), np.nan)

        avg_gain[period] = np.mean(gains[:period])
        avg_loss[period] = np.mean(losses[:period])

        for i in range(period + 1, len(data)):
            avg_gain[i] = (avg_gain[i - 1] * (period - 1) + gains[i - 1]) / period
            avg_loss[i] = (avg_loss[i - 1] * (period - 1) + losses[i - 1]) / period

            if avg_loss[i] == 0:
                rsi[i] = 100
            else:
                rs = avg_gain[i] / avg_loss[i]
                rsi[i] = 100 - (100 / (1 + rs))

        return rsi

    def set_symbol(self, symbol: str):
        """Set the chart symbol"""
        self.symbol = symbol
        self.symbol_label.setText(symbol)
        self.symbol_changed.emit(symbol)

    def set_data(self, data: np.ndarray):
        """Set chart data - numpy array: [time, open, high, low, close, volume]"""
        self.data = data

        if data is None or len(data) == 0:
            return

        # Update candlesticks
        self.candle_item.setData(data)

        # Update volume
        self.volume_item.setData(data)

        # Update price labels
        last_close = data[-1, 4]
        first_open = data[0, 1]
        change_pct = ((last_close - first_open) / first_open) * 100

        self.price_label.setText(f"${last_close:.2f}")

        color = COLORS['profit'] if change_pct >= 0 else COLORS['loss']
        sign = "+" if change_pct >= 0 else ""
        self.change_label.setText(f"{sign}{change_pct:.2f}%")
        self.change_label.setStyleSheet(f"font-size: 12px; color: {color};")

        # Re-draw indicators
        self.redraw_indicators()

        # Auto-range
        self.price_chart.autoRange()
        self.volume_chart.autoRange()

    def redraw_indicators(self):
        """Redraw all active indicators"""
        active_indicators = list(self.indicators.keys())
        self.clear_indicators()
        for ind in active_indicators:
            self.indicators[ind] = True
            self.add_indicator(ind)

    def update_last_candle(self, time: float, o: float, h: float, l: float, c: float, v: float = 0):
        """Update the last candle (for real-time updates)"""
        if self.data is None or len(self.data) == 0:
            return

        # Update last row
        self.data[-1] = [time, o, h, l, c, v]

        # Refresh chart
        self.candle_item.setData(self.data)
        self.volume_item.setData(self.data)

        # Update price display
        change_pct = ((c - self.data[0, 1]) / self.data[0, 1]) * 100
        self.price_label.setText(f"${c:.2f}")

        color = COLORS['profit'] if change_pct >= 0 else COLORS['loss']
        sign = "+" if change_pct >= 0 else ""
        self.change_label.setText(f"{sign}{change_pct:.2f}%")
        self.change_label.setStyleSheet(f"font-size: 12px; color: {color};")

    def add_candle(self, time: float, o: float, h: float, l: float, c: float, v: float = 0):
        """Add a new candle to the chart"""
        new_candle = np.array([[time, o, h, l, c, v]])

        if self.data is None:
            self.data = new_candle
        else:
            self.data = np.vstack([self.data, new_candle])

        self.candle_item.setData(self.data)
        self.volume_item.setData(self.data)
        self.redraw_indicators()


def generate_sample_data(symbol: str = "AAPL", num_bars: int = 100) -> np.ndarray:
    """Generate sample OHLCV data for testing"""
    np.random.seed(42)

    now = datetime.now()
    base_price = 180.0  # AAPL-like price

    data = []
    current_price = base_price

    for i in range(num_bars):
        # Time (5-minute bars going back)
        time = (now - timedelta(minutes=5 * (num_bars - i))).timestamp()

        # Random price movement
        change = np.random.normal(0, 0.5)
        current_price = max(1, current_price + change)

        # OHLC
        o = current_price + np.random.uniform(-0.2, 0.2)
        c = current_price + np.random.uniform(-0.2, 0.2)
        h = max(o, c) + np.random.uniform(0, 0.3)
        l = min(o, c) - np.random.uniform(0, 0.3)
        v = np.random.randint(10000, 500000)

        data.append([time, o, h, l, c, v])

    return np.array(data)
