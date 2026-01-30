"""
Dashboard Tab
Main portfolio overview with positions, P&L, and activity monitoring
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QSplitter, QFrame,
    QLabel, QScrollArea, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.styles import COLORS
from ui.widgets.portfolio_widget import PortfolioWidget
from ui.widgets.positions_table import PositionsTableWidget
from ui.widgets.stock_chart import StockChartWidget, generate_sample_data
from ui.widgets.trading_panel import TradingPanel


class ActivityLogWidget(QWidget):
    """Real-time activity log display"""

    def __init__(self, title: str = "Activity Log", parent=None):
        super().__init__(parent)
        self.max_entries = 100
        self.setup_ui(title)

    def setup_ui(self, title: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 8)  # Bottom margin
        layout.setSpacing(10)

        # Header
        header = QLabel(title)
        header.setStyleSheet(f"""
            font-size: 15px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(header)

        # Log display
        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
                font-family: 'Consolas', 'Monaco', 'Courier New', monospace;
                font-size: 12px;
                color: {COLORS['text_secondary']};
                line-height: 1.5;
            }}
        """)
        layout.addWidget(self.log_display)

    def add_entry(self, message: str, level: str = "INFO"):
        """Add a log entry with color coding"""
        from datetime import datetime

        timestamp = datetime.now().strftime("%H:%M:%S")

        # Color based on level
        color = COLORS['text_secondary']
        if level == "TRADE":
            color = COLORS['profit']
        elif level == "ERROR":
            color = COLORS['loss']
        elif level == "WARNING":
            color = COLORS['accent_yellow']
        elif level == "AI":
            color = COLORS['accent_blue']

        html = f'<span style="color: {COLORS["text_muted"]}">{timestamp}</span> '
        html += f'<span style="color: {color}">[{level}]</span> '
        html += f'<span style="color: {COLORS["text_primary"]}">{message}</span><br>'

        self.log_display.insertHtml(html)

        # Scroll to bottom
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def clear(self):
        """Clear the log"""
        self.log_display.clear()


class ModelPredictionWidget(QFrame):
    """Display current AI model prediction"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 10px;
            }}
        """)

        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 16, 20, 16)  # More padding
        layout.setSpacing(10)

        # Header
        header = QLabel("AI Prediction")
        header.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(header)

        # Signal
        self.signal_label = QLabel("No signal")
        self.signal_label.setStyleSheet(f"""
            font-size: 20px;
            font-weight: 700;
            color: {COLORS['text_muted']};
        """)
        layout.addWidget(self.signal_label)

        # Confidence
        self.confidence_label = QLabel("Confidence: --%")
        self.confidence_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.confidence_label)

        # Model name
        self.model_label = QLabel("Model: --")
        self.model_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.model_label)

        # Risk:Reward
        self.rr_label = QLabel("R:R: --")
        self.rr_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.rr_label)

        layout.addStretch()

    def update_prediction(self, prediction: dict):
        """Update the prediction display"""
        if not prediction:
            self.signal_label.setText("No signal")
            self.signal_label.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {COLORS['text_muted']};")
            self.confidence_label.setText("Confidence: --%")
            self.model_label.setText("Model: --")
            self.rr_label.setText("R:R: --")
            return

        action = prediction.get('action', 'HOLD')
        confidence = prediction.get('confidence', 0) * 100
        model_name = prediction.get('model', 'Unknown')
        rr_ratio = prediction.get('risk_reward_ratio', 0)

        # Signal color
        if action == 'BUY':
            color = COLORS['profit']
        elif action == 'SELL':
            color = COLORS['loss']
        else:
            color = COLORS['text_muted']

        self.signal_label.setText(action)
        self.signal_label.setStyleSheet(f"font-size: 20px; font-weight: 700; color: {color};")
        self.confidence_label.setText(f"Confidence: {confidence:.1f}%")
        self.model_label.setText(f"Model: {model_name}")
        self.rr_label.setText(f"R:R: {rr_ratio:.1f}:1")


class DashboardTab(QWidget):
    """Main dashboard tab combining all overview widgets"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(20, 20, 20, 20)  # More outer padding
        layout.setSpacing(20)  # More space between major sections

        # Main splitter (horizontal)
        main_splitter = QSplitter(Qt.Orientation.Horizontal)
        main_splitter.setHandleWidth(8)  # Wider splitter handle
        main_splitter.setStyleSheet(f"""
            QSplitter::handle {{
                background-color: {COLORS['bg_medium']};
                margin: 4px 2px;
            }}
            QSplitter::handle:hover {{
                background-color: {COLORS['border_light']};
            }}
        """)

        # Left side - Portfolio, Chart, and Positions
        left_widget = QWidget()
        left_layout = QVBoxLayout(left_widget)
        left_layout.setContentsMargins(0, 0, 12, 0)  # Right margin for splitter
        left_layout.setSpacing(20)  # More space between widgets

        # Portfolio widget
        self.portfolio_widget = PortfolioWidget()
        self.portfolio_widget.setMinimumHeight(220)
        self.portfolio_widget.setMaximumHeight(280)
        left_layout.addWidget(self.portfolio_widget)

        # Vertical splitter for chart and positions
        chart_positions_splitter = QSplitter(Qt.Orientation.Vertical)

        # Stock chart
        self.stock_chart = StockChartWidget()
        self.stock_chart.setMinimumHeight(300)
        chart_positions_splitter.addWidget(self.stock_chart)

        # Positions table
        self.positions_table = PositionsTableWidget()
        chart_positions_splitter.addWidget(self.positions_table)

        # Set splitter sizes (60/40 chart/positions)
        chart_positions_splitter.setSizes([400, 300])

        left_layout.addWidget(chart_positions_splitter, stretch=1)

        main_splitter.addWidget(left_widget)

        # Right side - Trading Panel, AI Predictions, and Activity Logs
        right_widget = QWidget()
        right_widget.setMinimumWidth(380)  # Ensure trading panel has enough space
        right_layout = QVBoxLayout(right_widget)
        right_layout.setContentsMargins(12, 0, 0, 0)  # Left margin for splitter
        right_layout.setSpacing(20)  # More space between widgets

        # Trading panel
        self.trading_panel = TradingPanel()
        right_layout.addWidget(self.trading_panel)

        # Model prediction
        self.model_prediction = ModelPredictionWidget()
        right_layout.addWidget(self.model_prediction)

        # Vertical splitter for logs
        logs_splitter = QSplitter(Qt.Orientation.Vertical)

        # Decision log
        self.decision_log = ActivityLogWidget("Decision Log")
        logs_splitter.addWidget(self.decision_log)

        # Activity log
        self.activity_log = ActivityLogWidget("Activity Log")
        logs_splitter.addWidget(self.activity_log)

        right_layout.addWidget(logs_splitter, stretch=1)

        main_splitter.addWidget(right_widget)

        # Set splitter sizes (65/35 split for more trading panel space)
        main_splitter.setSizes([650, 400])

        layout.addWidget(main_splitter)

    def update_portfolio(self, data: dict):
        """Update portfolio summary"""
        self.portfolio_widget.update_portfolio(data)

    def update_positions(self, positions: dict):
        """Update positions table"""
        self.positions_table.update_positions(positions)

    def update_prediction(self, prediction: dict):
        """Update AI prediction display"""
        self.model_prediction.update_prediction(prediction)

    def log_decision(self, message: str):
        """Add entry to decision log"""
        self.decision_log.add_entry(message, "TRADE")

    def log_activity(self, message: str, level: str = "INFO"):
        """Add entry to activity log"""
        self.activity_log.add_entry(message, level)

    def set_chart_symbol(self, symbol: str):
        """Set the chart symbol"""
        self.stock_chart.set_symbol(symbol)

    def set_chart_data(self, data):
        """Set chart data (numpy array: [time, open, high, low, close, volume])"""
        self.stock_chart.set_data(data)

    def load_sample_chart_data(self, symbol: str = "AAPL"):
        """Load sample chart data for demonstration"""
        data = generate_sample_data(symbol)
        self.stock_chart.set_symbol(symbol)
        self.stock_chart.set_data(data)
