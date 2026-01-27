"""
Auto Trading Control Panel
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QFrame, QComboBox, QSpinBox, QDoubleSpinBox, QGroupBox,
    QGridLayout, QCheckBox, QTextEdit
)
from PyQt6.QtCore import Qt, pyqtSignal
from datetime import datetime

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.styles import COLORS


class TradingStatsWidget(QFrame):
    """Display live trading statistics."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()

    def setup_ui(self):
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 6px;
            }}
        """)

        layout = QGridLayout(self)
        layout.setContentsMargins(12, 8, 12, 8)
        layout.setSpacing(8)

        # Row 0: Header
        header = QLabel("Session Stats")
        header.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600;")
        layout.addWidget(header, 0, 0, 1, 4)

        # Row 1: Trades, Wins, Losses
        self.trades_label = self._create_stat("Trades", "0")
        layout.addWidget(self.trades_label, 1, 0)

        self.wins_label = self._create_stat("Wins", "0", COLORS['profit'])
        layout.addWidget(self.wins_label, 1, 1)

        self.losses_label = self._create_stat("Losses", "0", COLORS['loss'])
        layout.addWidget(self.losses_label, 1, 2)

        self.winrate_label = self._create_stat("Win Rate", "0%")
        layout.addWidget(self.winrate_label, 1, 3)

        # Row 2: P&L
        self.pnl_label = self._create_stat("P&L", "$0.00", COLORS['text_primary'], large=True)
        layout.addWidget(self.pnl_label, 2, 0, 1, 2)

        self.positions_label = self._create_stat("Positions", "0/4")
        layout.addWidget(self.positions_label, 2, 2)

        self.signals_label = self._create_stat("Signals", "0")
        layout.addWidget(self.signals_label, 2, 3)

    def _create_stat(self, label: str, value: str, color: str = None, large: bool = False):
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(2)

        label_widget = QLabel(label)
        label_widget.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 10px;")
        layout.addWidget(label_widget)

        value_widget = QLabel(value)
        size = "16px" if large else "13px"
        color = color or COLORS['text_secondary']
        value_widget.setStyleSheet(f"color: {color}; font-size: {size}; font-weight: 600;")
        value_widget.setObjectName("value")
        layout.addWidget(value_widget)

        return widget

    def update_stats(self, stats: dict):
        """Update the statistics display."""
        self._update_value(self.trades_label, str(stats.get('trades', 0)))
        self._update_value(self.wins_label, str(stats.get('wins', 0)))
        self._update_value(self.losses_label, str(stats.get('losses', 0)))

        win_rate = stats.get('win_rate', 0)
        self._update_value(self.winrate_label, f"{win_rate:.1f}%")

        pnl = stats.get('pnl', 0)
        color = COLORS['profit'] if pnl >= 0 else COLORS['loss']
        pnl_widget = self.pnl_label.findChild(QLabel, "value")
        if pnl_widget:
            pnl_widget.setText(f"${pnl:+,.2f}")
            pnl_widget.setStyleSheet(f"color: {color}; font-size: 16px; font-weight: 600;")

        active = stats.get('active_positions', 0)
        self._update_value(self.positions_label, f"{active}/4")
        self._update_value(self.signals_label, str(stats.get('signals', 0)))

    def _update_value(self, widget, value: str):
        value_label = widget.findChild(QLabel, "value")
        if value_label:
            value_label.setText(value)


class AutoTradingPanel(QWidget):
    """Control panel for automated trading."""

    # Signals
    start_trading = pyqtSignal(str, list)  # mode, watchlist
    stop_trading = pyqtSignal()
    pause_trading = pyqtSignal()
    resume_trading = pyqtSignal()
    close_all = pyqtSignal()
    mode_changed = pyqtSignal(str)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.is_trading = False
        self.is_paused = False
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Header
        header = QLabel("Auto Trading")
        header.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(header)

        # Main frame
        main_frame = QFrame()
        main_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        frame_layout = QVBoxLayout(main_frame)
        frame_layout.setContentsMargins(16, 12, 16, 12)
        frame_layout.setSpacing(12)

        # Status indicator
        status_layout = QHBoxLayout()
        self.status_indicator = QLabel("●")
        self.status_indicator.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 16px;")
        status_layout.addWidget(self.status_indicator)

        self.status_label = QLabel("STOPPED")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: 600;")
        status_layout.addWidget(self.status_label)
        status_layout.addStretch()

        frame_layout.addLayout(status_layout)

        # Mode selection
        mode_layout = QHBoxLayout()
        mode_label = QLabel("Mode:")
        mode_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        mode_layout.addWidget(mode_label)

        self.mode_combo = QComboBox()
        self.mode_combo.addItems(["Day Trading", "Swing Trading"])
        self.mode_combo.setToolTip(
            "Trading Mode\n\n"
            "Day Trading: Uses 5-minute candles, 2% stop loss, 4% take profit.\n"
            "Positions are typically held for minutes to hours.\n\n"
            "Swing Trading: Uses daily candles, 5% stop loss, 12% take profit.\n"
            "Positions are typically held for days to weeks."
        )
        self.mode_combo.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 6px 12px;
                color: {COLORS['text_primary']};
                min-width: 120px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['accent_blue']};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
        """)
        self.mode_combo.currentTextChanged.connect(self.on_mode_changed)
        mode_layout.addWidget(self.mode_combo)
        mode_layout.addStretch()

        frame_layout.addLayout(mode_layout)

        # Settings row
        settings_layout = QHBoxLayout()

        # Max positions
        pos_label = QLabel("Max Pos:")
        pos_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        settings_layout.addWidget(pos_label)

        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 10)
        self.max_positions_spin.setValue(4)
        self.max_positions_spin.setToolTip(
            "Maximum Positions\n\n"
            "The maximum number of positions to hold at once.\n"
            "Higher = more diversification but more capital required.\n"
            "Recommended: 4-6 for $100k account."
        )
        self.max_positions_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS['text_primary']};
                max-width: 60px;
            }}
        """)
        settings_layout.addWidget(self.max_positions_spin)

        # Min confidence
        conf_label = QLabel("Min Conf:")
        conf_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        settings_layout.addWidget(conf_label)

        self.min_confidence_spin = QSpinBox()
        self.min_confidence_spin.setRange(50, 95)
        self.min_confidence_spin.setValue(70)
        self.min_confidence_spin.setSuffix("%")
        self.min_confidence_spin.setToolTip(
            "Minimum Confidence\n\n"
            "Only execute trades when AI model confidence is above this threshold.\n"
            "Higher = fewer but more confident trades.\n"
            "Recommended: 65-75% for balanced risk/reward."
        )
        self.min_confidence_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 4px 8px;
                color: {COLORS['text_primary']};
                max-width: 70px;
            }}
        """)
        settings_layout.addWidget(self.min_confidence_spin)
        settings_layout.addStretch()

        frame_layout.addLayout(settings_layout)

        # Auto-execute checkbox
        self.auto_execute_check = QCheckBox("Auto-execute trades")
        self.auto_execute_check.setChecked(True)
        self.auto_execute_check.setToolTip(
            "Auto-Execute Trades\n\n"
            "When checked: AI signals are automatically executed as trades.\n"
            "When unchecked: Signals are logged but not executed (paper mode)."
        )
        self.auto_execute_check.setStyleSheet(f"color: {COLORS['text_secondary']};")
        frame_layout.addWidget(self.auto_execute_check)

        # Control buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(8)

        self.start_button = QPushButton("▶ START")
        self.start_button.setToolTip(
            "Start Auto Trading\n\n"
            "Begin scanning for trade signals and executing trades automatically.\n"
            "The AI will analyze the watchlist and enter positions when signals are found."
        )
        self.start_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['profit']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #00c853;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.start_button.clicked.connect(self.on_start_clicked)
        buttons_layout.addWidget(self.start_button)

        self.pause_button = QPushButton("⏸ PAUSE")
        self.pause_button.setEnabled(False)
        self.pause_button.setToolTip(
            "Pause/Resume Trading\n\n"
            "Temporarily stop scanning for new signals.\n"
            "Existing positions remain open and are still monitored.\n"
            "Click again to resume trading."
        )
        self.pause_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_yellow']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #ffc107;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.pause_button.clicked.connect(self.on_pause_clicked)
        buttons_layout.addWidget(self.pause_button)

        self.stop_button = QPushButton("⏹ STOP")
        self.stop_button.setEnabled(False)
        self.stop_button.setToolTip(
            "Stop Auto Trading\n\n"
            "Completely stop the auto trading system.\n"
            "No new signals will be generated or executed.\n"
            "Existing positions remain open."
        )
        self.stop_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['loss']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 12px 20px;
                font-size: 14px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #ff1744;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.stop_button.clicked.connect(self.on_stop_clicked)
        buttons_layout.addWidget(self.stop_button)

        frame_layout.addLayout(buttons_layout)

        # Close all button
        self.close_all_button = QPushButton("Close All Positions")
        self.close_all_button.setToolTip(
            "Close All Positions\n\n"
            "Immediately sell all open positions at market price.\n"
            "Use with caution - this action cannot be undone."
        )
        self.close_all_button.setStyleSheet(f"""
            QPushButton {{
                background-color: transparent;
                color: {COLORS['loss']};
                border: 1px solid {COLORS['loss']};
                border-radius: 4px;
                padding: 8px 16px;
                font-size: 12px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['loss']};
                color: white;
            }}
        """)
        self.close_all_button.clicked.connect(self.close_all.emit)
        frame_layout.addWidget(self.close_all_button)

        layout.addWidget(main_frame)

        # Stats widget
        self.stats_widget = TradingStatsWidget()
        layout.addWidget(self.stats_widget)

        # Activity log
        log_label = QLabel("Trading Log")
        log_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-weight: 600;")
        layout.addWidget(log_label)

        self.log_display = QTextEdit()
        self.log_display.setReadOnly(True)
        self.log_display.setMaximumHeight(150)
        self.log_display.setStyleSheet(f"""
            QTextEdit {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                font-family: 'Consolas', 'Monaco', monospace;
                font-size: 11px;
                color: {COLORS['text_secondary']};
            }}
        """)
        layout.addWidget(self.log_display)

        layout.addStretch()

    def on_mode_changed(self, text: str):
        mode = 'day' if 'Day' in text else 'swing'
        self.mode_changed.emit(mode)

    def on_start_clicked(self):
        mode = 'day' if 'Day' in self.mode_combo.currentText() else 'swing'

        # Get watchlist based on mode
        from core.auto_trader import DEFAULT_DAY_WATCHLIST, DEFAULT_SWING_WATCHLIST
        watchlist = DEFAULT_DAY_WATCHLIST if mode == 'day' else DEFAULT_SWING_WATCHLIST

        self.start_trading.emit(mode, watchlist)
        self.set_trading_state(True)

    def on_pause_clicked(self):
        if self.is_paused:
            self.resume_trading.emit()
            self.is_paused = False
            self.pause_button.setText("⏸ PAUSE")
            self.set_status("RUNNING", COLORS['profit'])
        else:
            self.pause_trading.emit()
            self.is_paused = True
            self.pause_button.setText("▶ RESUME")
            self.set_status("PAUSED", COLORS['accent_yellow'])

    def on_stop_clicked(self):
        self.stop_trading.emit()
        self.set_trading_state(False)

    def set_trading_state(self, trading: bool):
        """Update UI based on trading state."""
        self.is_trading = trading
        self.is_paused = False

        self.start_button.setEnabled(not trading)
        self.pause_button.setEnabled(trading)
        self.stop_button.setEnabled(trading)
        self.mode_combo.setEnabled(not trading)
        self.max_positions_spin.setEnabled(not trading)
        self.min_confidence_spin.setEnabled(not trading)
        self.auto_execute_check.setEnabled(not trading)

        if trading:
            self.set_status("RUNNING", COLORS['profit'])
            self.pause_button.setText("⏸ PAUSE")
        else:
            self.set_status("STOPPED", COLORS['text_muted'])

    def set_status(self, status: str, color: str):
        """Update status display."""
        self.status_indicator.setStyleSheet(f"color: {color}; font-size: 16px;")
        self.status_label.setText(status)
        self.status_label.setStyleSheet(f"color: {color}; font-weight: 600;")

    def update_stats(self, stats: dict):
        """Update statistics display."""
        self.stats_widget.update_stats(stats)

    def add_log(self, message: str, level: str = "INFO"):
        """Add message to trading log."""
        timestamp = datetime.now().strftime("%H:%M:%S")

        color = COLORS['text_secondary']
        if level == "TRADE":
            color = COLORS['profit']
        elif level == "WIN":
            color = COLORS['profit']
        elif level == "LOSS":
            color = COLORS['loss']
        elif level == "ERROR":
            color = COLORS['loss']
        elif level == "SIGNAL":
            color = COLORS['accent_blue']

        html = f'<span style="color: {COLORS["text_muted"]}">{timestamp}</span> '
        html += f'<span style="color: {color}">{message}</span><br>'

        self.log_display.insertHtml(html)
        scrollbar = self.log_display.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def get_config(self) -> dict:
        """Get current configuration."""
        return {
            'mode': 'day' if 'Day' in self.mode_combo.currentText() else 'swing',
            'max_positions': self.max_positions_spin.value(),
            'min_confidence': self.min_confidence_spin.value() / 100,
            'auto_execute': self.auto_execute_check.isChecked()
        }

    def _set_running_state(self):
        """Set the panel to running state (for auto-start)."""
        self.set_trading_state(True)
        self.add_log("Auto trading started automatically", "TRADE")
