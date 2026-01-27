"""
Settings Tab
Application and trading configuration with persistence
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QGridLayout,
    QLabel, QFrame, QCheckBox, QSpinBox, QDoubleSpinBox,
    QComboBox, QPushButton, QGroupBox, QScrollArea,
    QLineEdit, QMessageBox
)
from PyQt6.QtCore import Qt, pyqtSignal, QSettings
from PyQt6.QtGui import QFont

import sys
from pathlib import Path

# Add parent for imports
PARENT_DIR = Path(__file__).parent.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from ui.styles import COLORS, get_profit_color


class SettingsSection(QGroupBox):
    """Base class for settings sections with consistent styling."""

    def __init__(self, title: str, parent=None):
        super().__init__(title, parent)
        self.setStyleSheet(f"""
            QGroupBox {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                margin-top: 12px;
                padding: 15px;
                font-weight: bold;
                color: {COLORS['text_primary']};
            }}
            QGroupBox::title {{
                subcontrol-origin: margin;
                left: 15px;
                padding: 0 8px;
                color: {COLORS['accent_blue']};
            }}
        """)


class TradingConfigSection(SettingsSection):
    """Trading configuration settings."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__("Trading Configuration", parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(15)

        row = 0

        # Auto Trading Enabled
        self.auto_trading_cb = QCheckBox("Auto Trading Enabled")
        self.auto_trading_cb.setToolTip("Enable/disable automatic trade execution based on AI signals")
        self.auto_trading_cb.setChecked(True)
        layout.addWidget(self.auto_trading_cb, row, 0, 1, 2)
        row += 1

        # Trading Modes
        layout.addWidget(QLabel("Trading Modes:"), row, 0)
        mode_layout = QHBoxLayout()
        self.day_trading_cb = QCheckBox("Day Trading (5-min)")
        self.day_trading_cb.setToolTip("Trade using 5-minute candles with tighter stops (2% SL, 4% TP)")
        self.day_trading_cb.setChecked(True)
        self.swing_trading_cb = QCheckBox("Swing Trading (Daily)")
        self.swing_trading_cb.setToolTip("Trade using daily candles with wider stops (5% SL, 12% TP)")
        self.swing_trading_cb.setChecked(True)
        mode_layout.addWidget(self.day_trading_cb)
        mode_layout.addWidget(self.swing_trading_cb)
        mode_layout.addStretch()
        layout.addLayout(mode_layout, row, 1)
        row += 1

        # Max Positions
        layout.addWidget(QLabel("Max Positions (per mode):"), row, 0)
        self.max_positions_spin = QSpinBox()
        self.max_positions_spin.setRange(1, 10)
        self.max_positions_spin.setValue(2)
        self.max_positions_spin.setToolTip("Maximum number of simultaneous positions per trading mode")
        layout.addWidget(self.max_positions_spin, row, 1)
        row += 1

        # Position Size
        layout.addWidget(QLabel("Position Size ($):"), row, 0)
        self.position_size_spin = QSpinBox()
        self.position_size_spin.setRange(100, 50000)
        self.position_size_spin.setSingleStep(500)
        self.position_size_spin.setValue(5000)
        self.position_size_spin.setPrefix("$")
        self.position_size_spin.setToolTip("Dollar amount for each trade")
        layout.addWidget(self.position_size_spin, row, 1)
        row += 1

        # Min Confidence
        layout.addWidget(QLabel("Min AI Confidence:"), row, 0)
        self.min_confidence_spin = QDoubleSpinBox()
        self.min_confidence_spin.setRange(0.50, 0.99)
        self.min_confidence_spin.setSingleStep(0.05)
        self.min_confidence_spin.setValue(0.70)
        self.min_confidence_spin.setToolTip("Minimum AI model confidence required to execute trades")
        layout.addWidget(self.min_confidence_spin, row, 1)
        row += 1

        # Day Trading Stop Loss
        layout.addWidget(QLabel("Day Trade Stop Loss %:"), row, 0)
        self.day_stop_loss_spin = QDoubleSpinBox()
        self.day_stop_loss_spin.setRange(0.5, 10.0)
        self.day_stop_loss_spin.setSingleStep(0.5)
        self.day_stop_loss_spin.setValue(2.0)
        self.day_stop_loss_spin.setSuffix("%")
        self.day_stop_loss_spin.setToolTip("Stop loss percentage for day trades")
        layout.addWidget(self.day_stop_loss_spin, row, 1)
        row += 1

        # Day Trading Take Profit
        layout.addWidget(QLabel("Day Trade Take Profit %:"), row, 0)
        self.day_take_profit_spin = QDoubleSpinBox()
        self.day_take_profit_spin.setRange(1.0, 20.0)
        self.day_take_profit_spin.setSingleStep(0.5)
        self.day_take_profit_spin.setValue(4.0)
        self.day_take_profit_spin.setSuffix("%")
        self.day_take_profit_spin.setToolTip("Take profit percentage for day trades")
        layout.addWidget(self.day_take_profit_spin, row, 1)
        row += 1

        # Swing Trading Stop Loss
        layout.addWidget(QLabel("Swing Trade Stop Loss %:"), row, 0)
        self.swing_stop_loss_spin = QDoubleSpinBox()
        self.swing_stop_loss_spin.setRange(1.0, 15.0)
        self.swing_stop_loss_spin.setSingleStep(0.5)
        self.swing_stop_loss_spin.setValue(5.0)
        self.swing_stop_loss_spin.setSuffix("%")
        self.swing_stop_loss_spin.setToolTip("Stop loss percentage for swing trades")
        layout.addWidget(self.swing_stop_loss_spin, row, 1)
        row += 1

        # Swing Trading Take Profit
        layout.addWidget(QLabel("Swing Trade Take Profit %:"), row, 0)
        self.swing_take_profit_spin = QDoubleSpinBox()
        self.swing_take_profit_spin.setRange(2.0, 30.0)
        self.swing_take_profit_spin.setSingleStep(1.0)
        self.swing_take_profit_spin.setValue(12.0)
        self.swing_take_profit_spin.setSuffix("%")
        self.swing_take_profit_spin.setToolTip("Take profit percentage for swing trades")
        layout.addWidget(self.swing_take_profit_spin, row, 1)

    def get_settings(self) -> dict:
        return {
            'auto_trading_enabled': self.auto_trading_cb.isChecked(),
            'day_trading_enabled': self.day_trading_cb.isChecked(),
            'swing_trading_enabled': self.swing_trading_cb.isChecked(),
            'max_positions': self.max_positions_spin.value(),
            'position_size': self.position_size_spin.value(),
            'min_confidence': self.min_confidence_spin.value(),
            'day_stop_loss_pct': self.day_stop_loss_spin.value() / 100,
            'day_take_profit_pct': self.day_take_profit_spin.value() / 100,
            'swing_stop_loss_pct': self.swing_stop_loss_spin.value() / 100,
            'swing_take_profit_pct': self.swing_take_profit_spin.value() / 100,
        }

    def load_settings(self, settings: dict):
        self.auto_trading_cb.setChecked(settings.get('auto_trading_enabled', True))
        self.day_trading_cb.setChecked(settings.get('day_trading_enabled', True))
        self.swing_trading_cb.setChecked(settings.get('swing_trading_enabled', True))
        self.max_positions_spin.setValue(settings.get('max_positions', 2))
        self.position_size_spin.setValue(settings.get('position_size', 5000))
        self.min_confidence_spin.setValue(settings.get('min_confidence', 0.70))
        self.day_stop_loss_spin.setValue(settings.get('day_stop_loss_pct', 0.02) * 100)
        self.day_take_profit_spin.setValue(settings.get('day_take_profit_pct', 0.04) * 100)
        self.swing_stop_loss_spin.setValue(settings.get('swing_stop_loss_pct', 0.05) * 100)
        self.swing_take_profit_spin.setValue(settings.get('swing_take_profit_pct', 0.12) * 100)


class APIConnectionsSection(SettingsSection):
    """API connection status and testing."""

    def __init__(self, parent=None):
        super().__init__("API Connections", parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(10)

        # Alpaca
        layout.addWidget(QLabel("Alpaca Trading:"), 0, 0)
        self.alpaca_status = QLabel("Checking...")
        self.alpaca_status.setStyleSheet(f"color: {COLORS['text_muted']};")
        layout.addWidget(self.alpaca_status, 0, 1)
        self.alpaca_test_btn = QPushButton("Test")
        self.alpaca_test_btn.setFixedWidth(80)
        self.alpaca_test_btn.clicked.connect(self.test_alpaca)
        layout.addWidget(self.alpaca_test_btn, 0, 2)

        # Gemini
        layout.addWidget(QLabel("Gemini AI:"), 1, 0)
        self.gemini_status = QLabel("Checking...")
        self.gemini_status.setStyleSheet(f"color: {COLORS['text_muted']};")
        layout.addWidget(self.gemini_status, 1, 1)
        self.gemini_test_btn = QPushButton("Test")
        self.gemini_test_btn.setFixedWidth(80)
        self.gemini_test_btn.clicked.connect(self.test_gemini)
        layout.addWidget(self.gemini_test_btn, 1, 2)

        # Polygon
        layout.addWidget(QLabel("Polygon Data:"), 2, 0)
        self.polygon_status = QLabel("Checking...")
        self.polygon_status.setStyleSheet(f"color: {COLORS['text_muted']};")
        layout.addWidget(self.polygon_status, 2, 1)
        self.polygon_test_btn = QPushButton("Test")
        self.polygon_test_btn.setFixedWidth(80)
        self.polygon_test_btn.clicked.connect(self.test_polygon)
        layout.addWidget(self.polygon_test_btn, 2, 2)

    def set_status(self, label: QLabel, connected: bool, message: str = ""):
        if connected:
            label.setText(f"Connected {message}")
            label.setStyleSheet(f"color: {COLORS['profit']};")
        else:
            label.setText(f"Disconnected {message}")
            label.setStyleSheet(f"color: {COLORS['loss']};")

    def test_alpaca(self):
        """Test Alpaca connection."""
        try:
            from config import ALPACA_API_KEY
            if ALPACA_API_KEY:
                self.set_status(self.alpaca_status, True, "")
            else:
                self.set_status(self.alpaca_status, False, "(No API key)")
        except Exception as e:
            self.set_status(self.alpaca_status, False, f"({str(e)[:20]})")

    def test_gemini(self):
        """Test Gemini connection."""
        try:
            from config import GEMINI_API_KEY
            if GEMINI_API_KEY:
                self.set_status(self.gemini_status, True, "")
            else:
                self.set_status(self.gemini_status, False, "(No API key)")
        except Exception as e:
            self.set_status(self.gemini_status, False, f"({str(e)[:20]})")

    def test_polygon(self):
        """Test Polygon connection."""
        try:
            from config import POLYGON_API_KEY
            if POLYGON_API_KEY:
                self.set_status(self.polygon_status, True, "")
            else:
                self.set_status(self.polygon_status, False, "(No API key)")
        except Exception as e:
            self.set_status(self.polygon_status, False, f"({str(e)[:20]})")

    def check_all(self):
        """Check all connections on startup."""
        self.test_alpaca()
        self.test_gemini()
        self.test_polygon()


class NotificationsSection(SettingsSection):
    """Notification preferences."""

    def __init__(self, parent=None):
        super().__init__("Notifications", parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(10)

        # Trade notifications
        self.trade_executed_cb = QCheckBox("Trade Executed")
        self.trade_executed_cb.setChecked(True)
        self.trade_executed_cb.setToolTip("Show notification when a trade is executed")
        layout.addWidget(self.trade_executed_cb, 0, 0)

        self.stop_loss_cb = QCheckBox("Stop Loss Hit")
        self.stop_loss_cb.setChecked(True)
        self.stop_loss_cb.setToolTip("Show notification when stop loss is triggered")
        layout.addWidget(self.stop_loss_cb, 0, 1)

        self.take_profit_cb = QCheckBox("Take Profit Hit")
        self.take_profit_cb.setChecked(True)
        self.take_profit_cb.setToolTip("Show notification when take profit is hit")
        layout.addWidget(self.take_profit_cb, 1, 0)

        self.market_hours_cb = QCheckBox("Market Open/Close")
        self.market_hours_cb.setChecked(False)
        self.market_hours_cb.setToolTip("Show notification at market open and close")
        layout.addWidget(self.market_hours_cb, 1, 1)

        self.desktop_notif_cb = QCheckBox("Desktop Notifications")
        self.desktop_notif_cb.setChecked(False)
        self.desktop_notif_cb.setToolTip("Show system desktop notifications")
        layout.addWidget(self.desktop_notif_cb, 2, 0)

        self.sound_alerts_cb = QCheckBox("Sound Alerts")
        self.sound_alerts_cb.setChecked(False)
        self.sound_alerts_cb.setToolTip("Play sound on important events")
        layout.addWidget(self.sound_alerts_cb, 2, 1)

    def get_settings(self) -> dict:
        return {
            'notify_trade_executed': self.trade_executed_cb.isChecked(),
            'notify_stop_loss': self.stop_loss_cb.isChecked(),
            'notify_take_profit': self.take_profit_cb.isChecked(),
            'notify_market_hours': self.market_hours_cb.isChecked(),
            'desktop_notifications': self.desktop_notif_cb.isChecked(),
            'sound_alerts': self.sound_alerts_cb.isChecked(),
        }

    def load_settings(self, settings: dict):
        self.trade_executed_cb.setChecked(settings.get('notify_trade_executed', True))
        self.stop_loss_cb.setChecked(settings.get('notify_stop_loss', True))
        self.take_profit_cb.setChecked(settings.get('notify_take_profit', True))
        self.market_hours_cb.setChecked(settings.get('notify_market_hours', False))
        self.desktop_notif_cb.setChecked(settings.get('desktop_notifications', False))
        self.sound_alerts_cb.setChecked(settings.get('sound_alerts', False))


class AppearanceSection(SettingsSection):
    """Appearance and UI settings."""

    def __init__(self, parent=None):
        super().__init__("Appearance", parent)
        self.setup_ui()

    def setup_ui(self):
        layout = QGridLayout(self)
        layout.setSpacing(10)

        # Theme
        layout.addWidget(QLabel("Theme:"), 0, 0)
        self.theme_combo = QComboBox()
        self.theme_combo.addItems(["Dark", "Light", "System"])
        self.theme_combo.setCurrentText("Dark")
        self.theme_combo.setToolTip("Application color theme")
        layout.addWidget(self.theme_combo, 0, 1)

        # Font Size
        layout.addWidget(QLabel("Font Size:"), 1, 0)
        self.font_size_combo = QComboBox()
        self.font_size_combo.addItems(["Small", "Medium", "Large"])
        self.font_size_combo.setCurrentText("Medium")
        self.font_size_combo.setToolTip("UI font size")
        layout.addWidget(self.font_size_combo, 1, 1)

        # Show Tooltips
        self.show_tooltips_cb = QCheckBox("Show Tooltips")
        self.show_tooltips_cb.setChecked(True)
        self.show_tooltips_cb.setToolTip("Display helpful tooltips on hover")
        layout.addWidget(self.show_tooltips_cb, 2, 0)

        # Animate Charts
        self.animate_charts_cb = QCheckBox("Animate Charts")
        self.animate_charts_cb.setChecked(True)
        self.animate_charts_cb.setToolTip("Enable smooth chart animations")
        layout.addWidget(self.animate_charts_cb, 2, 1)

    def get_settings(self) -> dict:
        return {
            'theme': self.theme_combo.currentText().lower(),
            'font_size': self.font_size_combo.currentText().lower(),
            'show_tooltips': self.show_tooltips_cb.isChecked(),
            'animate_charts': self.animate_charts_cb.isChecked(),
        }

    def load_settings(self, settings: dict):
        self.theme_combo.setCurrentText(settings.get('theme', 'dark').capitalize())
        self.font_size_combo.setCurrentText(settings.get('font_size', 'medium').capitalize())
        self.show_tooltips_cb.setChecked(settings.get('show_tooltips', True))
        self.animate_charts_cb.setChecked(settings.get('animate_charts', True))


class SettingsTab(QWidget):
    """Settings tab with all configuration options."""

    settings_changed = pyqtSignal(dict)

    def __init__(self, parent=None):
        super().__init__(parent)
        self.qsettings = QSettings("30Bots", "TradingApp")
        self.setup_ui()
        self.load_all_settings()

    def setup_ui(self):
        # Main layout with scroll area
        main_layout = QVBoxLayout(self)
        main_layout.setContentsMargins(0, 0, 0, 0)

        # Scroll area for settings
        scroll = QScrollArea()
        scroll.setWidgetResizable(True)
        scroll.setStyleSheet(f"""
            QScrollArea {{
                border: none;
                background-color: {COLORS['bg_dark']};
            }}
        """)

        # Container widget
        container = QWidget()
        layout = QVBoxLayout(container)
        layout.setSpacing(20)
        layout.setContentsMargins(20, 20, 20, 20)

        # Title
        title = QLabel("Settings")
        title.setFont(QFont("Segoe UI", 18, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        layout.addWidget(title)

        # Sections
        self.trading_config = TradingConfigSection()
        layout.addWidget(self.trading_config)

        self.api_connections = APIConnectionsSection()
        layout.addWidget(self.api_connections)

        self.notifications = NotificationsSection()
        layout.addWidget(self.notifications)

        self.appearance = AppearanceSection()
        layout.addWidget(self.appearance)

        # Action buttons
        btn_layout = QHBoxLayout()
        btn_layout.setSpacing(15)

        self.save_btn = QPushButton("Save Settings")
        self.save_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: white;
                border: none;
                padding: 12px 30px;
                border-radius: 6px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_blue']}cc;
            }}
        """)
        self.save_btn.setToolTip("Save all settings to disk")
        self.save_btn.clicked.connect(self.save_all_settings)
        btn_layout.addWidget(self.save_btn)

        self.reset_btn = QPushButton("Reset to Defaults")
        self.reset_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_primary']};
                border: 1px solid {COLORS['border']};
                padding: 12px 30px;
                border-radius: 6px;
            }}
            QPushButton:hover {{
                background-color: {COLORS['bg_medium']};
            }}
        """)
        self.reset_btn.setToolTip("Reset all settings to default values")
        self.reset_btn.clicked.connect(self.reset_to_defaults)
        btn_layout.addWidget(self.reset_btn)

        btn_layout.addStretch()
        layout.addLayout(btn_layout)

        layout.addStretch()

        scroll.setWidget(container)
        main_layout.addWidget(scroll)

        # Check API connections on startup
        self.api_connections.check_all()

    def get_all_settings(self) -> dict:
        """Get all settings as a dictionary."""
        return {
            'trading': self.trading_config.get_settings(),
            'notifications': self.notifications.get_settings(),
            'appearance': self.appearance.get_settings(),
        }

    def save_all_settings(self):
        """Save all settings to QSettings."""
        settings = self.get_all_settings()

        # Save trading settings
        for key, value in settings['trading'].items():
            self.qsettings.setValue(f"trading/{key}", value)

        # Save notification settings
        for key, value in settings['notifications'].items():
            self.qsettings.setValue(f"notifications/{key}", value)

        # Save appearance settings
        for key, value in settings['appearance'].items():
            self.qsettings.setValue(f"appearance/{key}", value)

        self.qsettings.sync()

        # Emit signal for main window to apply settings
        self.settings_changed.emit(settings)

        # Show confirmation
        QMessageBox.information(self, "Settings Saved", "Your settings have been saved successfully.")

    def load_all_settings(self):
        """Load all settings from QSettings."""
        # Load trading settings
        trading_settings = {}
        for key in ['auto_trading_enabled', 'day_trading_enabled', 'swing_trading_enabled',
                    'max_positions', 'position_size', 'min_confidence',
                    'day_stop_loss_pct', 'day_take_profit_pct',
                    'swing_stop_loss_pct', 'swing_take_profit_pct']:
            value = self.qsettings.value(f"trading/{key}")
            if value is not None:
                # Handle type conversion
                if key in ['auto_trading_enabled', 'day_trading_enabled', 'swing_trading_enabled']:
                    trading_settings[key] = value == 'true' or value == True
                elif key in ['max_positions', 'position_size']:
                    trading_settings[key] = int(value)
                else:
                    trading_settings[key] = float(value)
        if trading_settings:
            self.trading_config.load_settings(trading_settings)

        # Load notification settings
        notif_settings = {}
        for key in ['notify_trade_executed', 'notify_stop_loss', 'notify_take_profit',
                    'notify_market_hours', 'desktop_notifications', 'sound_alerts']:
            value = self.qsettings.value(f"notifications/{key}")
            if value is not None:
                notif_settings[key] = value == 'true' or value == True
        if notif_settings:
            self.notifications.load_settings(notif_settings)

        # Load appearance settings
        appearance_settings = {}
        for key in ['theme', 'font_size', 'show_tooltips', 'animate_charts']:
            value = self.qsettings.value(f"appearance/{key}")
            if value is not None:
                if key in ['show_tooltips', 'animate_charts']:
                    appearance_settings[key] = value == 'true' or value == True
                else:
                    appearance_settings[key] = value
        if appearance_settings:
            self.appearance.load_settings(appearance_settings)

    def reset_to_defaults(self):
        """Reset all settings to default values."""
        reply = QMessageBox.question(
            self, "Reset Settings",
            "Are you sure you want to reset all settings to defaults?",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No
        )

        if reply == QMessageBox.StandardButton.Yes:
            # Clear stored settings
            self.qsettings.clear()

            # Reload with defaults
            self.trading_config.load_settings({})
            self.notifications.load_settings({})
            self.appearance.load_settings({})

            QMessageBox.information(self, "Settings Reset", "All settings have been reset to defaults.")
