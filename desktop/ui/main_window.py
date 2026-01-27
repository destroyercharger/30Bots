"""
30Bots Desktop Trading Application - Main Window
"""

from PyQt6.QtWidgets import (
    QMainWindow, QWidget, QVBoxLayout, QHBoxLayout,
    QTabWidget, QLabel, QStatusBar, QFrame, QSplitter,
    QMenuBar, QMenu, QToolBar
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QAction, QFont

from datetime import datetime
try:
    from zoneinfo import ZoneInfo
except ImportError:
    from backports.zoneinfo import ZoneInfo

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(__file__)))

from config import (
    APP_NAME, APP_VERSION, WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT,
    PRICE_UPDATE_INTERVAL, ALPACA_API_KEY, ALPACA_SECRET_KEY
)
from ui.styles import DARK_THEME, COLORS
from ui.tabs.dashboard_tab import DashboardTab
from ui.tabs.ai_trading_tab import AITradingTab
from ui.tabs.news_trading_tab import NewsTradingTab
from ui.tabs.crypto_tab import CryptoTab
from ui.tabs.analytics_tab import AnalyticsTab
from ui.tabs.settings_tab import SettingsTab
from ui.widgets.ai_terminal import AITerminalDock
from data.broker_adapter import AlpacaBroker, timeframe_to_alpaca, get_lookback_days
from data.websocket_worker import PriceUpdateManager, WEBSOCKET_AVAILABLE
from ai.prediction_worker import AIPredictionWorker, get_ai_brain, AI_AVAILABLE, create_sample_prediction
from core.trade_logger import get_trade_logger
from core.position_monitor import PositionMonitor
from core.auto_trader import AutoTrader, DEFAULT_DAY_WATCHLIST, DEFAULT_SWING_WATCHLIST


class MarketStatusBar(QFrame):
    """Top status bar showing market status, time, and key indicators"""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.setup_ui()
        self.start_timer()

    def setup_ui(self):
        self.setFixedHeight(36)
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_medium']};
                border-bottom: 1px solid {COLORS['border']};
            }}
            QLabel {{
                font-size: 12px;
            }}
        """)

        layout = QHBoxLayout(self)
        layout.setContentsMargins(16, 0, 16, 0)
        layout.setSpacing(24)

        # Market status
        self.market_status = QLabel("MARKET: CLOSED")
        self.market_status.setStyleSheet(f"color: {COLORS['text_muted']}; font-weight: 600;")
        layout.addWidget(self.market_status)

        # Time
        self.time_label = QLabel("--:--:-- ET")
        self.time_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.time_label)

        # Separator
        layout.addWidget(self._create_separator())

        # SPY indicator
        self.spy_label = QLabel("SPY: --")
        self.spy_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.spy_label)

        # VIX indicator
        self.vix_label = QLabel("VIX: --")
        self.vix_label.setStyleSheet(f"color: {COLORS['text_secondary']};")
        layout.addWidget(self.vix_label)

        layout.addStretch()

        # Separator
        layout.addWidget(self._create_separator())

        # Trading mode
        self.mode_label = QLabel("MODE: PAPER")
        self.mode_label.setStyleSheet(f"color: {COLORS['accent_yellow']}; font-weight: 600;")
        layout.addWidget(self.mode_label)

        # Connection status
        self.connection_label = QLabel("DISCONNECTED")
        self.connection_label.setStyleSheet(f"color: {COLORS['text_muted']};")
        layout.addWidget(self.connection_label)

    def _create_separator(self):
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.VLine)
        sep.setStyleSheet(f"color: {COLORS['border']};")
        return sep

    def start_timer(self):
        self.timer = QTimer(self)
        self.timer.timeout.connect(self.update_time)
        self.timer.start(1000)
        self.update_time()

    def update_time(self):
        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)
        self.time_label.setText(now.strftime("%H:%M:%S ET"))

        # Update market status based on time
        hour = now.hour
        minute = now.minute
        weekday = now.weekday()

        if weekday >= 5:  # Weekend
            self.set_market_status("WEEKEND", COLORS['text_muted'])
        elif hour < 4:
            self.set_market_status("CLOSED", COLORS['text_muted'])
        elif hour < 9 or (hour == 9 and minute < 30):
            self.set_market_status("PRE-MARKET", COLORS['accent_yellow'])
        elif hour < 16:
            self.set_market_status("OPEN", COLORS['profit'])
        elif hour < 20:
            self.set_market_status("AFTER-HOURS", COLORS['accent_orange'])
        else:
            self.set_market_status("CLOSED", COLORS['text_muted'])

    def set_market_status(self, status: str, color: str):
        self.market_status.setText(f"MARKET: {status}")
        self.market_status.setStyleSheet(f"color: {color}; font-weight: 600;")

    def update_spy(self, price: float, change_pct: float):
        color = COLORS['profit'] if change_pct >= 0 else COLORS['loss']
        sign = "+" if change_pct >= 0 else ""
        self.spy_label.setText(f"SPY: {sign}{change_pct:.2f}%")
        self.spy_label.setStyleSheet(f"color: {color};")

    def update_vix(self, value: float):
        color = COLORS['profit'] if value < 20 else (COLORS['accent_yellow'] if value < 30 else COLORS['loss'])
        self.vix_label.setText(f"VIX: {value:.1f}")
        self.vix_label.setStyleSheet(f"color: {color};")

    def set_connected(self, connected: bool):
        if connected:
            self.connection_label.setText("CONNECTED")
            self.connection_label.setStyleSheet(f"color: {COLORS['profit']};")
        else:
            self.connection_label.setText("DISCONNECTED")
            self.connection_label.setStyleSheet(f"color: {COLORS['text_muted']};")


class MainWindow(QMainWindow):
    """Main application window"""

    def __init__(self):
        super().__init__()
        self.current_chart_symbol = "AAPL"
        self.current_timeframe = "5m"
        self.broker = None
        self.price_manager = None
        self.ai_worker = None
        self.position_monitor = None
        self.auto_trader = None
        # Dual mode trading - day and swing traders
        self.day_trader = None
        self.swing_trader = None
        self.day_ai_worker = None
        self.swing_ai_worker = None
        self.trade_logger = get_trade_logger()
        self.subscribed_symbols = set()
        self.setup_broker()
        self.setup_ai()
        self.setup_trading_engine()
        self.setup_ui()
        self.setup_menu()

    def closeEvent(self, event):
        """Clean up on window close"""
        # Stop dual mode traders
        if self.day_trader:
            self.day_trader.stop()
            self.day_trader.wait(2000)
        if self.swing_trader:
            self.swing_trader.stop()
            self.swing_trader.wait(2000)
        if self.auto_trader:
            self.auto_trader.stop()
            self.auto_trader.wait(2000)
        if self.position_monitor:
            self.position_monitor.stop()
            self.position_monitor.wait(2000)
        if self.price_manager:
            self.price_manager.stop()
        if self.ai_worker:
            self.ai_worker.stop()
            self.ai_worker.wait(2000)
        if self.day_ai_worker:
            self.day_ai_worker.stop()
            self.day_ai_worker.wait(2000)
        if self.swing_ai_worker:
            self.swing_ai_worker.stop()
            self.swing_ai_worker.wait(2000)
        event.accept()

    def setup_ai(self):
        """Initialize the AI prediction system with dual mode support."""
        if AI_AVAILABLE:
            brain = get_ai_brain()
            if brain and brain.loaded:
                print(f"[AI] Loaded {len(brain.models)} models")
                print(f"[AI] Top model: {brain.model_priority[0]} ({brain.models[brain.model_priority[0]]['win_rate']*100:.1f}% win rate)")

                # Create AI worker for general use
                self.ai_worker = AIPredictionWorker(trading_mode='day')
                self.ai_worker.prediction_ready.connect(self.on_ai_prediction)
                self.ai_worker.error.connect(self.on_ai_error)

                # Create separate AI workers for day and swing trading
                self.day_ai_worker = AIPredictionWorker(trading_mode='day')
                self.day_ai_worker.prediction_ready.connect(self.on_ai_prediction)
                self.day_ai_worker.error.connect(self.on_ai_error)
                print("[AI] Day trading worker initialized (5-min candles)")

                self.swing_ai_worker = AIPredictionWorker(trading_mode='swing')
                self.swing_ai_worker.prediction_ready.connect(self.on_ai_prediction)
                self.swing_ai_worker.error.connect(self.on_ai_error)
                print("[AI] Swing trading worker initialized (daily candles)")
            else:
                print("[AI] Models not loaded - check model paths")
        else:
            print("[AI] AI brain not available")

    def setup_trading_engine(self):
        """Initialize the automated trading engine with dual mode support."""
        # Position monitor (shared between both traders)
        self.position_monitor = PositionMonitor(broker=self.broker)
        self.position_monitor.stop_loss_triggered.connect(self.on_stop_loss_triggered)
        self.position_monitor.take_profit_triggered.connect(self.on_take_profit_triggered)
        self.position_monitor.trailing_stop_triggered.connect(self.on_trailing_stop_triggered)
        self.position_monitor.trade_closed.connect(self.on_trade_closed)

        # Legacy auto trader (for compatibility)
        self.auto_trader = AutoTrader(broker=self.broker)
        self.auto_trader.set_ai_worker(self.ai_worker)
        self.auto_trader.set_position_monitor(self.position_monitor)
        self.auto_trader.trade_executed.connect(self.on_auto_trade_executed)
        self.auto_trader.signal_generated.connect(self.on_signal_generated)
        self.auto_trader.status_update.connect(self.on_trader_status)
        self.auto_trader.stats_updated.connect(self.on_stats_updated)
        self.auto_trader.error.connect(self.on_trader_error)

        # Day Trader (5-minute candles, tighter stops)
        self.day_trader = AutoTrader(broker=self.broker)
        self.day_trader.set_ai_worker(self.day_ai_worker)
        self.day_trader.set_position_monitor(self.position_monitor)
        self.day_trader.set_mode('day')
        self.day_trader.config['max_positions'] = 4  # Max 4 day trades at a time
        self.day_trader.trade_executed.connect(self.on_auto_trade_executed)
        self.day_trader.signal_generated.connect(self.on_signal_generated)
        self.day_trader.status_update.connect(lambda s: self.on_trader_status(f"[DAY] {s}"))
        self.day_trader.stats_updated.connect(self.on_stats_updated)
        self.day_trader.error.connect(self.on_trader_error)

        # Swing Trader (daily candles, wider stops)
        self.swing_trader = AutoTrader(broker=self.broker)
        self.swing_trader.set_ai_worker(self.swing_ai_worker)
        self.swing_trader.set_position_monitor(self.position_monitor)
        self.swing_trader.set_mode('swing')
        self.swing_trader.config['max_positions'] = 4  # Max 4 swing trades at a time
        self.swing_trader.trade_executed.connect(self.on_auto_trade_executed)
        self.swing_trader.signal_generated.connect(self.on_signal_generated)
        self.swing_trader.status_update.connect(lambda s: self.on_trader_status(f"[SWING] {s}"))
        self.swing_trader.stats_updated.connect(self.on_stats_updated)
        self.swing_trader.error.connect(self.on_trader_error)

        print("[Trading] Dual mode traders initialized:")
        print("[Trading]   Day Trader: 5-min candles, 2% SL, 4% TP")
        print("[Trading]   Swing Trader: Daily candles, 5% SL, 12% TP")

    def setup_broker(self):
        """Initialize Alpaca broker connection"""
        if ALPACA_API_KEY and ALPACA_SECRET_KEY:
            try:
                self.broker = AlpacaBroker(
                    api_key=ALPACA_API_KEY,
                    secret_key=ALPACA_SECRET_KEY,
                    paper=True  # Use paper trading
                )
                # Test connection
                account = self.broker.get_account()
                print(f"Connected to Alpaca - Account: ${account.equity:,.2f}")

                # Setup WebSocket streaming
                if WEBSOCKET_AVAILABLE:
                    self.price_manager = PriceUpdateManager(
                        api_key=ALPACA_API_KEY,
                        secret_key=ALPACA_SECRET_KEY
                    )
                    self.price_manager.add_callback(self.on_price_update)
                    self.price_manager.start()
                    print("WebSocket streaming enabled")
                else:
                    print("WebSocket not available - install websocket-client")

            except Exception as e:
                print(f"Failed to connect to Alpaca: {e}")
                self.broker = None
        else:
            print("Alpaca API keys not set - using sample data")

    def setup_ui(self):
        self.setWindowTitle(f"{APP_NAME} v{APP_VERSION}")
        self.setMinimumSize(WINDOW_MIN_WIDTH, WINDOW_MIN_HEIGHT)
        self.setStyleSheet(DARK_THEME)

        # Central widget
        central_widget = QWidget()
        self.setCentralWidget(central_widget)

        # Main layout
        main_layout = QVBoxLayout(central_widget)
        main_layout.setContentsMargins(0, 0, 0, 0)
        main_layout.setSpacing(0)

        # Market status bar
        self.market_bar = MarketStatusBar()
        main_layout.addWidget(self.market_bar)

        # Tab widget
        self.tabs = QTabWidget()
        self.tabs.setDocumentMode(True)
        main_layout.addWidget(self.tabs)

        # Create tabs
        self.create_tabs()

        # Status bar
        self.setup_status_bar()

        # Load recent trades and model performance (after status bar is created)
        self._load_trading_history()

    def create_tabs(self):
        """Create main application tabs"""
        # Dashboard tab (real implementation)
        self.dashboard_tab = DashboardTab()
        self.tabs.addTab(self.dashboard_tab, "Dashboard")

        # Add some sample data for testing
        self.load_sample_data()

        # AI Trading tab
        self.ai_trading_tab = AITradingTab()
        self.tabs.addTab(self.ai_trading_tab, "AI Trading")

        # Connect AI Trading panel signals
        self.ai_trading_tab.auto_panel.start_trading.connect(self.start_auto_trading)
        self.ai_trading_tab.auto_panel.stop_trading.connect(self.stop_auto_trading)
        self.ai_trading_tab.auto_panel.pause_trading.connect(self.pause_auto_trading)
        self.ai_trading_tab.auto_panel.resume_trading.connect(self.resume_auto_trading)
        self.ai_trading_tab.auto_panel.close_all.connect(self.close_all_positions)

        # Load default watchlist
        self.ai_trading_tab.load_default_watchlist('day')

        # Auto-start trading after 3 seconds (allows UI to fully initialize)
        QTimer.singleShot(3000, self._auto_start_trading)

        # News Trading tab
        self.news_tab = NewsTradingTab(broker=self.broker)
        self.news_tab.trade_signal.connect(self.on_news_trade_signal)
        self.tabs.addTab(self.news_tab, "News")

        # Crypto tab
        self.crypto_tab = CryptoTab(broker=self.broker)
        self.tabs.addTab(self.crypto_tab, "Crypto")

        # Analytics tab
        self.analytics_tab = AnalyticsTab(trade_logger=self.trade_logger)
        self.tabs.addTab(self.analytics_tab, "Analytics")

        # Settings tab
        self.settings_tab = SettingsTab()
        self.settings_tab.settings_changed.connect(self.on_settings_changed)
        self.tabs.addTab(self.settings_tab, "Settings")

        # AI Terminal dock (dockable panel)
        self.ai_terminal_dock = AITerminalDock(self)
        self.addDockWidget(Qt.DockWidgetArea.RightDockWidgetArea, self.ai_terminal_dock)
        self.ai_terminal_dock.hide()  # Hidden by default, toggle with Ctrl+Shift+A

    def create_placeholder_tab(self, title: str, description: str) -> QWidget:
        """Create a placeholder tab (will be replaced with real implementations)"""
        widget = QWidget()
        layout = QVBoxLayout(widget)
        layout.setAlignment(Qt.AlignmentFlag.AlignCenter)

        title_label = QLabel(title)
        title_label.setStyleSheet(f"""
            font-size: 24px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        title_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(title_label)

        desc_label = QLabel(description)
        desc_label.setStyleSheet(f"""
            font-size: 14px;
            color: {COLORS['text_secondary']};
        """)
        desc_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(desc_label)

        coming_soon = QLabel("Implementation in progress...")
        coming_soon.setStyleSheet(f"""
            font-size: 12px;
            color: {COLORS['text_muted']};
            margin-top: 20px;
        """)
        coming_soon.setAlignment(Qt.AlignmentFlag.AlignCenter)
        layout.addWidget(coming_soon)

        return widget

    def _auto_start_trading(self):
        """Automatically start DUAL MODE trading on launch (day + swing)."""
        try:
            # Check if market is open or extended hours
            from datetime import datetime
            try:
                from zoneinfo import ZoneInfo
            except ImportError:
                from backports.zoneinfo import ZoneInfo

            eastern = ZoneInfo("America/New_York")
            now = datetime.now(eastern)
            hour = now.hour
            weekday = now.weekday()

            # Only auto-start during market hours (4 AM - 8 PM ET, weekdays)
            if weekday < 5 and 4 <= hour < 20:
                # Update the auto panel UI to reflect started state
                self.ai_trading_tab.auto_panel._set_running_state()

                # Start BOTH traders simultaneously
                self.start_dual_trading()

                print("[AUTO-START] Dual mode trading started (Day + Swing)")
                self.dashboard_tab.log_activity(
                    "Auto-started DUAL MODE trading (Day + Swing)",
                    "INFO"
                )
            else:
                print("[AUTO-START] Market closed - auto trading not started")
                self.dashboard_tab.log_activity(
                    "Market closed - auto trading will not start automatically",
                    "INFO"
                )
        except Exception as e:
            print(f"[AUTO-START] Error: {e}")
            self.dashboard_tab.log_activity(f"Auto-start error: {e}", "ERROR")

    def _load_trading_history(self):
        """Load recent trades and model performance from database."""
        try:
            # Load recent trades
            recent_trades = self.trade_logger.get_recent_trades(20)
            self.ai_trading_tab.recent_trades.load_trades(recent_trades)

            # Load model performance
            model_perf = self.trade_logger.get_model_performance(7)
            self.ai_trading_tab.model_performance.update_performance(model_perf)

            # Load today's stats
            today_stats = self.trade_logger.get_today_stats()
            self.update_pnl(today_stats['total_pnl'])

        except Exception as e:
            print(f"Error loading trading history: {e}")

    def setup_menu(self):
        """Setup menu bar"""
        menubar = self.menuBar()

        # File menu
        file_menu = menubar.addMenu("File")

        new_action = QAction("New Trade", self)
        new_action.setShortcut("Ctrl+N")
        file_menu.addAction(new_action)

        file_menu.addSeparator()

        exit_action = QAction("Exit", self)
        exit_action.setShortcut("Ctrl+Q")
        exit_action.triggered.connect(self.close)
        file_menu.addAction(exit_action)

        # View menu
        view_menu = menubar.addMenu("View")

        refresh_action = QAction("Refresh", self)
        refresh_action.setShortcut("F5")
        view_menu.addAction(refresh_action)

        view_menu.addSeparator()

        # AI Assistant toggle
        self.ai_assistant_action = QAction("AI Assistant", self)
        self.ai_assistant_action.setShortcut("Ctrl+Shift+A")
        self.ai_assistant_action.setCheckable(True)
        self.ai_assistant_action.setChecked(False)
        self.ai_assistant_action.triggered.connect(self.toggle_ai_terminal)
        view_menu.addAction(self.ai_assistant_action)

        # Trading menu
        trading_menu = menubar.addMenu("Trading")

        pause_action = QAction("Pause Trading", self)
        trading_menu.addAction(pause_action)

        resume_action = QAction("Resume Trading", self)
        trading_menu.addAction(resume_action)

        trading_menu.addSeparator()

        close_all_action = QAction("Close All Positions", self)
        trading_menu.addAction(close_all_action)

        # AI menu
        ai_menu = menubar.addMenu("AI")

        scan_action = QAction("Scan Current Symbol", self)
        scan_action.setShortcut("F9")
        scan_action.triggered.connect(lambda: self.run_ai_scan())
        ai_menu.addAction(scan_action)

        ai_menu.addSeparator()

        start_scanning_action = QAction("Start Continuous Scanning", self)
        start_scanning_action.triggered.connect(lambda: self.start_ai_scanning())
        ai_menu.addAction(start_scanning_action)

        stop_scanning_action = QAction("Stop Scanning", self)
        stop_scanning_action.triggered.connect(self.stop_ai_scanning)
        ai_menu.addAction(stop_scanning_action)

        # Help menu
        help_menu = menubar.addMenu("Help")

        about_action = QAction("About", self)
        help_menu.addAction(about_action)

    def setup_status_bar(self):
        """Setup bottom status bar"""
        status_bar = self.statusBar()
        status_bar.setStyleSheet(f"""
            QStatusBar {{
                background-color: {COLORS['bg_light']};
                border-top: 1px solid {COLORS['border']};
            }}
        """)

        # Positions count
        self.positions_label = QLabel("Positions: 0/6")
        self.positions_label.setToolTip(
            "Open Positions\n\n"
            "Current number of open positions / Maximum allowed.\n"
            "When at maximum, no new trades will be opened."
        )
        status_bar.addWidget(self.positions_label)

        # Spacer
        spacer = QWidget()
        spacer.setFixedWidth(20)
        status_bar.addWidget(spacer)

        # P&L
        self.pnl_label = QLabel("Day P&L: $0.00")
        self.pnl_label.setToolTip(
            "Day Profit & Loss\n\n"
            "Your total profit or loss for today.\n"
            "Includes both realized and unrealized gains/losses."
        )
        status_bar.addWidget(self.pnl_label)

        # Permanent message on the right
        self.status_message = QLabel("Ready")
        self.status_message.setToolTip(
            "System Status\n\n"
            "Shows the current status of the trading system."
        )
        status_bar.addPermanentWidget(self.status_message)

    def update_positions_count(self, current: int, max_pos: int):
        if hasattr(self, 'positions_label') and self.positions_label:
            self.positions_label.setText(f"Positions: {current}/{max_pos}")

    def update_pnl(self, pnl: float):
        if hasattr(self, 'pnl_label') and self.pnl_label:
            color = COLORS['profit'] if pnl >= 0 else COLORS['loss']
            sign = "+" if pnl >= 0 else ""
            self.pnl_label.setText(f"Day P&L: {sign}${abs(pnl):,.2f}")
            self.pnl_label.setStyleSheet(f"color: {color};")

    def set_status_message(self, message: str):
        if hasattr(self, 'status_message') and self.status_message:
            self.status_message.setText(message)

    def load_sample_data(self):
        """Load data - real from Alpaca if connected, otherwise sample"""
        self.dashboard_tab.log_activity("Application started", "INFO")

        # Setup trading panel
        self.setup_trading_panel()

        if self.broker:
            self.load_real_data()
        else:
            self.load_demo_data()

        # Connect chart signals
        self.dashboard_tab.stock_chart.timeframe_changed.connect(self.on_timeframe_changed)
        self.dashboard_tab.positions_table.view_chart_requested.connect(self.on_view_chart_requested)
        self.dashboard_tab.stock_chart.symbol_changed.connect(self.on_chart_symbol_changed)

    def setup_trading_panel(self):
        """Setup the trading panel with broker and signals"""
        trading_panel = self.dashboard_tab.trading_panel

        # Connect broker
        trading_panel.set_broker(self.broker)

        # Set initial symbol
        trading_panel.set_symbol(self.current_chart_symbol)

        # Connect order signals
        trading_panel.order_executed.connect(self.on_order_executed)
        trading_panel.order_submitted.connect(self.on_order_submitted)

        # Connect AI scan signal
        trading_panel.ai_scan_requested.connect(self.run_ai_scan)

    def on_chart_symbol_changed(self, symbol: str):
        """Handle chart symbol change - update trading panel"""
        self.dashboard_tab.trading_panel.set_symbol(symbol)

    def on_order_executed(self, symbol: str, side: str, price: float, qty: int, total: float):
        """Handle order execution"""
        side_upper = side.upper()
        self.dashboard_tab.log_decision(f"{side_upper} {qty} {symbol} @ ${price:.2f}")
        self.set_status_message(f"Order executed: {side_upper} {qty} {symbol}")

        # Refresh positions after a short delay
        QTimer.singleShot(1000, self.refresh_positions)

    def on_order_submitted(self, order_details: dict):
        """Handle order submission"""
        self.dashboard_tab.log_activity(
            f"Order submitted: {order_details['side'].upper()} {order_details['qty']} {order_details['symbol']}",
            "TRADE"
        )

    def refresh_positions(self):
        """Refresh positions from broker"""
        if not self.broker:
            return

        try:
            positions = self.broker.get_positions()
            account = self.broker.get_account()

            # Update portfolio
            portfolio_data = {
                'total_value': account.equity,
                'day_pnl': account.equity - account.portfolio_value,
                'cash': account.cash,
                'positions': len(positions),
                'max_positions': 6,
                'win_rate': 0,
                'total_trades': 0
            }
            self.dashboard_tab.update_portfolio(portfolio_data)

            # Update positions table
            positions_dict = {}
            for pos in positions:
                positions_dict[pos.symbol] = {
                    'entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'shares': int(pos.qty),
                    'stop_loss': pos.avg_entry_price * 0.98,
                    'profit_target': pos.avg_entry_price * 1.06,
                    'risk_score': 70,
                    'ai_model_name': 'Manual'
                }
            self.dashboard_tab.update_positions(positions_dict)

            # Subscribe to new position symbols
            if self.price_manager:
                self.subscribe_symbols(list(positions_dict.keys()))

        except Exception as e:
            self.dashboard_tab.log_activity(f"Refresh error: {e}", "ERROR")

    def load_real_data(self):
        """Load real data from Alpaca"""
        try:
            # Get account info
            account = self.broker.get_account()

            # Get positions
            positions = self.broker.get_positions()

            # Update portfolio widget
            portfolio_data = {
                'total_value': account.equity,
                'day_pnl': account.equity - account.portfolio_value,  # Approximate
                'cash': account.cash,
                'positions': len(positions),
                'max_positions': 6,
                'win_rate': 0,  # Would need trade history
                'total_trades': 0
            }
            self.dashboard_tab.update_portfolio(portfolio_data)

            # Update positions table
            positions_dict = {}
            for pos in positions:
                positions_dict[pos.symbol] = {
                    'entry_price': pos.avg_entry_price,
                    'current_price': pos.current_price,
                    'shares': int(pos.qty),
                    'stop_loss': pos.avg_entry_price * 0.98,  # 2% default
                    'profit_target': pos.avg_entry_price * 1.06,  # 6% default
                    'risk_score': 70,
                    'ai_model_name': 'Manual'
                }
            self.dashboard_tab.update_positions(positions_dict)

            # Update market bar connection status
            self.market_bar.set_connected(True)

            # Log
            self.dashboard_tab.log_activity(f"Connected to Alpaca (${account.equity:,.2f})", "INFO")
            self.dashboard_tab.log_activity(f"Loaded {len(positions)} positions", "INFO")

            # Load chart data
            self.load_chart_data(self.current_chart_symbol, self.current_timeframe)

            # Subscribe to WebSocket for real-time updates
            if self.price_manager:
                # Subscribe to position symbols
                position_symbols = list(positions_dict.keys())
                if position_symbols:
                    self.subscribe_symbols(position_symbols)
                    self.dashboard_tab.log_activity(f"Streaming {len(position_symbols)} symbols", "INFO")

                # Subscribe to chart symbol and SPY
                self.subscribe_symbols([self.current_chart_symbol, "SPY"])

            # Check market status
            if self.broker.is_market_open():
                self.dashboard_tab.log_activity("Market is OPEN", "INFO")
            else:
                self.dashboard_tab.log_activity("Market is CLOSED", "INFO")

        except Exception as e:
            self.dashboard_tab.log_activity(f"Error loading data: {e}", "ERROR")
            self.load_demo_data()

    def load_demo_data(self):
        """Load sample/demo data"""
        self.dashboard_tab.log_activity("Using demo data (no API keys)", "WARNING")

        # Sample portfolio data
        portfolio_data = {
            'total_value': 105234.56,
            'day_pnl': 523.45,
            'cash': 35000.00,
            'positions': 3,
            'max_positions': 6,
            'win_rate': 72.5,
            'total_trades': 45
        }
        self.dashboard_tab.update_portfolio(portfolio_data)

        # Sample positions
        positions = {
            'AAPL': {
                'entry_price': 178.50,
                'current_price': 182.30,
                'shares': 50,
                'stop_loss': 175.00,
                'profit_target': 190.00,
                'risk_score': 85,
                'ai_model_name': 'Momentum_Selective'
            },
            'NVDA': {
                'entry_price': 485.00,
                'current_price': 512.50,
                'shares': 25,
                'stop_loss': 470.00,
                'profit_target': 550.00,
                'risk_score': 78,
                'ai_model_name': 'TrendFollowing_Moderate'
            },
            'MSFT': {
                'entry_price': 380.00,
                'current_price': 375.50,
                'shares': 30,
                'stop_loss': 365.00,
                'profit_target': 400.00,
                'risk_score': 62,
                'ai_model_name': 'MeanReversion_Selective'
            }
        }
        self.dashboard_tab.update_positions(positions)

        # Sample AI prediction
        prediction = {
            'action': 'BUY',
            'confidence': 0.948,
            'model': 'Momentum_Selective',
            'risk_reward_ratio': 3.0
        }
        self.dashboard_tab.update_prediction(prediction)

        # Load sample chart data
        self.dashboard_tab.load_sample_chart_data("AAPL")

    def load_chart_data(self, symbol: str, timeframe: str):
        """Load chart data from Alpaca"""
        self.current_chart_symbol = symbol
        self.current_timeframe = timeframe

        # Subscribe to WebSocket for this symbol
        self.subscribe_symbol(symbol)

        if not self.broker:
            self.dashboard_tab.load_sample_chart_data(symbol)
            return

        try:
            self.set_status_message(f"Loading {symbol} {timeframe}...")

            # Convert timeframe and get lookback
            alpaca_tf = timeframe_to_alpaca(timeframe)
            lookback = get_lookback_days(timeframe)

            # Calculate start date
            from datetime import timedelta
            end_date = datetime.now()
            start_date = end_date - timedelta(days=lookback)

            # Get bars from Alpaca
            bars = self.broker.get_bars_numpy(
                symbol=symbol,
                timeframe=alpaca_tf,
                start=start_date.strftime('%Y-%m-%d'),
                end=end_date.strftime('%Y-%m-%d'),
                limit=500
            )

            if len(bars) > 0:
                self.dashboard_tab.stock_chart.set_symbol(symbol)
                self.dashboard_tab.stock_chart.set_data(bars)
                self.set_status_message(f"Loaded {len(bars)} bars for {symbol}")
                self.dashboard_tab.log_activity(f"Loaded {symbol} chart ({len(bars)} bars)", "INFO")

                # Update trading panel with symbol and last price
                last_price = bars[-1, 4]  # Close price
                self.dashboard_tab.trading_panel.set_symbol(symbol)
                self.dashboard_tab.trading_panel.set_price(last_price)
            else:
                self.set_status_message(f"No data for {symbol}")
                self.dashboard_tab.log_activity(f"No chart data for {symbol}", "WARNING")
                # Fall back to sample data
                self.dashboard_tab.load_sample_chart_data(symbol)
                self.dashboard_tab.trading_panel.set_symbol(symbol)

        except Exception as e:
            self.set_status_message(f"Error: {str(e)[:50]}")
            self.dashboard_tab.log_activity(f"Chart error: {e}", "ERROR")
            # Fall back to sample data
            self.dashboard_tab.load_sample_chart_data(symbol)

    def on_timeframe_changed(self, timeframe: str):
        """Handle timeframe change from chart"""
        self.load_chart_data(self.current_chart_symbol, timeframe)

    def on_view_chart_requested(self, symbol: str):
        """Handle request to view chart for a symbol"""
        self.load_chart_data(symbol, self.current_timeframe)

        # Subscribe to WebSocket for this symbol
        self.subscribe_symbol(symbol)

    def subscribe_symbol(self, symbol: str):
        """Subscribe to real-time updates for a symbol"""
        if self.price_manager and symbol not in self.subscribed_symbols:
            self.price_manager.subscribe([symbol])
            self.subscribed_symbols.add(symbol)

    def subscribe_symbols(self, symbols: list):
        """Subscribe to real-time updates for multiple symbols"""
        if self.price_manager:
            new_symbols = [s for s in symbols if s not in self.subscribed_symbols]
            if new_symbols:
                self.price_manager.subscribe(new_symbols)
                self.subscribed_symbols.update(new_symbols)

    def on_price_update(self, symbol: str, price: float, bid: float, ask: float):
        """Handle real-time price update from WebSocket"""
        # Update position price in the table
        self.dashboard_tab.positions_table.update_price(symbol, price)

        # Update position monitor with price (for stop/target checking)
        if self.position_monitor:
            self.position_monitor.update_price(symbol, price)

        # Update trading panel if this is the current symbol
        if symbol == self.current_chart_symbol:
            self.dashboard_tab.trading_panel.set_price(price)

            # Update the price label on the chart
            chart = self.dashboard_tab.stock_chart
            if chart.data is not None and len(chart.data) > 0:
                # Update the last candle's close price
                last_time = chart.data[-1, 0]
                last_open = chart.data[-1, 1]
                last_high = max(chart.data[-1, 2], price)
                last_low = min(chart.data[-1, 3], price)
                last_vol = chart.data[-1, 5]
                chart.update_last_candle(last_time, last_open, last_high, last_low, price, last_vol)

        # Update SPY in market bar
        if symbol == "SPY":
            # Calculate change percent (would need previous close for accuracy)
            self.market_bar.update_spy(price, 0)

    def on_ai_prediction(self, prediction: dict):
        """Handle AI prediction result."""
        symbol = prediction.get('symbol', '')
        action = prediction.get('action', 'HOLD')
        confidence = prediction.get('confidence', 0)
        model = prediction.get('model', 'Unknown')

        # Update the prediction display
        self.dashboard_tab.update_prediction(prediction)

        # Log the prediction
        if action != 'HOLD':
            self.dashboard_tab.log_activity(
                f"AI Signal: {action} {symbol} ({confidence*100:.1f}% - {model})",
                "AI"
            )
        else:
            self.dashboard_tab.log_activity(
                f"AI Signal: HOLD {symbol} (no strong signal)",
                "AI"
            )

        # Update trading panel with AI-suggested risk parameters
        trading_panel = self.dashboard_tab.trading_panel
        if action in ['BUY', 'SELL'] and symbol == self.current_chart_symbol:
            # Store and display risk params
            ai_params = {
                'stop_loss_pct': prediction.get('stop_loss_pct', 0.02),
                'take_profit_pct': prediction.get('take_profit_pct', 0.06),
                'trailing_activation_pct': prediction.get('trailing_activation_pct', 0.015),
                'trailing_stop_pct': prediction.get('trailing_stop_pct', 0.01),
                'model': model
            }
            trading_panel.update_ai_params(ai_params)
        else:
            # Clear AI params on HOLD signal
            trading_panel.clear_ai_params()

    def on_ai_error(self, error: str):
        """Handle AI prediction error."""
        self.dashboard_tab.log_activity(f"AI Error: {error}", "ERROR")

    def run_ai_scan(self, symbol: str = None):
        """Run AI prediction for a symbol."""
        if not self.ai_worker:
            # Use sample prediction if AI not available
            prediction = create_sample_prediction(symbol or self.current_chart_symbol)
            self.on_ai_prediction(prediction)
            return

        # Set broker if not set
        if self.broker and not self.ai_worker.broker:
            self.ai_worker.set_broker(self.broker)

        # Run scan
        target_symbol = symbol or self.current_chart_symbol
        self.dashboard_tab.log_activity(f"Running AI scan for {target_symbol}...", "AI")
        self.ai_worker.scan_once(target_symbol)

    def start_ai_scanning(self, symbols: list = None):
        """Start continuous AI scanning."""
        if not self.ai_worker:
            return

        if self.broker:
            self.ai_worker.set_broker(self.broker)

        if symbols:
            self.ai_worker.set_symbols(symbols)
        else:
            # Scan current chart symbol and positions
            scan_symbols = [self.current_chart_symbol]
            self.ai_worker.set_symbols(scan_symbols)

        self.ai_worker.start()
        self.dashboard_tab.log_activity("AI scanning started", "AI")

    def stop_ai_scanning(self):
        """Stop continuous AI scanning."""
        if self.ai_worker and self.ai_worker.running:
            self.ai_worker.stop()
            self.dashboard_tab.log_activity("AI scanning stopped", "AI")

    # =========================================================================
    # Auto Trading Handlers
    # =========================================================================

    def on_stop_loss_triggered(self, symbol: str, price: float, reason: str):
        """Handle stop loss trigger."""
        self.dashboard_tab.log_decision(f"STOP LOSS: {symbol} @ ${price:.2f}")
        self.dashboard_tab.log_activity(f"Stop loss hit for {symbol}", "TRADE")
        QTimer.singleShot(1000, self.refresh_positions)

    def on_take_profit_triggered(self, symbol: str, price: float, reason: str):
        """Handle take profit trigger."""
        self.dashboard_tab.log_decision(f"TAKE PROFIT: {symbol} @ ${price:.2f}")
        self.dashboard_tab.log_activity(f"Take profit hit for {symbol}", "TRADE")
        QTimer.singleShot(1000, self.refresh_positions)

    def on_trailing_stop_triggered(self, symbol: str, price: float, reason: str):
        """Handle trailing stop trigger."""
        self.dashboard_tab.log_decision(f"TRAILING STOP: {symbol} @ ${price:.2f}")
        self.dashboard_tab.log_activity(f"Trailing stop hit for {symbol}", "TRADE")
        QTimer.singleShot(1000, self.refresh_positions)

    def on_trade_closed(self, result: dict):
        """Handle trade closed event."""
        symbol = result.get('symbol', '')
        pnl = result.get('pnl', 0)
        pnl_pct = result.get('pnl_pct', 0)
        reason = result.get('exit_reason', 'unknown')
        status = "WIN" if pnl > 0 else "LOSS"

        level = "TRADE" if pnl > 0 else "ERROR"
        self.dashboard_tab.log_activity(
            f"{status}: {symbol} ${pnl:+.2f} ({pnl_pct:+.1f}%) - {reason}",
            level
        )

        # Update stats display
        today_stats = self.trade_logger.get_today_stats()
        self.update_pnl(today_stats['total_pnl'])

        # Count positions from both traders
        total_positions = 0
        if self.day_trader:
            total_positions += len(self.day_trader.active_positions)
        if self.swing_trader:
            total_positions += len(self.swing_trader.active_positions)
        self.update_positions_count(total_positions, 4)

        # Update auto trading panel with combined stats
        if hasattr(self, 'auto_trading_tab') and hasattr(self.auto_trading_tab, 'auto_panel'):
            combined_stats = self._get_combined_trader_stats()
            self.auto_trading_tab.auto_panel.update_stats(combined_stats)

    def on_auto_trade_executed(self, trade_info: dict):
        """Handle auto trade execution."""
        symbol = trade_info.get('symbol', '')
        action = trade_info.get('action', 'BUY')
        shares = trade_info.get('shares', 0)
        price = trade_info.get('price', 0)
        model = trade_info.get('model', 'Unknown')

        self.dashboard_tab.log_decision(
            f"{action} {shares} {symbol} @ ${price:.2f} ({model})"
        )
        self.dashboard_tab.log_activity(
            f"Auto trade: {action} {shares} {symbol} @ ${price:.2f}",
            "TRADE"
        )

        # Subscribe to price updates for new position
        self.subscribe_symbol(symbol)

        # Update positions
        QTimer.singleShot(1000, self.refresh_positions)

    def on_signal_generated(self, signal: dict):
        """Handle AI signal generation."""
        symbol = signal.get('symbol', '')
        action = signal.get('action', 'HOLD')
        confidence = signal.get('confidence', 0)
        model = signal.get('model', 'Unknown')

        if action != 'HOLD':
            self.dashboard_tab.log_activity(
                f"Signal: {action} {symbol} ({confidence*100:.1f}% - {model})",
                "AI"
            )

    def on_trader_status(self, message: str):
        """Handle trader status update."""
        self.set_status_message(message)
        self.dashboard_tab.log_activity(message, "INFO")

        # Update auto trading panel log if exists
        if hasattr(self, 'auto_trading_tab') and hasattr(self.auto_trading_tab, 'auto_panel'):
            self.auto_trading_tab.auto_panel.add_log(message)

    def on_news_trade_signal(self, signal: dict):
        """Handle trade signal from news sentiment analysis."""
        symbol = signal.get('symbol', '')
        action = signal.get('action', 'BUY')
        confidence = signal.get('confidence', 0)
        source = signal.get('source', 'news')
        headline = signal.get('headline', '')[:50]

        self.dashboard_tab.log_activity(
            f"News signal: {action} {symbol} ({confidence*100:.0f}%) - {headline}...",
            "NEWS"
        )

        # Execute the trade if broker is available
        if self.broker and action in ('BUY', 'SELL'):
            try:
                from data.broker_adapter import OrderSide, OrderType

                # Get current price
                snapshot = self.broker.get_snapshot(symbol)
                if not snapshot:
                    print(f"[News] Could not get price for {symbol}")
                    return

                current_price = snapshot.get('price', 0)
                if current_price <= 0:
                    return

                # Calculate position size (5% of portfolio for news trades)
                account = self.broker.get_account()
                portfolio_value = float(account.equity)
                position_value = portfolio_value * 0.05
                shares = int(position_value / current_price)

                if shares <= 0:
                    return

                # Place order
                order_side = OrderSide.BUY if action == 'BUY' else OrderSide.SELL
                order = self.broker.place_order(
                    symbol=symbol,
                    qty=shares,
                    side=order_side,
                    type=OrderType.MARKET
                )

                if order:
                    self.dashboard_tab.log_activity(
                        f"News trade executed: {action} {shares} {symbol} @ ${current_price:.2f}",
                        "TRADE"
                    )

                    # Add to position monitor with default stops
                    if self.position_monitor and action == 'BUY':
                        stop_loss = current_price * 0.98  # 2% stop
                        take_profit = current_price * 1.05  # 5% profit
                        self.position_monitor.add_position(
                            symbol=symbol,
                            entry_price=current_price,
                            shares=shares,
                            stop_loss=stop_loss,
                            take_profit=take_profit,
                            ai_model='news_sentiment',
                            ai_confidence=confidence,
                            trade_type='news'
                        )

                    # Refresh positions
                    QTimer.singleShot(1000, self.refresh_positions)

            except Exception as e:
                print(f"[News] Trade execution error: {e}")
                self.dashboard_tab.log_activity(f"News trade error: {e}", "ERROR")

    def on_stats_updated(self, stats: dict):
        """Handle stats update from auto trader."""
        # Update status bar
        self.update_pnl(stats.get('pnl', 0))
        self.update_positions_count(stats.get('active_positions', 0), 4)

        # Update auto trading panel if exists
        if hasattr(self, 'auto_trading_tab') and hasattr(self.auto_trading_tab, 'auto_panel'):
            self.auto_trading_tab.auto_panel.update_stats(stats)

    def on_trader_error(self, error: str):
        """Handle trader error."""
        self.dashboard_tab.log_activity(f"Trader Error: {error}", "ERROR")

    def start_dual_trading(self):
        """Start BOTH day trading and swing trading simultaneously."""
        # Set broker on AI workers if not already set
        if self.broker:
            if self.day_ai_worker and not self.day_ai_worker.broker:
                self.day_ai_worker.set_broker(self.broker)
            if self.swing_ai_worker and not self.swing_ai_worker.broker:
                self.swing_ai_worker.set_broker(self.broker)

        # Start position monitor first (shared by both traders)
        if not self.position_monitor.isRunning():
            self.position_monitor.start()
            # Sync with existing broker positions so they get monitored
            self.position_monitor.sync_with_broker(default_stop_pct=0.02, default_profit_pct=0.05)

        # Configure and start Day Trader (5-minute candles)
        if self.day_trader:
            day_watchlist = DEFAULT_DAY_WATCHLIST[:10]  # 10 high-volume stocks for day trading
            self.day_trader.config.update({
                'max_positions': 2,
                'min_confidence': 0.70,
                'auto_execute': True,
                'stop_loss_pct': 0.02,      # 2% stop loss for day trading
                'take_profit_pct': 0.04     # 4% take profit for day trading
            })
            self.day_trader.set_watchlist(day_watchlist)
            self.day_trader.start()
            print(f"[DAY TRADER] Started with {len(day_watchlist)} symbols (5-min candles)")

        # Configure and start Swing Trader (daily candles)
        if self.swing_trader:
            swing_watchlist = DEFAULT_SWING_WATCHLIST[:10]  # 10 stocks for swing trading
            self.swing_trader.config.update({
                'max_positions': 2,
                'min_confidence': 0.70,
                'auto_execute': True,
                'stop_loss_pct': 0.05,      # 5% stop loss for swing trading
                'take_profit_pct': 0.12     # 12% take profit for swing trading
            })
            self.swing_trader.set_watchlist(swing_watchlist)
            self.swing_trader.start()
            print(f"[SWING TRADER] Started with {len(swing_watchlist)} symbols (daily candles)")

        self.dashboard_tab.log_activity(
            "DUAL MODE started: Day (5-min, 2% SL) + Swing (daily, 5% SL)",
            "TRADE"
        )

    def start_auto_trading(self, mode: str, watchlist: list):
        """Start automated trading (single mode - legacy support)."""
        # Start position monitor
        if not self.position_monitor.isRunning():
            self.position_monitor.start()
            self.position_monitor.sync_with_broker(default_stop_pct=0.02, default_profit_pct=0.05)

        # Configure based on mode
        config = {
            'max_positions': 2,
            'min_confidence': 0.70,
            'auto_execute': True
        }

        if mode == 'day' and self.day_trader:
            self.day_trader.config.update(config)
            self.day_trader.set_watchlist(watchlist)
            self.day_trader.start()
        elif mode == 'swing' and self.swing_trader:
            self.swing_trader.config.update(config)
            self.swing_trader.set_watchlist(watchlist)
            self.swing_trader.start()
        elif self.auto_trader:
            # Legacy fallback
            self.auto_trader.config.update(config)
            self.auto_trader.set_mode(mode)
            self.auto_trader.set_watchlist(watchlist)
            self.auto_trader.start()

        self.dashboard_tab.log_activity(
            f"Auto trading started ({mode.upper()} mode, {len(watchlist)} symbols)",
            "TRADE"
        )

    def stop_auto_trading(self):
        """Stop all automated trading (day + swing)."""
        if self.day_trader:
            self.day_trader.stop()
        if self.swing_trader:
            self.swing_trader.stop()
        if self.auto_trader:
            self.auto_trader.stop()
        self.dashboard_tab.log_activity("All auto trading stopped", "TRADE")

    def pause_auto_trading(self):
        """Pause all automated trading."""
        if self.day_trader:
            self.day_trader.pause()
        if self.swing_trader:
            self.swing_trader.pause()
        if self.auto_trader:
            self.auto_trader.pause()
        self.dashboard_tab.log_activity("All auto trading paused", "TRADE")

    def resume_auto_trading(self):
        """Resume all automated trading."""
        if self.day_trader:
            self.day_trader.resume()
        if self.swing_trader:
            self.swing_trader.resume()
        if self.auto_trader:
            self.auto_trader.resume()
        self.dashboard_tab.log_activity("All auto trading resumed", "TRADE")

    def close_all_positions(self):
        """Close all positions from all traders."""
        if self.day_trader:
            self.day_trader.close_all_positions()
        if self.swing_trader:
            self.swing_trader.close_all_positions()
        if self.auto_trader:
            self.auto_trader.close_all_positions()
        self.dashboard_tab.log_activity("Closing all positions...", "TRADE")

    def _get_combined_trader_stats(self) -> dict:
        """Get combined stats from both day and swing traders."""
        combined = {
            'signals_generated': 0,
            'trades_executed': 0,
            'positions_open': 0,
            'total_pnl': 0.0,
            'win_rate': 0.0,
            'day_signals': 0,
            'day_trades': 0,
            'swing_signals': 0,
            'swing_trades': 0
        }

        wins = 0
        total_trades = 0

        if self.day_trader:
            day_stats = self.day_trader.get_stats()
            combined['day_signals'] = day_stats.get('signals_generated', 0)
            combined['day_trades'] = day_stats.get('trades_executed', 0)
            combined['signals_generated'] += day_stats.get('signals_generated', 0)
            combined['trades_executed'] += day_stats.get('trades_executed', 0)
            combined['positions_open'] += len(self.day_trader.active_positions)
            combined['total_pnl'] += day_stats.get('total_pnl', 0)
            wins += day_stats.get('winning_trades', 0)
            total_trades += day_stats.get('total_closed_trades', 0)

        if self.swing_trader:
            swing_stats = self.swing_trader.get_stats()
            combined['swing_signals'] = swing_stats.get('signals_generated', 0)
            combined['swing_trades'] = swing_stats.get('trades_executed', 0)
            combined['signals_generated'] += swing_stats.get('signals_generated', 0)
            combined['trades_executed'] += swing_stats.get('trades_executed', 0)
            combined['positions_open'] += len(self.swing_trader.active_positions)
            combined['total_pnl'] += swing_stats.get('total_pnl', 0)
            wins += swing_stats.get('winning_trades', 0)
            total_trades += swing_stats.get('total_closed_trades', 0)

        if total_trades > 0:
            combined['win_rate'] = wins / total_trades

        return combined

    def toggle_ai_terminal(self, checked: bool = None):
        """Toggle the AI Assistant terminal panel."""
        if checked is None:
            checked = not self.ai_terminal_dock.isVisible()

        if checked:
            self.ai_terminal_dock.show()
            # Update AI terminal with current trading context
            self.update_ai_terminal_context()
        else:
            self.ai_terminal_dock.hide()

        self.ai_assistant_action.setChecked(checked)

    def update_ai_terminal_context(self):
        """Update the AI terminal with current trading context."""
        if not hasattr(self, 'ai_terminal_dock'):
            return

        try:
            # Build context from current state
            context = {}

            # Account info
            if self.broker:
                try:
                    account = self.broker.get_account()
                    context['account'] = {
                        'equity': float(account.equity),
                        'cash': float(account.cash),
                        'day_pnl': float(account.equity) - float(account.last_equity)
                    }
                except Exception:
                    pass

            # Current positions
            if self.broker:
                try:
                    positions = self.broker.get_positions()
                    context['positions'] = []
                    for pos in positions:
                        context['positions'].append({
                            'symbol': pos.symbol,
                            'qty': float(pos.qty),
                            'entry': float(pos.avg_entry_price),
                            'current': float(pos.current_price),
                            'pnl': float(pos.unrealized_pl),
                            'pnl_pct': float(pos.unrealized_plpc) * 100
                        })
                except Exception:
                    pass

            # Model performance
            if AI_AVAILABLE:
                brain = get_ai_brain()
                if brain and brain.loaded:
                    top_models = list(brain.model_priority[:3])
                    context['model_performance'] = ', '.join(top_models)

            self.ai_terminal_dock.update_trading_context(context)

        except Exception as e:
            print(f"[AI Terminal] Error updating context: {e}")

    def on_settings_changed(self, settings: dict):
        """Handle settings changes from the Settings tab."""
        trading = settings.get('trading', {})

        # Apply trading settings to day trader
        if self.day_trader and trading.get('day_trading_enabled', True):
            self.day_trader.config['max_positions'] = trading.get('max_positions', 4)
            self.day_trader.config['min_confidence'] = trading.get('min_confidence', 0.70)
            self.day_trader.config['stop_loss_pct'] = trading.get('day_stop_loss_pct', 0.02)
            self.day_trader.config['take_profit_pct'] = trading.get('day_take_profit_pct', 0.04)

        # Apply trading settings to swing trader
        if self.swing_trader and trading.get('swing_trading_enabled', True):
            self.swing_trader.config['max_positions'] = trading.get('max_positions', 4)
            self.swing_trader.config['min_confidence'] = trading.get('min_confidence', 0.70)
            self.swing_trader.config['stop_loss_pct'] = trading.get('swing_stop_loss_pct', 0.05)
            self.swing_trader.config['take_profit_pct'] = trading.get('swing_take_profit_pct', 0.12)

        # Enable/disable traders
        if not trading.get('day_trading_enabled', True) and self.day_trader:
            self.day_trader.pause()
        elif trading.get('day_trading_enabled', True) and self.day_trader:
            self.day_trader.resume()

        if not trading.get('swing_trading_enabled', True) and self.swing_trader:
            self.swing_trader.pause()
        elif trading.get('swing_trading_enabled', True) and self.swing_trader:
            self.swing_trader.resume()

        print("[Settings] Trading configuration updated")
        self.dashboard_tab.log_activity("Settings updated and applied", "INFO")
