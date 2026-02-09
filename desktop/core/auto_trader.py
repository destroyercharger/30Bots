"""
Auto Trader - Automated trading using AI models
"""

from typing import Dict, List, Optional
from datetime import datetime, time
from PyQt6.QtCore import QThread, pyqtSignal, QMutex, QTimer
import sys
from pathlib import Path

# Add parent for AI imports
PARENT_DIR = Path(__file__).parent.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from .trade_logger import get_trade_logger
from .position_monitor import PositionMonitor
from .market_data import get_market_data_calculator, ORBLevels

# Import Live Performance Tracker for position scaling
try:
    from .live_performance_tracker import get_performance_tracker, LivePerformanceTracker
    PERFORMANCE_TRACKER_AVAILABLE = True
except ImportError:
    PERFORMANCE_TRACKER_AVAILABLE = False
    print("[AutoTrader] Live Performance Tracker not available")

# Performance scaling log file
SCALING_LOG_PATH = Path(__file__).parent.parent / "data" / "performance_scaling.log"

# Import Paper Broker for paper trading mode
try:
    from data.paper_broker import get_paper_broker, PaperBroker
    PAPER_BROKER_AVAILABLE = True
except ImportError:
    PAPER_BROKER_AVAILABLE = False
    print("[AutoTrader] Paper Broker not available")

# Import Model Learner for adaptive model weights
try:
    from ai.model_learner import get_model_learner
    MODEL_LEARNER_AVAILABLE = True
except ImportError:
    MODEL_LEARNER_AVAILABLE = False
    print("[AutoTrader] Model Learner not available")

# Import Trade Learner for learning from trade outcomes
try:
    from ai.trade_learner import get_trade_learner
    TRADE_LEARNER_AVAILABLE = True
except ImportError:
    TRADE_LEARNER_AVAILABLE = False
    print("[AutoTrader] Trade Learner not available")

# Import ALLOW_SHORTING from config
try:
    from config import ALLOW_SHORTING
except ImportError:
    ALLOW_SHORTING = True  # Default if config not found

# Import Unified Brain for multi-model trade confirmation
try:
    from ai.unified_brain import get_unified_brain, UnifiedBrain
    from ai.regime_classifier import MarketRegime
    UNIFIED_BRAIN_AVAILABLE = True
except ImportError:
    UNIFIED_BRAIN_AVAILABLE = False
    UnifiedBrain = None
    MarketRegime = None
    print("[AutoTrader] Unified Brain not available")

# Legacy: Import RL Analyzer (fallback if unified brain not available)
try:
    from ai.rl_analyzer import get_rl_analyzer
    RL_ANALYZER_AVAILABLE = True
except ImportError:
    RL_ANALYZER_AVAILABLE = False
    print("[AutoTrader] RL Analyzer not available")

# Import specialized trading model brains (NEW!)
try:
    from ai.models import get_model_brain, analyze_symbol
    from ai.trading_types import TradingType
    SPECIALIZED_MODELS_AVAILABLE = True
except ImportError as e:
    SPECIALIZED_MODELS_AVAILABLE = False
    TradingType = None
    print(f"[AutoTrader] Specialized models not available: {e}")


class AutoTrader(QThread):
    """
    Automated trading engine using AI models.

    Modes:
    - Day Trading: Quick entries/exits, tighter stops
    - Swing Trading: Longer holds, wider stops

    Features:
    - Scans watchlist for AI signals
    - Executes trades automatically
    - Manages position sizing
    - Tracks performance
    """

    # Signals
    trade_executed = pyqtSignal(dict)  # Trade details
    signal_generated = pyqtSignal(dict)  # AI signal
    status_update = pyqtSignal(str)  # Status message
    stats_updated = pyqtSignal(dict)  # Performance stats
    error = pyqtSignal(str)

    def __init__(self, broker=None, parent=None):
        super().__init__(parent)
        self.broker = broker
        self.ai_worker = None
        self.position_monitor = None
        self.running = False
        self.paused = False
        self.mutex = QMutex()

        # =================================================================
        # RESEARCH-BACKED TRADING CONFIGURATION
        # Based on: Berkeley studies, SSRN ORB paper, professional risk mgmt
        # =================================================================

        self.config = {
            'mode': 'day',  # 'day' or 'swing'
            'max_positions': 3,  # Research: Focus on fewer, higher quality trades
            'risk_per_trade_pct': 0.01,  # THE 1% RULE - Never risk more than 1% per trade
            'min_confidence': 0.80,  # High confidence only - research shows this matters
            'scan_interval': 30000,  # 30 seconds
            'market_hours_only': True,
            'auto_execute': True,
            'allow_shorting': ALLOW_SHORTING,
            'use_unified_brain': True,
            'use_rl_confirmation': True,
            'require_confirmation': False,
            'respect_regime': True,

            # ANTI-OVERTRADING: Research shows overtrading = 7% worse returns
            'max_trades_per_day': 3,  # Limit trades to prevent overtrading
            'daily_loss_limit_pct': 0.02,  # Stop trading after 2% daily loss
            'min_relative_volume': 1.5,  # Only trade stocks with 1.5x normal volume
            'require_vwap_alignment': True,  # Only trade in VWAP direction

            # TIME FILTERS: Research shows best trading hours
            'trade_first_hour': True,  # 9:30-10:30 = highest volatility
            'avoid_lunch_hours': True,  # 12:00-2:00 = low volume, choppy
            'trade_power_hour': True,  # 3:00-4:00 = strong moves

            # OPENING RANGE BREAKOUT (ORB): Research shows 1,600% returns
            'use_orb_strategy': True,  # Enable ORB signals
            'orb_minutes': 30,  # First 30 minutes define the range
            'orb_confirmation_required': False,  # Require AI confirmation for ORB

            # PAPER TRADING: Safe testing without real money
            'paper_trading': False,  # Enable paper trading mode
            'paper_starting_cash': 100000,  # Starting cash for paper trading

            # VIX BLOCK FILTER: Prevent trades during high volatility
            # Based on pattern learning: "volatility_spike" is #1 failure (16 occurrences)
            'use_vix_block': True,  # Enable VIX blocking
            'vix_block_threshold': 28.0,  # Block new entries when VIX > 28
            'vix_warning_threshold': 25.0,  # Log warning when VIX > 25

            # TREND CONFIRMATION FILTER: Prevent coin-flip trades
            # Based on pattern learning: "wrong_direction" is #2 failure (4 occurrences)
            # All wrong_direction failures had neutral indicators (RSI=50, MACD=neutral, SMA=neutral)
            'use_trend_confirmation': True,  # Enable trend confirmation
            'trend_min_signals': 1,  # Require at least 1 confirming signal
            'trend_rsi_threshold': 10,  # RSI must be > 10 points from 50 to count
        }

        # VIX tracking
        self.current_vix = 20.0
        self.vix_last_update = None

        # Paper trading mode
        self.paper_mode = False
        self.paper_broker = None

        # Day trading settings - Research: 1:3 Risk/Reward minimum
        # With 1:3 R:R, only need 30% win rate to be profitable
        self.day_settings = {
            'stop_loss_pct': 0.01,  # 1% stop (tight, as research suggests)
            'take_profit_pct': 0.03,  # 3% target (1:3 ratio)
            'trailing_activation_pct': 0.015,  # Trail after 1.5% profit
            'trailing_stop_pct': 0.007,  # Tight trailing to lock profits
        }

        # Swing trading settings - WIDER STOPS for multi-day holds
        # Swing trades should be held for days/weeks, need room to breathe
        self.swing_settings = {
            'stop_loss_pct': 0.05,  # 5% stop (wider for daily volatility)
            'take_profit_pct': 0.15,  # 15% target (1:3 ratio)
            'trailing_activation_pct': 0.06,  # Trail after 6% profit
            'trailing_stop_pct': 0.03,  # 3% trailing (gives room to run)
        }

        # Crypto settings - WIDER STOPS for high volatility
        # Crypto can move 5-10% in a day, need proper room
        self.crypto_settings = {
            'stop_loss_pct': 0.05,  # 5% stop for crypto (was 2% - way too tight!)
            'take_profit_pct': 0.15,  # 15% target (1:3 ratio)
            'trailing_activation_pct': 0.06,  # Trail after 6% profit
            'trailing_stop_pct': 0.03,  # 3% trailing stop
            'max_risk_pct': 0.005,  # Only 0.5% risk on crypto (half normal)
        }

        # Daily tracking for loss limits and trade counts
        self.daily_stats = {
            'trades_today': 0,
            'pnl_today': 0.0,
            'last_reset': None,
        }

        # Watchlist
        self.watchlist: List[str] = []

        # Current positions (symbol -> position data)
        self.active_positions: Dict[str, Dict] = {}

        # Trade logger
        self.trade_logger = get_trade_logger()

        # Live Performance Tracker for position scaling based on live P&L
        self.performance_tracker = None
        if PERFORMANCE_TRACKER_AVAILABLE:
            try:
                self.performance_tracker = get_performance_tracker()
                print("[AutoTrader] Live Performance Tracker loaded - position scaling enabled")
            except Exception as e:
                print(f"[AutoTrader] Live Performance Tracker error: {e}")

        # Model Learner for adaptive weights based on performance
        self.model_learner = None
        if MODEL_LEARNER_AVAILABLE:
            try:
                self.model_learner = get_model_learner()
                print("[AutoTrader] Model Learner loaded - adaptive weights enabled")
            except Exception as e:
                print(f"[AutoTrader] Model Learner error: {e}")

        # Trade Learner for learning from WHY trades fail
        self.trade_learner = None
        if TRADE_LEARNER_AVAILABLE:
            try:
                self.trade_learner = get_trade_learner()
                stats = self.trade_learner.get_learning_stats()
                print(f"[AutoTrader] Trade Learner loaded - {stats.get('losing_patterns', 0)} losing patterns to avoid")
            except Exception as e:
                print(f"[AutoTrader] Trade Learner error: {e}")

        # Market Data Calculator for VWAP, relative volume, ORB
        self.market_data = get_market_data_calculator()
        print("[AutoTrader] Market Data Calculator loaded (VWAP, RelVol, ORB)")

        # Unified Brain for multi-model trade confirmation
        self.unified_brain = None
        self.rl_analyzer = None  # Legacy fallback
        self.current_regime = None

        if UNIFIED_BRAIN_AVAILABLE and self.config.get('use_unified_brain', True):
            try:
                self.unified_brain = get_unified_brain(verbose=True)
                print("[AutoTrader] Unified Brain loaded (all analyzers)")
                # Get initial analyzer status
                status = self.unified_brain.get_status()
                loaded = [k for k, v in status.items() if v.get('loaded', False)]
                print(f"[AutoTrader] Active analyzers: {', '.join(loaded)}")
            except Exception as e:
                print(f"[AutoTrader] Unified Brain error: {e}")
                self.unified_brain = None

        # Fallback to RL-only if unified brain not available
        if self.unified_brain is None and RL_ANALYZER_AVAILABLE and self.config.get('use_rl_confirmation', True):
            try:
                self.rl_analyzer = get_rl_analyzer(verbose=True)
                if self.rl_analyzer and self.rl_analyzer.loaded:
                    print("[AutoTrader] RL Analyzer loaded (fallback mode)")
                else:
                    print("[AutoTrader] RL Analyzer failed to load")
                    self.rl_analyzer = None
            except Exception as e:
                print(f"[AutoTrader] RL Analyzer error: {e}")
                self.rl_analyzer = None

        # Specialized Trading Model Brains (NEW!)
        # These are trading-type-specific 10-model ensembles
        self.day_trading_brain = None
        self.swing_trading_brain = None
        self.use_specialized_models = True  # Enable by default

        if SPECIALIZED_MODELS_AVAILABLE:
            try:
                # Initialize both brains - we switch between them based on mode
                self.day_trading_brain = get_model_brain(TradingType.DAY_TRADING)
                print("[AutoTrader] Day Trading Brain loaded (10 intraday models)")

                self.swing_trading_brain = get_model_brain(TradingType.SWING_TRADING)
                print("[AutoTrader] Swing Trading Brain loaded (10 multi-day models)")
            except Exception as e:
                print(f"[AutoTrader] Specialized model error: {e}")
                self.use_specialized_models = False

        # Statistics
        self.session_stats = {
            'trades': 0,
            'wins': 0,
            'losses': 0,
            'pnl': 0.0,
            'signals': 0
        }

    def set_broker(self, broker):
        """Set the broker for trading."""
        self.broker = broker
        if self.position_monitor:
            self.position_monitor.set_broker(broker)

    def set_ai_worker(self, ai_worker):
        """Set the AI prediction worker."""
        self.ai_worker = ai_worker
        # Sync trading mode with AI worker
        if ai_worker:
            ai_worker.set_trading_mode(self.config['mode'])

    def set_position_monitor(self, monitor: PositionMonitor):
        """Set the position monitor."""
        self.position_monitor = monitor
        monitor.trade_closed.connect(self.on_trade_closed)

    def set_mode(self, mode: str):
        """Set trading mode (day/swing)."""
        self.config['mode'] = mode
        # Update AI worker timeframe based on mode
        if self.ai_worker:
            self.ai_worker.set_trading_mode(mode)
        self.status_update.emit(f"Trading mode: {mode.upper()}")

    def set_allow_shorting(self, allow: bool):
        """Enable or disable short selling."""
        self.config['allow_shorting'] = allow
        status = "enabled" if allow else "disabled"
        print(f"[AutoTrader] Shorting {status}")
        self.status_update.emit(f"Shorting {status}")

    def set_rl_confirmation(self, enabled: bool):
        """Enable or disable RL confirmation for trades."""
        self.config['use_rl_confirmation'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"[AutoTrader] RL confirmation {status}")
        self.status_update.emit(f"RL confirmation {status}")

    def set_require_rl_agreement(self, required: bool):
        """Set whether RL must agree with XGBoost for trades to execute."""
        self.config['require_rl_agreement'] = required
        status = "required" if required else "optional"
        print(f"[AutoTrader] RL agreement {status}")
        self.status_update.emit(f"RL agreement {status}")

    def get_rl_status(self) -> Dict:
        """Get RL analyzer status (legacy)."""
        if self.rl_analyzer and self.rl_analyzer.loaded:
            return {
                'loaded': True,
                'enabled': self.config.get('use_rl_confirmation', True),
                'required': self.config.get('require_confirmation', False)
            }
        return {'loaded': False, 'enabled': False, 'required': False}

    def get_brain_status(self) -> Dict:
        """Get unified brain status."""
        if self.unified_brain:
            analyzer_status = self.unified_brain.get_status()
            return {
                'loaded': True,
                'enabled': self.config.get('use_unified_brain', True),
                'require_confirmation': self.config.get('require_confirmation', False),
                'respect_regime': self.config.get('respect_regime', True),
                'current_regime': self.current_regime.value if self.current_regime else 'UNKNOWN',
                'analyzers': analyzer_status
            }
        elif self.rl_analyzer and self.rl_analyzer.loaded:
            return {
                'loaded': True,
                'enabled': True,
                'mode': 'RL-only (fallback)',
                'analyzers': {'rl': {'loaded': True}}
            }
        return {'loaded': False, 'enabled': False, 'analyzers': {}}

    def set_unified_brain(self, enabled: bool):
        """Enable or disable unified brain confirmation."""
        self.config['use_unified_brain'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"[AutoTrader] Unified Brain {status}")
        self.status_update.emit(f"Unified Brain {status}")

    def set_respect_regime(self, enabled: bool):
        """Enable or disable market regime filtering."""
        self.config['respect_regime'] = enabled
        status = "enabled" if enabled else "disabled"
        print(f"[AutoTrader] Regime filtering {status}")
        self.status_update.emit(f"Regime filtering {status}")

    def set_paper_trading(self, enabled: bool, starting_cash: float = None):
        """
        Enable or disable paper trading mode.

        Paper trading simulates trades without using real money.
        Great for testing strategies before going live.
        """
        if enabled and not PAPER_BROKER_AVAILABLE:
            print("[AutoTrader] Paper trading not available")
            self.error.emit("Paper trading module not available")
            return

        if enabled:
            cash = starting_cash or self.config.get('paper_starting_cash', 100000)
            self.paper_broker = get_paper_broker(starting_cash=cash)
            self.paper_mode = True
            self.config['paper_trading'] = True

            # Switch to paper broker
            self._original_broker = self.broker
            self.broker = self.paper_broker

            print(f"[AutoTrader] PAPER TRADING ENABLED (${cash:,.0f} starting cash)")
            self.status_update.emit(f"Paper trading: ${cash:,.0f}")
        else:
            self.paper_mode = False
            self.config['paper_trading'] = False

            # Restore real broker
            if hasattr(self, '_original_broker') and self._original_broker:
                self.broker = self._original_broker

            print("[AutoTrader] Paper trading DISABLED - using real broker")
            self.status_update.emit("Live trading mode")

    def get_paper_stats(self) -> Dict:
        """Get paper trading performance stats."""
        if not self.paper_mode or not self.paper_broker:
            return {'enabled': False}

        summary = self.paper_broker.get_performance_summary()
        summary['enabled'] = True
        return summary

    def reset_paper_trading(self):
        """Reset paper trading to initial state."""
        if self.paper_broker:
            self.paper_broker.reset()
            print("[AutoTrader] Paper trading reset")
            self.status_update.emit("Paper trading reset")

    def set_watchlist(self, symbols: List[str]):
        """Set watchlist of symbols to scan."""
        self.mutex.lock()
        self.watchlist = list(symbols)
        self.mutex.unlock()
        self.status_update.emit(f"Watchlist: {len(symbols)} symbols")

    def add_to_watchlist(self, symbol: str):
        """Add symbol to watchlist."""
        self.mutex.lock()
        if symbol not in self.watchlist:
            self.watchlist.append(symbol)
        self.mutex.unlock()

    def is_market_open(self) -> bool:
        """Check if market is open for trading."""
        if not self.config['market_hours_only']:
            return True

        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo

        eastern = ZoneInfo("America/New_York")
        now = datetime.now(eastern)

        # Weekend check
        if now.weekday() >= 5:
            return False

        # Market hours: 9:30 AM - 4:00 PM ET
        current_time = now.time()
        market_open = time(9, 30)
        market_close = time(16, 0)

        return market_open <= current_time <= market_close

    def get_risk_settings(self, symbol: str = None) -> Dict:
        """Get current risk settings based on mode and asset type."""
        # Check if it's a crypto symbol
        if symbol and ('USD' in symbol.upper() and len(symbol) <= 10):
            crypto_symbols = ['BTCUSD', 'ETHUSD', 'SOLUSD', 'LINKUSD', 'AVAXUSD', 'DOGEUSD', 'XRPUSD']
            if symbol.upper() in crypto_symbols or symbol.upper().replace('/', '') in crypto_symbols:
                return self.crypto_settings

        if self.config['mode'] == 'swing':
            return self.swing_settings
        return self.day_settings

    def calculate_position_size(self, price: float, symbol: str = None,
                                 stop_loss_pct: float = None, model_name: str = None) -> int:
        """
        Calculate position size using THE 1% RISK RULE with LIVE PERFORMANCE SCALING.

        Research-backed formula:
        Position Size = (Account × Risk %) ÷ Stop Loss Distance × Performance Multiplier

        This ensures we never lose more than 1% of account on any single trade,
        regardless of price or volatility.

        NEW: Performance scaling reduces position size for underperforming models:
        - Profitable models: up to 1.5x position size
        - Losing models: down to 0.25x position size (never fully disabled)
        - Unknown models: 0.5x (cautious start)
        """
        if not self.broker:
            return 0

        try:
            account = self.broker.get_account()
            portfolio_value = float(account.equity)

            # Get risk settings for this asset type
            risk_settings = self.get_risk_settings(symbol)

            # Use crypto-specific risk if applicable
            if 'max_risk_pct' in risk_settings:
                risk_pct = risk_settings['max_risk_pct']  # 0.5% for crypto
            else:
                risk_pct = self.config['risk_per_trade_pct']  # 1% standard

            # Get stop loss percentage
            if stop_loss_pct is None:
                stop_loss_pct = risk_settings['stop_loss_pct']

            # THE 1% RULE FORMULA:
            # Max $ we can lose = Portfolio × Risk %
            # Stop distance in $ = Price × Stop Loss %
            # Shares = Max loss / Stop distance
            max_loss_dollars = portfolio_value * risk_pct
            stop_distance_dollars = price * stop_loss_pct

            if stop_distance_dollars <= 0:
                return 0

            shares = int(max_loss_dollars / stop_distance_dollars)

            # ===============================================================
            # LIVE PERFORMANCE SCALING (NEW!)
            # Scale position size based on model's live trading performance
            # ===============================================================
            performance_multiplier = 1.0
            performance_reason = ""
            if self.performance_tracker and model_name:
                should_reduce, multiplier, reason = self.performance_tracker.should_reduce_position(model_name)
                performance_multiplier = multiplier
                performance_reason = reason
                original_shares = shares

                if multiplier != 1.0:
                    shares = int(shares * multiplier)
                    timestamp = datetime.now().strftime("%Y-%m-%d %H:%M:%S")

                    # Log with clear indicators based on scaling direction
                    if multiplier < 0.5:
                        # Severe reduction - model is really struggling
                        log_msg = f"⚠️  STRUGGLING | {model_name} | {multiplier:.0%} | {original_shares}→{shares} shares | {reason}"
                        print(f"[AutoTrader] ⚠️  POSITION REDUCED (STRUGGLING MODEL): {model_name}")
                        print(f"[AutoTrader]     Reason: {reason}")
                        print(f"[AutoTrader]     Shares: {original_shares} → {shares} ({multiplier:.0%} of normal)")
                        self.status_update.emit(f"⚠️ {model_name}: Position reduced to {multiplier:.0%} - {reason}")
                    elif multiplier < 0.8:
                        # Moderate reduction - model underperforming
                        log_msg = f"📉 UNDERPERFORM | {model_name} | {multiplier:.0%} | {original_shares}→{shares} shares | {reason}"
                        print(f"[AutoTrader] 📉 POSITION REDUCED (UNDERPERFORMING): {model_name}")
                        print(f"[AutoTrader]     Reason: {reason}")
                        print(f"[AutoTrader]     Shares: {original_shares} → {shares} ({multiplier:.0%} of normal)")
                        self.status_update.emit(f"📉 {model_name}: Position reduced to {multiplier:.0%}")
                    elif multiplier < 1.0:
                        # Slight reduction
                        log_msg = f"📊 ADJUSTED | {model_name} | {multiplier:.0%} | {original_shares}→{shares} shares | {reason}"
                        print(f"[AutoTrader] 📊 Position adjusted: {model_name} ({multiplier:.0%})")
                    else:
                        # Boosted position - model outperforming
                        log_msg = f"📈 BOOSTED | {model_name} | {multiplier:.0%} | {original_shares}→{shares} shares | {reason}"
                        print(f"[AutoTrader] 📈 POSITION BOOSTED (OUTPERFORMING): {model_name}")
                        print(f"[AutoTrader]     Reason: {reason}")
                        print(f"[AutoTrader]     Shares: {original_shares} → {shares} ({multiplier:.0%} of normal)")
                        self.status_update.emit(f"📈 {model_name}: Position boosted to {multiplier:.0%}")

                    # Write to log file
                    try:
                        SCALING_LOG_PATH.parent.mkdir(parents=True, exist_ok=True)
                        with open(SCALING_LOG_PATH, 'a') as f:
                            f.write(f"{timestamp} | {symbol} | {log_msg}\n")
                    except Exception as e:
                        print(f"[AutoTrader] Could not write to scaling log: {e}")

            # Sanity check: cap at 10% of portfolio regardless
            max_position_value = portfolio_value * 0.10
            max_shares_by_value = int(max_position_value / price)
            shares = min(shares, max_shares_by_value)

            print(f"[AutoTrader] Position sizing: ${portfolio_value:,.0f} × {risk_pct*100}% risk = ${max_loss_dollars:.0f} max loss")
            print(f"[AutoTrader] Stop: {stop_loss_pct*100}% = ${stop_distance_dollars:.2f}/share → {shares} shares")
            if performance_multiplier != 1.0:
                print(f"[AutoTrader] Performance multiplier: {performance_multiplier:.2f}x ({performance_reason[:50]})")

            return max(1, shares) if shares > 0 else 0
        except Exception as e:
            self.error.emit(f"Position size error: {e}")
            return 0

    def can_open_position(self) -> bool:
        """Check if we can open a new position based on ACTUAL broker positions."""
        # Check actual broker positions, not just internally tracked ones
        if self.broker:
            try:
                actual_positions = self.broker.get_positions()
                num_positions = len(actual_positions)
                if num_positions >= self.config['max_positions']:
                    return False
            except Exception as e:
                print(f"[AutoTrader] Error checking positions: {e}")
                return False  # Be conservative on error

        return len(self.active_positions) < self.config['max_positions']

    # =================================================================
    # RESEARCH-BACKED FILTERS (Anti-Overtrading, Loss Limits, etc.)
    # =================================================================

    def reset_daily_stats(self):
        """Reset daily stats at market open."""
        from datetime import date
        today = date.today()
        if self.daily_stats['last_reset'] != today:
            self.daily_stats = {
                'trades_today': 0,
                'pnl_today': 0.0,
                'last_reset': today,
            }
            print(f"[AutoTrader] Daily stats reset for {today}")

    def record_trade(self, pnl: float = 0):
        """Record a trade for daily tracking."""
        self.reset_daily_stats()  # Ensure we're on current day
        self.daily_stats['trades_today'] += 1
        self.daily_stats['pnl_today'] += pnl

    def check_daily_limits(self) -> tuple:
        """
        Check if we've hit daily limits. Returns (can_trade, reason).
        Research: Daily loss limits prevent revenge trading spirals.
        """
        self.reset_daily_stats()

        # Check trade count limit
        max_trades = self.config.get('max_trades_per_day', 3)
        if self.daily_stats['trades_today'] >= max_trades:
            return False, f"Max trades reached ({max_trades}/day)"

        # Check daily loss limit
        if self.broker:
            try:
                account = self.broker.get_account()
                portfolio = float(account.equity)
                loss_limit = portfolio * self.config.get('daily_loss_limit_pct', 0.02)

                if self.daily_stats['pnl_today'] <= -loss_limit:
                    return False, f"Daily loss limit hit (${-loss_limit:.0f})"
            except Exception:
                pass

        return True, "OK"

    def is_good_trading_time(self) -> tuple:
        """
        Check if current time is good for trading.
        Research: Best times are 9:30-10:30 (open volatility) and 3:00-4:00 (power hour).
        Avoid: 12:00-2:00 (lunch = low volume, choppy).
        """
        from datetime import datetime
        try:
            from zoneinfo import ZoneInfo
        except ImportError:
            from backports.zoneinfo import ZoneInfo

        now = datetime.now(ZoneInfo('America/New_York'))
        hour = now.hour
        minute = now.minute
        time_decimal = hour + minute / 60

        # Market hours: 9:30 AM - 4:00 PM ET
        if time_decimal < 9.5 or time_decimal >= 16:
            return False, "Market closed"

        # First hour: 9:30-10:30 (best volatility)
        if 9.5 <= time_decimal < 10.5:
            if self.config.get('trade_first_hour', True):
                return True, "First hour - high volatility"

        # Lunch hours: 12:00-14:00 (avoid - choppy, low volume)
        if 12 <= time_decimal < 14:
            if self.config.get('avoid_lunch_hours', True):
                return False, "Lunch hours - low volume"

        # Power hour: 15:00-16:00 (strong directional moves)
        if 15 <= time_decimal < 16:
            if self.config.get('trade_power_hour', True):
                return True, "Power hour - strong moves"

        # Mid-day: 10:30-12:00 and 14:00-15:00 (moderate)
        return True, "Regular hours"

    def check_volume_filter(self, symbol: str, current_volume: int = None) -> tuple:
        """
        Check if stock has sufficient volume (is 'in play').
        Research: Only trade stocks with 1.5x+ normal volume.
        Uses real relative volume calculation vs 20-day average.
        """
        min_rel_volume = self.config.get('min_relative_volume', 1.5)

        # Use market data calculator for real relative volume
        rel_vol, description = self.market_data.get_relative_volume(symbol, self.broker)

        if rel_vol <= 0:
            # Couldn't calculate - allow trade but note it
            return True, "Volume data unavailable"

        if rel_vol < min_rel_volume:
            return False, f"Low relative volume ({rel_vol:.1f}x < {min_rel_volume}x required)"

        return True, f"Volume {rel_vol:.1f}x average (in play)"

    def check_vwap_alignment(self, symbol: str, action: str, current_price: float, vwap: float = None) -> tuple:
        """
        Check if trade direction aligns with VWAP.
        Research: Trade long above VWAP, short below VWAP.
        Uses real VWAP calculation from intraday data.
        """
        if not self.config.get('require_vwap_alignment', True):
            return True, "VWAP check disabled"

        # Fetch real VWAP if not provided
        if vwap is None or vwap <= 0:
            vwap = self.market_data.get_vwap(symbol, self.broker)

        if vwap is None or vwap <= 0:
            return True, "VWAP data unavailable"

        if action == 'BUY' and current_price < vwap:
            return False, f"BUY below VWAP (${current_price:.2f} < ${vwap:.2f})"

        if action == 'SELL' and current_price > vwap:
            return False, f"SELL above VWAP (${current_price:.2f} > ${vwap:.2f})"

        return True, f"VWAP aligned (${vwap:.2f})"

    def check_orb_breakout(self, symbol: str, current_price: float) -> tuple:
        """
        Check for Opening Range Breakout signal.
        Research: ORB strategy showed 1,600% returns (SSRN 2016-2023).

        Returns:
            Tuple of (signal: 'BUY'|'SELL'|'NONE', ORBLevels or None, reason)
        """
        if not self.config.get('use_orb_strategy', True):
            return 'NONE', None, "ORB disabled"

        # Only check after opening range is established (30+ minutes into session)
        good_time, time_reason = self.is_good_trading_time()
        if not good_time and "closed" in time_reason.lower():
            return 'NONE', None, "Market closed"

        # Get ORB levels
        orb = self.market_data.get_orb_levels(symbol)

        if orb is None:
            return 'NONE', None, "ORB levels not available"

        # Check for breakout
        if current_price > orb.breakout_long:
            return 'BUY', orb, f"ORB breakout LONG (>${orb.high:.2f})"

        if current_price < orb.breakout_short:
            return 'SELL', orb, f"ORB breakout SHORT (<${orb.low:.2f})"

        return 'NONE', orb, f"Inside ORB range (${orb.low:.2f}-${orb.high:.2f})"

    def generate_orb_signal(self, symbol: str) -> Optional[Dict]:
        """
        Generate an ORB-based trading signal if breakout detected.

        Returns:
            Signal dict or None
        """
        if not self.broker:
            return None

        try:
            # Get current price
            snapshot = self.broker.get_snapshot(symbol)
            if not snapshot:
                return None

            current_price = 0
            if 'latestTrade' in snapshot and snapshot['latestTrade']:
                current_price = snapshot['latestTrade'].get('p', 0)
            elif 'latestQuote' in snapshot and snapshot['latestQuote']:
                bid = snapshot['latestQuote'].get('bp', 0)
                ask = snapshot['latestQuote'].get('ap', 0)
                if bid and ask:
                    current_price = (bid + ask) / 2

            if current_price <= 0:
                return None

            # Check for ORB breakout
            signal, orb, reason = self.check_orb_breakout(symbol, current_price)

            if signal == 'NONE' or orb is None:
                return None

            # Create signal with ORB-specific risk levels
            risk_settings = self.get_risk_settings(symbol)

            if signal == 'BUY':
                stop_price = orb.stop_long
                target_price = orb.target_long
            else:
                stop_price = orb.stop_short
                target_price = orb.target_short

            # Calculate risk/reward
            if signal == 'BUY':
                risk = current_price - stop_price
                reward = target_price - current_price
            else:
                risk = stop_price - current_price
                reward = current_price - target_price

            rr_ratio = reward / risk if risk > 0 else 0

            # Only take ORB trades with good R:R
            if rr_ratio < 2.0:
                print(f"[ORB] {symbol}: Weak R:R ratio ({rr_ratio:.1f}:1), skipping")
                return None

            print(f"[ORB] {symbol}: {signal} signal! {reason}")
            print(f"[ORB] Entry: ${current_price:.2f}, Stop: ${stop_price:.2f}, Target: ${target_price:.2f}")
            print(f"[ORB] R:R = {rr_ratio:.1f}:1")

            return {
                'symbol': symbol,
                'action': signal,
                'price': current_price,
                'confidence': 0.85,  # ORB signals get high base confidence
                'model': 'ORB_Strategy',
                'stop_loss_price': stop_price,
                'take_profit_price': target_price,
                'stop_loss_pct': risk / current_price,
                'take_profit_pct': reward / current_price,
                'orb_levels': orb,
                'reason': reason
            }

        except Exception as e:
            print(f"[ORB] Error generating signal for {symbol}: {e}")
            return None

    def passes_all_filters(self, symbol: str, action: str, price: float,
                           volume: int = None, vwap: float = None) -> tuple:
        """
        Run all research-backed filters. Returns (passes, reasons).
        """
        reasons = []

        # 1. Daily limits (trade count, loss limit)
        can_trade, reason = self.check_daily_limits()
        if not can_trade:
            return False, [reason]
        reasons.append(f"Daily limits: {reason}")

        # 2. Time of day
        good_time, reason = self.is_good_trading_time()
        if not good_time:
            return False, [reason]
        reasons.append(f"Time: {reason}")

        # 3. Position limit
        if not self.can_open_position():
            return False, ["Max positions reached"]
        reasons.append("Position limit: OK")

        # 4. Volume filter
        vol_ok, reason = self.check_volume_filter(symbol, volume)
        if not vol_ok:
            return False, [reason]
        reasons.append(f"Volume: {reason}")

        # 5. VWAP alignment
        vwap_ok, reason = self.check_vwap_alignment(symbol, action, price, vwap)
        if not vwap_ok:
            return False, [reason]
        reasons.append(f"VWAP: {reason}")

        # 6. VIX BLOCK filter - Prevent trades during high volatility
        if self.config.get('use_vix_block', True):
            vix_ok, reason = self.check_vix_filter()
            if not vix_ok:
                return False, [reason]
            reasons.append(f"VIX: {reason}")

        # 7. TREND CONFIRMATION filter - Prevent coin-flip trades
        if self.config.get('use_trend_confirmation', True):
            trend_ok, reason = self.check_trend_confirmation(symbol, action)
            if not trend_ok:
                return False, [reason]
            reasons.append(f"Trend: {reason}")

        return True, reasons

    def check_vix_filter(self) -> tuple:
        """
        Check if VIX is below blocking threshold.
        Based on pattern learning: volatility_spike is #1 failure type.

        Returns (ok, reason) tuple.
        """
        try:
            # Update VIX if stale (older than 5 minutes)
            now = datetime.now()
            if self.vix_last_update is None or (now - self.vix_last_update).total_seconds() > 300:
                self._update_vix()

            vix_block = self.config.get('vix_block_threshold', 28.0)
            vix_warn = self.config.get('vix_warning_threshold', 25.0)

            if self.current_vix >= vix_block:
                return False, f"VIX {self.current_vix:.1f} >= {vix_block} (too volatile, blocking trades)"

            if self.current_vix >= vix_warn:
                return True, f"VIX {self.current_vix:.1f} elevated (caution)"

            return True, f"VIX {self.current_vix:.1f} OK"

        except Exception as e:
            print(f"[AutoTrader] VIX check error: {e}")
            return True, "VIX check failed, allowing trade"

    def _update_vix(self):
        """Fetch current VIX level."""
        try:
            import yfinance as yf
            vix = yf.Ticker("^VIX")
            hist = vix.history(period="1d")
            if len(hist) > 0:
                self.current_vix = float(hist['Close'].iloc[-1])
                self.vix_last_update = datetime.now()
                print(f"[AutoTrader] VIX updated: {self.current_vix:.1f}")
        except Exception as e:
            print(f"[AutoTrader] VIX update error: {e}")

    def check_trend_confirmation(self, symbol: str, action: str) -> tuple:
        """
        Check if there's sufficient trend confirmation for the trade direction.
        Based on pattern learning: wrong_direction failures all had neutral indicators.

        Requires at least 1 of:
        - RSI > 60 (for buys) or RSI < 40 (for sells)
        - MACD above signal line (for buys) or below (for sells)
        - Price above SMA20 (for buys) or below (for sells)

        Returns (ok, reason) tuple.
        """
        try:
            import yfinance as yf
            import pandas as pd

            # Fetch recent data
            ticker = yf.Ticker(symbol)
            hist = ticker.history(period="2mo", interval="1d")

            if len(hist) < 26:
                return True, "Insufficient data for trend check"

            close = hist['Close']
            current_price = close.iloc[-1]

            # Calculate RSI
            delta = close.diff()
            gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
            loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
            rs = gain / loss
            rsi = (100 - (100 / (1 + rs))).iloc[-1]

            # Calculate MACD
            ema12 = close.ewm(span=12, adjust=False).mean()
            ema26 = close.ewm(span=26, adjust=False).mean()
            macd = (ema12 - ema26).iloc[-1]
            signal = (ema12 - ema26).ewm(span=9, adjust=False).mean().iloc[-1]

            # Calculate SMA20
            sma20 = close.rolling(20).mean().iloc[-1]

            # Check confirmations
            rsi_threshold = self.config.get('trend_rsi_threshold', 10)
            min_signals = self.config.get('trend_min_signals', 1)
            confirmations = []

            if action == 'BUY':
                if rsi > (50 + rsi_threshold):
                    confirmations.append(f"RSI {rsi:.1f}>60")
                if macd > signal:
                    confirmations.append("MACD bullish")
                if current_price > sma20:
                    confirmations.append("Price>SMA20")
            else:  # SELL
                if rsi < (50 - rsi_threshold):
                    confirmations.append(f"RSI {rsi:.1f}<40")
                if macd < signal:
                    confirmations.append("MACD bearish")
                if current_price < sma20:
                    confirmations.append("Price<SMA20")

            if len(confirmations) >= min_signals:
                return True, f"{len(confirmations)} confirms: {', '.join(confirmations)}"
            else:
                return False, f"No trend confirmation for {action} (RSI={rsi:.1f}, need {min_signals}+ signals)"

        except Exception as e:
            print(f"[AutoTrader] Trend confirmation error for {symbol}: {e}")
            return True, "Trend check failed, allowing trade"

    def has_position(self, symbol: str) -> bool:
        """Check if we have a position in a symbol (checks actual broker)."""
        # Check actual broker positions
        if self.broker:
            try:
                actual_positions = {p.symbol for p in self.broker.get_positions()}
                if symbol in actual_positions:
                    return True
            except Exception:
                pass
        return symbol in self.active_positions

    def get_current_brain(self):
        """Get the appropriate specialized brain for current trading mode."""
        if not self.use_specialized_models:
            return None

        if self.config['mode'] == 'swing':
            return self.swing_trading_brain
        else:  # day trading is default
            return self.day_trading_brain

    def get_specialized_prediction(self, symbol: str, df=None) -> Optional[Dict]:
        """
        Get prediction from the specialized trading brain (Day or Swing).

        Returns:
            Dict with action, confidence, stop_loss, take_profit, model, reasoning
        """
        brain = self.get_current_brain()
        if not brain:
            return None

        try:
            # Fetch data if not provided
            if df is None:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                if self.config['mode'] == 'day':
                    # Intraday data for day trading
                    df = ticker.history(period="5d", interval="5m")
                else:
                    # Daily data for swing trading
                    df = ticker.history(period="3mo", interval="1d")

                if len(df) < 20:
                    return None
                df.columns = [c.lower() for c in df.columns]

            # Get analysis from specialized brain
            result = brain.analyze(df, symbol)

            # Convert to standard signal format
            if result:
                return {
                    'symbol': symbol,
                    'action': result.get('action', 'HOLD'),
                    'confidence': result.get('confidence', 0),
                    'model': result.get('model', f'{self.config["mode"]}_brain'),
                    'stop_loss_pct': result.get('stop_loss_pct', self.get_risk_settings(symbol)['stop_loss_pct']),
                    'take_profit_pct': result.get('take_profit_pct', self.get_risk_settings(symbol)['take_profit_pct']),
                    'reasoning': result.get('reasoning', ''),
                    'signals': result.get('signals', {}),
                    'session': result.get('session', 'unknown'),
                    'phase': result.get('phase', 'unknown'),
                }
            return None

        except Exception as e:
            print(f"[AutoTrader] Specialized brain error for {symbol}: {e}")
            return None

    def get_all_specialized_predictions(self, symbol: str) -> List[Dict]:
        """
        Get predictions from all individual models in the specialized brain.

        Returns a list of predictions from each model, sorted by confidence.
        """
        brain = self.get_current_brain()
        if not brain:
            return []

        predictions = []
        try:
            # Fetch data once for all models
            import yfinance as yf
            ticker = yf.Ticker(symbol)
            if self.config['mode'] == 'day':
                df = ticker.history(period="5d", interval="5m")
            else:
                df = ticker.history(period="3mo", interval="1d")

            if len(df) < 20:
                return []
            df.columns = [c.lower() for c in df.columns]

            # Get all signals from the brain (it returns individual model signals)
            result = brain.analyze(df, symbol)
            if not result:
                return []

            # Extract individual model signals if available
            signals = result.get('signals', {})
            risk = self.get_risk_settings(symbol)

            for model_name, signal_data in signals.items():
                if isinstance(signal_data, dict):
                    action = signal_data.get('action', 'HOLD')
                    confidence = signal_data.get('confidence', 0)

                    if action in ('BUY', 'SELL') and confidence > 0:
                        predictions.append({
                            'symbol': symbol,
                            'action': action,
                            'confidence': confidence,
                            'model': model_name,
                            'stop_loss_pct': signal_data.get('stop_loss_pct', risk['stop_loss_pct']),
                            'take_profit_pct': signal_data.get('take_profit_pct', risk['take_profit_pct']),
                            'reasoning': signal_data.get('reasoning', ''),
                        })

            # If no individual signals, use the combined result
            if not predictions and result.get('action') in ('BUY', 'SELL'):
                predictions.append({
                    'symbol': symbol,
                    'action': result.get('action', 'HOLD'),
                    'confidence': result.get('confidence', 0),
                    'model': result.get('model', f'{self.config["mode"]}_brain'),
                    'stop_loss_pct': result.get('stop_loss_pct', risk['stop_loss_pct']),
                    'take_profit_pct': result.get('take_profit_pct', risk['take_profit_pct']),
                    'reasoning': result.get('reasoning', ''),
                })

            # Sort by confidence (highest first)
            predictions.sort(key=lambda x: x.get('confidence', 0), reverse=True)

        except Exception as e:
            print(f"[AutoTrader] Error getting all specialized predictions for {symbol}: {e}")

        return predictions

    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """
        Scan a symbol for AI signals.

        NEW: Uses specialized trading brains (Day Trading or Swing Trading)
        based on the current mode. Falls back to legacy ai_worker if specialized
        models are not available.
        """
        # TRY SPECIALIZED BRAIN FIRST (Day Trading or Swing Trading models)
        if self.use_specialized_models and self.get_current_brain():
            try:
                prediction = self.get_specialized_prediction(symbol)
                if prediction:
                    action = prediction.get('action', 'NONE')
                    confidence = prediction.get('confidence', 0)
                    model = prediction.get('model', 'Unknown')
                    session = prediction.get('session', '')
                    phase = prediction.get('phase', '')

                    if action in ('BUY', 'SELL'):
                        mode_label = "DAY" if self.config['mode'] == 'day' else "SWING"
                        context = f"[{session}]" if session else f"[{phase}]" if phase else ""
                        print(f"[AutoTrader] {symbol}: {action} from {mode_label} brain {context} ({confidence*100:.1f}%)")
                        print(f"[AutoTrader]   Model: {model}")
                        if prediction.get('reasoning'):
                            print(f"[AutoTrader]   Reason: {prediction['reasoning'][:100]}")

                    self.session_stats['signals'] += 1
                    return prediction
            except Exception as e:
                print(f"[AutoTrader] {symbol}: Specialized brain error - {e}")
                # Fall through to legacy ai_worker

        # FALLBACK: Legacy AI worker (original 30-model system)
        if not self.ai_worker or not self.ai_worker.ai_brain:
            print(f"[AutoTrader] {symbol}: No AI worker or brain available")
            return None

        try:
            prediction = self.ai_worker.get_prediction_for_symbol(symbol)
            if prediction:
                action = prediction.get('action', 'NONE')
                confidence = prediction.get('confidence', 0)
                model = prediction.get('model', 'Unknown')
                if action in ('BUY', 'SELL'):
                    print(f"[AutoTrader] {symbol}: {action} signal from {model} ({confidence*100:.1f}%)")
                self.session_stats['signals'] += 1
                return prediction
            else:
                # Debug: why no prediction?
                pass  # Too verbose to log every None
        except Exception as e:
            self.error.emit(f"Scan error for {symbol}: {e}")
            print(f"[AutoTrader] {symbol}: Scan error - {e}")

        return None

    def execute_trade(self, signal: Dict) -> bool:
        """Execute a trade based on AI signal (BUY or SELL). Returns True if executed."""
        symbol = signal.get('symbol')
        action = signal.get('action')
        confidence = signal.get('confidence', 0)
        model = signal.get('model', 'Unknown')

        # Accept both BUY and SELL signals
        if action not in ('BUY', 'SELL'):
            print(f"[AutoTrader] {symbol}: Rejected - invalid action '{action}'")
            return False

        # Check if shorting is allowed
        if action == 'SELL' and not self.config.get('allow_shorting', True):
            print(f"[AutoTrader] {symbol}: Rejected - shorting disabled")
            return False

        # =================================================================
        # RESEARCH-BACKED FILTERS
        # Check daily limits, time of day, volume, VWAP alignment
        # =================================================================
        try:
            # Get current price for VWAP check
            current_price = signal.get('price', 0)
            if current_price <= 0 and self.broker:
                try:
                    quote = self.broker.get_latest_quote(symbol)
                    current_price = float(quote.ask_price) if action == 'BUY' else float(quote.bid_price)
                except:
                    pass

            # Run all filters
            passes, reasons = self.passes_all_filters(
                symbol=symbol,
                action=action,
                price=current_price,
                volume=signal.get('volume'),
                vwap=signal.get('vwap')
            )

            if not passes:
                reason_str = reasons[0] if reasons else "Filter failed"
                print(f"[AutoTrader] {symbol}: Rejected - {reason_str}")
                self.status_update.emit(f"{symbol}: {reason_str}")
                return False
            else:
                print(f"[AutoTrader] {symbol}: Passed filters - {', '.join(reasons)}")

        except Exception as e:
            print(f"[AutoTrader] {symbol}: Filter check error: {e}")
            # Continue with trade if filters fail to run

        if not self.can_open_position():
            print(f"[AutoTrader] {symbol}: Rejected - max positions reached")
            self.status_update.emit(f"Max positions reached, skipping {symbol}")
            return False

        # Check ACTUAL broker positions - don't open opposite position if we already have one
        # Positions should only close via stop loss, take profit, or trailing stop
        if self.broker:
            try:
                existing_positions = {p.symbol: float(p.qty) for p in self.broker.get_positions()}
                if symbol in existing_positions:
                    existing_qty = existing_positions[symbol]
                    # If we have a LONG and get SELL signal, skip (let stops handle exit)
                    # If we have a SHORT and get BUY signal, skip (let stops handle exit)
                    if (existing_qty > 0 and action == 'SELL') or (existing_qty < 0 and action == 'BUY'):
                        print(f"[AutoTrader] Skipping {action} {symbol} - holding position for stop/profit exit")
                        return False
                    # If same direction, skip (already have position)
                    if (existing_qty > 0 and action == 'BUY') or (existing_qty < 0 and action == 'SELL'):
                        self.status_update.emit(f"Already have position in {symbol}")
                        return False
            except Exception as e:
                print(f"[AutoTrader] Error checking positions: {e}")

        if self.has_position(symbol):
            print(f"[AutoTrader] {symbol}: Rejected - already have position (internal check)")
            self.status_update.emit(f"Already have position in {symbol}")
            return False

        if confidence < self.config['min_confidence']:
            print(f"[AutoTrader] {symbol}: Rejected - confidence {confidence*100:.1f}% < {self.config['min_confidence']*100:.1f}%")
            self.status_update.emit(f"{symbol} confidence too low: {confidence*100:.1f}%")
            return False

        # Store unified brain result for later use
        brain_result = None
        trade_df = None  # Store for trade learner

        # Trade Learner: Check if current setup matches known losing patterns
        if self.trade_learner:
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                trade_df = ticker.history(period="3mo", interval="1d")
                if len(trade_df) >= 20:
                    trade_df.columns = [c.lower() for c in trade_df.columns]
                    adjustment, warnings = self.trade_learner.check_setup(action.lower(), trade_df)

                    # Log warnings
                    for warning in warnings:
                        print(f"[TradeLearner] {symbol}: {warning}")

                    # Apply confidence adjustment
                    if adjustment != 0:
                        old_conf = confidence
                        confidence = max(0.3, min(0.95, confidence + adjustment))
                        print(f"[TradeLearner] {symbol}: Adjusted confidence {old_conf*100:.1f}% -> {confidence*100:.1f}% ({adjustment*100:+.1f}%)")

                        # If confidence dropped below threshold after adjustment, reject
                        if confidence < self.config['min_confidence']:
                            print(f"[AutoTrader] {symbol}: Rejected - pattern-adjusted confidence too low")
                            self.status_update.emit(f"Known losing pattern for {symbol}")
                            return False
            except Exception as e:
                print(f"[TradeLearner] {symbol}: Error checking setup: {e}")

        # Multi-Model Confirmation Check (Unified Brain)
        if self.unified_brain and self.config.get('use_unified_brain', True):
            try:
                # Get historical data for analysis (reuse if already fetched)
                if trade_df is None:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    trade_df = ticker.history(period="3mo", interval="1d")
                df = trade_df
                if len(df) >= 20:
                    df.columns = [c.lower() for c in df.columns]

                    # Build XGBoost signal for context
                    xgb_signal = {'action': action, 'confidence': confidence, 'model': model}

                    # Get unified brain analysis
                    result = self.unified_brain.analyze(symbol, df, xgb_signal)
                    brain_result = result  # Store for later use
                    self.current_regime = result.regime

                    # Log analysis
                    print(f"[AutoTrader] {symbol}: Unified Brain Analysis:")
                    print(f"  Regime: {result.regime.value}")
                    print(f"  Action: {result.action} (conf: {result.confidence:.2f})")
                    print(f"  Entry Quality: {result.entry_quality}")
                    if result.week_high > 0:
                        print(f"  Week Range: ${result.week_low:.2f} - ${result.week_high:.2f}")
                        print(f"  Suggested Stop: ${result.suggested_stop:.2f}, Target: ${result.suggested_target:.2f}")
                    for reason in result.reasoning[:5]:  # Top 5 reasons
                        print(f"    - {reason}")

                    # Reject poor entry quality trades
                    if result.entry_quality == "poor" and action == "BUY":
                        print(f"[AutoTrader] {symbol}: Rejected - poor entry quality based on historical analysis")
                        self.status_update.emit(f"Poor entry quality for {symbol}")
                        return False

                    # Check if unified brain confirms
                    confirms = result.action == action

                    if confirms:
                        print(f"[AutoTrader] {symbol}: CONFIRMED by {len(result.signals)} analyzers")
                        self.status_update.emit(f"Brain confirms {action} {symbol} ({result.regime.value})")
                        # Boost confidence if multiple models agree
                        confidence = min(confidence * 1.1, 0.95)
                    else:
                        print(f"[AutoTrader] {symbol}: Brain says {result.action}, XGBoost says {action}")

                        # Check regime alignment
                        if self.config.get('respect_regime', True):
                            if action == 'BUY' and result.regime in [MarketRegime.BEAR, MarketRegime.STRONG_BEAR]:
                                print(f"[AutoTrader] {symbol}: Rejected - BUY in {result.regime.value} market")
                                self.status_update.emit(f"Blocked: BUY in {result.regime.value}")
                                return False
                            if action == 'SELL' and result.regime in [MarketRegime.BULL, MarketRegime.STRONG_BULL]:
                                print(f"[AutoTrader] {symbol}: Rejected - SELL in {result.regime.value} market")
                                self.status_update.emit(f"Blocked: SELL in {result.regime.value}")
                                return False

                        # If require_confirmation is True, reject trades that brain doesn't confirm
                        if self.config.get('require_confirmation', False):
                            if result.action != action and result.action != 'HOLD':
                                print(f"[AutoTrader] {symbol}: Rejected - Brain disagrees ({result.action} vs {action})")
                                self.status_update.emit(f"Brain disagrees: {result.action} vs {action}")
                                return False

            except Exception as e:
                print(f"[AutoTrader] {symbol}: Unified Brain error: {e}")
                # Continue without confirmation if it fails

        # Legacy: RL-only confirmation (fallback)
        elif self.rl_analyzer and self.config.get('use_rl_confirmation', True):
            try:
                import yfinance as yf
                ticker = yf.Ticker(symbol)
                df = ticker.history(period="3mo", interval="1d")
                if len(df) >= 20:
                    df.columns = [c.lower() for c in df.columns]
                    rl_confirms, rl_conf = self.rl_analyzer.confirms_trade(df, action)
                    rl_signal = self.rl_analyzer.get_signal(df)

                    if rl_confirms:
                        print(f"[AutoTrader] {symbol}: RL CONFIRMS {action} (conf: {rl_conf:.2f})")
                        self.status_update.emit(f"RL confirms {action} {symbol}")
                    else:
                        rl_action = rl_signal.get('action', 'HOLD')
                        print(f"[AutoTrader] {symbol}: RL says {rl_action}, XGBoost says {action}")

                        if self.config.get('require_confirmation', False):
                            if rl_action != action and rl_action != 'HOLD':
                                print(f"[AutoTrader] {symbol}: Rejected - RL disagrees ({rl_action} vs {action})")
                                self.status_update.emit(f"RL disagrees: {rl_action} vs {action}")
                                return False
            except Exception as e:
                print(f"[AutoTrader] {symbol}: RL check error: {e}")

        if not self.broker:
            print(f"[AutoTrader] {symbol}: Rejected - no broker connected")
            self.error.emit("No broker connected")
            return False

        try:
            # Get current price
            snapshot = self.broker.get_snapshot(symbol)
            if not snapshot:
                print(f"[AutoTrader] {symbol}: Rejected - no snapshot data")
                return False

            # Alpaca snapshot returns nested structure: latestTrade.p or latestQuote for price
            current_price = 0
            if 'latestTrade' in snapshot and snapshot['latestTrade']:
                current_price = snapshot['latestTrade'].get('p', 0)
            elif 'latestQuote' in snapshot and snapshot['latestQuote']:
                # Use midpoint of bid/ask
                bid = snapshot['latestQuote'].get('bp', 0)
                ask = snapshot['latestQuote'].get('ap', 0)
                if bid and ask:
                    current_price = (bid + ask) / 2
            elif 'price' in snapshot:
                # Fallback for other formats
                current_price = snapshot.get('price', 0)

            if current_price <= 0:
                print(f"[AutoTrader] {symbol}: Rejected - invalid price: {current_price} (snapshot: {list(snapshot.keys())})")
                return False

            # Calculate position size (reduced for crypto, scaled by model performance)
            shares = self.calculate_position_size(current_price, symbol, model_name=model)
            if shares <= 0:
                print(f"[AutoTrader] {symbol}: Rejected - calculated 0 shares at ${current_price:.2f}")
                return False

            print(f"[AutoTrader] {symbol}: Calculated {shares} shares at ${current_price:.2f}")

            # Get risk settings (from AI or defaults, crypto gets different settings)
            risk = self.get_risk_settings(symbol)
            stop_loss_pct = signal.get('stop_loss_pct', risk['stop_loss_pct'])
            take_profit_pct = signal.get('take_profit_pct', risk['take_profit_pct'])
            trailing_activation = signal.get('trailing_activation_pct', risk['trailing_activation_pct'])
            trailing_stop = signal.get('trailing_stop_pct', risk['trailing_stop_pct'])

            # Calculate stop/target based on direction
            from data.broker_adapter import OrderSide, OrderType

            if action == 'BUY':
                order_side = OrderSide.BUY
                stop_price = current_price * (1 - stop_loss_pct)
                target_price = current_price * (1 + take_profit_pct)

                # Use historical analysis suggested levels if available and better
                if brain_result and brain_result.suggested_stop > 0:
                    historical_stop = brain_result.suggested_stop
                    historical_target = brain_result.suggested_target
                    # Use historical stop if it's tighter (closer to entry) but still protective
                    if historical_stop > stop_price and historical_stop < current_price:
                        print(f"[AutoTrader] Using historical stop: ${historical_stop:.2f} (was ${stop_price:.2f})")
                        stop_price = historical_stop
                    # Use historical target if it's higher
                    if historical_target > target_price:
                        print(f"[AutoTrader] Using historical target: ${historical_target:.2f} (was ${target_price:.2f})")
                        target_price = historical_target

            else:  # SELL (short)
                order_side = OrderSide.SELL
                stop_price = current_price * (1 + stop_loss_pct)  # Stop above for short
                target_price = current_price * (1 - take_profit_pct)  # Target below for short

            print(f"[AutoTrader] {action} signal for {symbol}: {model} ({confidence*100:.1f}%)")
            print(f"[AutoTrader] Risk params from model: SL={stop_loss_pct*100:.1f}%, TP={take_profit_pct*100:.1f}%")
            if brain_result and brain_result.entry_quality != "unknown":
                print(f"[AutoTrader] Historical: {brain_result.entry_quality} entry, week ${brain_result.week_low:.2f}-${brain_result.week_high:.2f}")

            if self.config['auto_execute']:
                order = self.broker.place_order(
                    symbol=symbol,
                    qty=shares,
                    side=order_side,
                    type=OrderType.MARKET
                )

                if order:
                    # Add to position monitor
                    if self.position_monitor:
                        self.position_monitor.add_position(
                            symbol=symbol,
                            entry_price=current_price,
                            shares=shares,
                            stop_loss=stop_price,
                            take_profit=target_price,
                            trailing_activation_pct=trailing_activation,
                            trailing_stop_pct=trailing_stop,
                            ai_model=model,
                            ai_confidence=confidence,
                            trade_type=self.config['mode']
                        )

                    # Track position
                    self.active_positions[symbol] = {
                        'entry_price': current_price,
                        'shares': shares if action == 'BUY' else -shares,  # Negative for shorts
                        'side': action,
                        'stop_loss': stop_price,
                        'take_profit': target_price,
                        'model': model,
                        'confidence': confidence,
                        'entry_time': datetime.now().isoformat()
                    }

                    self.session_stats['trades'] += 1

                    # Log trade entry to database
                    try:
                        trade_id = self.trade_logger.log_entry(
                            symbol=symbol,
                            side=action.lower(),  # 'buy' or 'sell'
                            price=current_price,
                            shares=shares,
                            ai_model=model,
                            ai_confidence=confidence,
                            stop_loss=stop_price,
                            take_profit=target_price,
                            trade_type=self.config['mode']
                        )
                        self.active_positions[symbol]['trade_id'] = trade_id
                    except Exception as e:
                        print(f"[AutoTrader] Error logging trade: {e}")

                    trade_info = {
                        'symbol': symbol,
                        'action': action,  # BUY or SELL
                        'price': current_price,
                        'shares': shares,
                        'stop_loss': stop_price,
                        'take_profit': target_price,
                        'model': model,
                        'confidence': confidence,
                        'mode': self.config['mode'],
                        'stop_loss_pct': stop_loss_pct,
                        'take_profit_pct': take_profit_pct
                    }

                    self.trade_executed.emit(trade_info)
                    self.status_update.emit(
                        f"{action} {shares} {symbol} @ ${current_price:.2f} "
                        f"(SL: ${stop_price:.2f}, TP: ${target_price:.2f})"
                    )

                    # Record trade for daily tracking (anti-overtrading)
                    self.record_trade(pnl=0)  # P&L will be updated when trade closes
                    print(f"[AutoTrader] Trades today: {self.daily_stats['trades_today']}/{self.config.get('max_trades_per_day', 3)}")

                    return True

        except Exception as e:
            self.error.emit(f"Trade execution error: {e}")

        return False

    def on_trade_closed(self, result: Dict):
        """Handle trade closed from position monitor."""
        symbol = result.get('symbol')
        pnl = result.get('pnl', 0)
        exit_price = result.get('exit_price', 0)

        # Update daily P&L for loss limit tracking
        self.reset_daily_stats()
        self.daily_stats['pnl_today'] += pnl
        print(f"[AutoTrader] Daily P&L: ${self.daily_stats['pnl_today']:+,.2f}")
        exit_reason = result.get('exit_reason', 'unknown')

        # Update stats
        if pnl > 0:
            self.session_stats['wins'] += 1
        else:
            self.session_stats['losses'] += 1
        self.session_stats['pnl'] += pnl

        # Get model and trade_id before removing position
        model = None
        trade_id = None
        if symbol in self.active_positions:
            model = self.active_positions[symbol].get('model')
            trade_id = self.active_positions[symbol].get('trade_id')
            del self.active_positions[symbol]

        # Log trade exit to database
        try:
            if trade_id:
                self.trade_logger.log_exit(trade_id, exit_price, exit_reason)
            else:
                # Fallback: try to find by symbol
                self.trade_logger.log_exit_by_symbol(symbol, exit_price, exit_reason)
        except Exception as e:
            print(f"[AutoTrader] Error logging trade exit: {e}")

        # Record trade outcome for model learning
        if self.model_learner and model:
            try:
                self.model_learner.record_trade_outcome(model, pnl)
                weight = self.model_learner.get_model_weight(model)
                enabled = self.model_learner.is_model_enabled(model)
                status_str = f"weight={weight:.2f}" if enabled else "DISABLED"
                print(f"[ModelLearner] Recorded {model}: PnL=${pnl:+.2f} ({status_str})")
            except Exception as e:
                print(f"[AutoTrader] Error recording model outcome: {e}")

        # Analyze WHY the trade won or lost (Trade Learner)
        if self.trade_learner:
            try:
                # Get position details
                entry_price = result.get('entry_price', 0)
                entry_time = result.get('entry_time')
                exit_time = result.get('exit_time', datetime.now())

                # Calculate duration
                if entry_time and exit_time:
                    if isinstance(entry_time, str):
                        entry_time = datetime.fromisoformat(entry_time)
                    if isinstance(exit_time, str):
                        exit_time = datetime.fromisoformat(exit_time)
                    duration = int((exit_time - entry_time).total_seconds() / 60)
                else:
                    duration = 0

                # Calculate PnL percentage
                pnl_pct = (pnl / (entry_price * abs(result.get('shares', 1)))) if entry_price > 0 else 0

                # Prepare trade dict for analysis
                trade_dict = {
                    'id': trade_id or 0,
                    'symbol': symbol,
                    'side': result.get('side', 'buy'),
                    'entry_price': entry_price,
                    'exit_price': exit_price,
                    'pnl': pnl,
                    'pnl_pct': pnl_pct,
                    'duration_minutes': duration,
                    'exit_reason': exit_reason
                }

                # Get historical data for market conditions analysis
                try:
                    import yfinance as yf
                    ticker = yf.Ticker(symbol)
                    df = ticker.history(period="1mo", interval="1d")
                    if len(df) >= 20:
                        df.columns = [c.lower() for c in df.columns]
                    else:
                        df = None
                except:
                    df = None

                # Analyze the trade
                analysis = self.trade_learner.analyze_trade(trade_dict, df)
                self.trade_learner.store_analysis(analysis)

                if analysis.failure_type:
                    print(f"[TradeLearner] {symbol}: {analysis.failure_type} - {analysis.failure_details}")
                if analysis.should_have_avoided:
                    print(f"[TradeLearner] {symbol}: Marked as avoidable loss - pattern stored for future reference")

            except Exception as e:
                print(f"[TradeLearner] Error analyzing trade: {e}")

        # Emit updated stats
        self.stats_updated.emit(self.get_stats())

        status = "WIN" if pnl > 0 else "LOSS"
        model_info = f" [{model}]" if model else ""
        self.status_update.emit(
            f"CLOSED {symbol}{model_info}: {status} ${pnl:+.2f} ({result.get('exit_reason', 'unknown')})"
        )

    def get_stats(self) -> Dict:
        """Get current session statistics."""
        total = self.session_stats['wins'] + self.session_stats['losses']
        win_rate = (self.session_stats['wins'] / total * 100) if total > 0 else 0

        return {
            'trades': self.session_stats['trades'],
            'wins': self.session_stats['wins'],
            'losses': self.session_stats['losses'],
            'win_rate': win_rate,
            'pnl': self.session_stats['pnl'],
            'signals': self.session_stats['signals'],
            'active_positions': len(self.active_positions)
        }

    def get_model_performance_status(self) -> Dict:
        """
        Get live performance status for all models.

        Returns dict with model performance and position multipliers.
        """
        if not self.performance_tracker:
            return {'error': 'Performance tracker not available'}

        return self.performance_tracker.get_model_performance()

    def print_performance_status(self):
        """Print current model performance status for debugging."""
        if self.performance_tracker:
            self.performance_tracker.print_status()
        else:
            print("[AutoTrader] Performance tracker not available")

    def run(self):
        """Main trading loop."""
        self.running = True
        brain_type = "Specialized" if self.use_specialized_models else "Legacy"
        model_count = "10" if self.use_specialized_models else "30"
        print(f"[AutoTrader] Started in {self.config['mode'].upper()} mode ({brain_type} brain, {model_count} models)")
        self.status_update.emit(f"Auto trader started ({self.config['mode'].upper()} mode - {brain_type})")

        scan_count = 0
        while self.running:
            if self.paused:
                self.msleep(1000)
                continue

            if not self.is_market_open():
                self.status_update.emit("Market closed - waiting...")
                print("[AutoTrader] Market closed - waiting...")
                self.msleep(60000)  # Check every minute when market closed
                continue

            # Scan watchlist
            self.mutex.lock()
            symbols = list(self.watchlist)
            self.mutex.unlock()

            scan_count += 1
            brain_info = f"{self.config['mode'].upper()} brain" if self.use_specialized_models else "30 models"
            print(f"[AutoTrader] Scan #{scan_count} - Scanning {len(symbols)} symbols ({brain_info})...")

            signals_found = 0
            trades_executed = 0

            for symbol in symbols:
                if not self.running or self.paused:
                    break

                # Skip if we already have a position in this symbol
                if self.has_position(symbol):
                    continue

                # Check if we can open more positions
                if not self.can_open_position():
                    print(f"[AutoTrader] Max positions reached ({self.config['max_positions']})")
                    break

                # =================================================================
                # ORB STRATEGY CHECK (Research: 1,600% returns 2016-2023)
                # Check for Opening Range Breakout signals FIRST
                # =================================================================
                if self.config.get('use_orb_strategy', True):
                    orb_signal = self.generate_orb_signal(symbol)
                    if orb_signal:
                        signals_found += 1
                        self.signal_generated.emit(orb_signal)

                        if self.config['auto_execute']:
                            print(f"[AutoTrader] Trying ORB {orb_signal['action']} for {symbol}...")
                            success = self.execute_trade(orb_signal)
                            if success:
                                trades_executed += 1
                                continue  # Move to next symbol after ORB trade

                # Get predictions from specialized brain OR legacy models
                all_predictions = []

                # TRY SPECIALIZED BRAIN FIRST (Day Trading or Swing Trading models)
                if self.use_specialized_models and self.get_current_brain():
                    all_predictions = self.get_all_specialized_predictions(symbol)
                    if all_predictions:
                        mode_label = "DAY" if self.config['mode'] == 'day' else "SWING"
                        print(f"[AutoTrader] {symbol}: Got {len(all_predictions)} signals from {mode_label} brain")

                # FALLBACK to legacy 30-model system
                if not all_predictions and self.ai_worker:
                    all_predictions = self.ai_worker.get_all_model_predictions(symbol)

                    # Filter out disabled models and apply adaptive weights
                    if self.model_learner:
                        filtered_predictions = []
                        for pred in all_predictions:
                            model_name = pred.get('model', 'Unknown')
                            if self.model_learner.is_model_enabled(model_name):
                                # Apply model weight to confidence for sorting
                                weight = self.model_learner.get_model_weight(model_name)
                                pred['_weighted_confidence'] = pred.get('confidence', 0) * weight
                                filtered_predictions.append(pred)
                            else:
                                print(f"[AutoTrader] Skipping disabled model: {model_name}")
                        all_predictions = filtered_predictions
                        # Sort by weighted confidence (models with better track records rank higher)
                        all_predictions.sort(key=lambda x: x.get('_weighted_confidence', 0), reverse=True)
                    else:
                        # Fallback: Sort by raw confidence
                        all_predictions.sort(key=lambda x: x.get('confidence', 0), reverse=True)

                # Go through each model's prediction
                for prediction in all_predictions:
                    if not self.running or self.paused:
                        break

                    action = prediction.get('action', 'HOLD')
                    confidence = prediction.get('confidence', 0)
                    model = prediction.get('model', 'Unknown')

                    # Skip HOLD signals or low confidence
                    if action == 'HOLD':
                        continue

                    if confidence < self.config['min_confidence']:
                        continue

                    signals_found += 1
                    prediction['symbol'] = symbol
                    print(f"[AutoTrader] SIGNAL: {action} {symbol} ({confidence*100:.1f}% - {model})")
                    self.signal_generated.emit(prediction)

                    if self.config['auto_execute']:
                        print(f"[AutoTrader] Trying {action} trade for {symbol} via {model}...")
                        success = self.execute_trade(prediction)
                        if success:
                            trades_executed += 1
                            break  # Trade executed, move to next symbol
                        else:
                            print(f"[AutoTrader] {model} signal didn't execute, trying next model...")
                            continue  # Try next model in the list

                # Small delay between symbols
                self.msleep(500)

            # Summary after each scan
            print(f"[AutoTrader] Scan complete: {signals_found} signals, {trades_executed} trades")

            # Wait before next scan cycle
            print(f"[AutoTrader] Next scan in {self.config['scan_interval']//1000}s...")
            self.msleep(self.config['scan_interval'])

        self.status_update.emit("Auto trader stopped")

    def stop(self):
        """Stop the auto trader."""
        self.running = False

    def pause(self):
        """Pause trading."""
        self.paused = True
        self.status_update.emit("Trading PAUSED")

    def resume(self):
        """Resume trading."""
        self.paused = False
        self.status_update.emit("Trading RESUMED")

    def close_all_positions(self):
        """Close all active positions."""
        if not self.broker:
            return

        from data.broker_adapter import OrderSide, OrderType

        for symbol, pos in list(self.active_positions.items()):
            try:
                self.broker.place_order(
                    symbol=symbol,
                    qty=pos['shares'],
                    side=OrderSide.SELL,
                    type=OrderType.MARKET
                )
                self.status_update.emit(f"Closed position: {symbol}")
            except Exception as e:
                self.error.emit(f"Failed to close {symbol}: {e}")


# Default watchlist for day trading
DEFAULT_DAY_WATCHLIST = [
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'TSLA',
    'GOOGL', 'AMZN', 'META', 'NFLX', 'SPY',
    'QQQ', 'COIN', 'PLTR', 'SOFI', 'RIVN'
]

# Default watchlist for swing trading
DEFAULT_SWING_WATCHLIST = [
    'AAPL', 'MSFT', 'NVDA', 'AMD', 'GOOGL',
    'AMZN', 'META', 'JPM', 'V', 'MA',
    'UNH', 'LLY', 'AVGO', 'COST', 'HD'
]
