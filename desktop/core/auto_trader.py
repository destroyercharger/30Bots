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

        # Trading configuration
        self.config = {
            'mode': 'day',  # 'day' or 'swing'
            'max_positions': 4,
            'position_size_pct': 0.10,  # 10% of portfolio per position
            'min_confidence': 0.65,  # Minimum AI confidence (lowered from 0.70)
            'scan_interval': 30000,  # 30 seconds
            'market_hours_only': True,
            'auto_execute': True,  # Auto-execute trades or just signal
        }

        # Day trading settings
        self.day_settings = {
            'stop_loss_pct': 0.02,
            'take_profit_pct': 0.04,
            'trailing_activation_pct': 0.015,
            'trailing_stop_pct': 0.008,
        }

        # Swing trading settings
        self.swing_settings = {
            'stop_loss_pct': 0.05,
            'take_profit_pct': 0.12,
            'trailing_activation_pct': 0.04,
            'trailing_stop_pct': 0.025,
        }

        # Watchlist
        self.watchlist: List[str] = []

        # Current positions (symbol -> position data)
        self.active_positions: Dict[str, Dict] = {}

        # Trade logger
        self.trade_logger = get_trade_logger()

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

    def get_risk_settings(self) -> Dict:
        """Get current risk settings based on mode."""
        if self.config['mode'] == 'swing':
            return self.swing_settings
        return self.day_settings

    def calculate_position_size(self, price: float) -> int:
        """Calculate position size based on portfolio."""
        if not self.broker:
            return 0

        try:
            account = self.broker.get_account()
            portfolio_value = float(account.equity)
            position_value = portfolio_value * self.config['position_size_pct']
            shares = int(position_value / price)
            return max(1, shares)
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

    def scan_symbol(self, symbol: str) -> Optional[Dict]:
        """Scan a symbol for AI signals."""
        if not self.ai_worker or not self.ai_worker.ai_brain:
            print(f"[AutoTrader] {symbol}: No AI worker or brain")
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

    def execute_trade(self, signal: Dict):
        """Execute a trade based on AI signal (BUY or SELL)."""
        symbol = signal.get('symbol')
        action = signal.get('action')
        confidence = signal.get('confidence', 0)
        model = signal.get('model', 'Unknown')

        # Accept both BUY and SELL signals
        if action not in ('BUY', 'SELL'):
            return

        if not self.can_open_position():
            self.status_update.emit(f"Max positions reached, skipping {symbol}")
            return

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
                        return
                    # If same direction, skip (already have position)
                    if (existing_qty > 0 and action == 'BUY') or (existing_qty < 0 and action == 'SELL'):
                        self.status_update.emit(f"Already have position in {symbol}")
                        return
            except Exception as e:
                print(f"[AutoTrader] Error checking positions: {e}")

        if self.has_position(symbol):
            self.status_update.emit(f"Already have position in {symbol}")
            return

        if confidence < self.config['min_confidence']:
            self.status_update.emit(f"{symbol} confidence too low: {confidence*100:.1f}%")
            return

        if not self.broker:
            self.error.emit("No broker connected")
            return

        try:
            # Get current price
            snapshot = self.broker.get_snapshot(symbol)
            if not snapshot:
                return

            current_price = snapshot.get('price', 0)
            if current_price <= 0:
                return

            # Calculate position size
            shares = self.calculate_position_size(current_price)
            if shares <= 0:
                return

            # Get risk settings (from AI or defaults)
            risk = self.get_risk_settings()
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
            else:  # SELL (short)
                order_side = OrderSide.SELL
                stop_price = current_price * (1 + stop_loss_pct)  # Stop above for short
                target_price = current_price * (1 - take_profit_pct)  # Target below for short

            print(f"[AutoTrader] {action} signal for {symbol}: {model} ({confidence*100:.1f}%)")
            print(f"[AutoTrader] Risk params from model: SL={stop_loss_pct*100:.1f}%, TP={take_profit_pct*100:.1f}%")

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

        except Exception as e:
            self.error.emit(f"Trade execution error: {e}")

    def on_trade_closed(self, result: Dict):
        """Handle trade closed from position monitor."""
        symbol = result.get('symbol')
        pnl = result.get('pnl', 0)

        # Update stats
        if pnl > 0:
            self.session_stats['wins'] += 1
        else:
            self.session_stats['losses'] += 1
        self.session_stats['pnl'] += pnl

        # Remove from active positions
        if symbol in self.active_positions:
            del self.active_positions[symbol]

        # Emit updated stats
        self.stats_updated.emit(self.get_stats())

        status = "WIN" if pnl > 0 else "LOSS"
        self.status_update.emit(
            f"CLOSED {symbol}: {status} ${pnl:+.2f} ({result.get('exit_reason', 'unknown')})"
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

    def run(self):
        """Main trading loop."""
        self.running = True
        print(f"[AutoTrader] Started in {self.config['mode'].upper()} mode")
        self.status_update.emit(f"Auto trader started ({self.config['mode'].upper()} mode)")

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
            print(f"[AutoTrader] Scan #{scan_count} - Scanning {len(symbols)} symbols...")

            signals_found = 0
            for symbol in symbols:
                if not self.running or self.paused:
                    break

                # Skip if we already have a position
                if self.has_position(symbol):
                    continue

                # Scan for signals
                signal = self.scan_symbol(symbol)

                if signal and signal.get('action') in ('BUY', 'SELL'):
                    signals_found += 1
                    action = signal.get('action')
                    confidence = signal.get('confidence', 0)
                    model = signal.get('model', 'Unknown')
                    print(f"[AutoTrader] SIGNAL: {action} {symbol} ({confidence*100:.1f}% - {model})")
                    self.signal_generated.emit(signal)

                    if self.config['auto_execute'] and confidence >= self.config['min_confidence']:
                        print(f"[AutoTrader] Executing {action} trade for {symbol}...")
                        self.execute_trade(signal)

                # Small delay between scans
                self.msleep(500)

            # Summary after each scan
            if signals_found > 0:
                print(f"[AutoTrader] Scan complete: {signals_found} BUY signals found")
            else:
                print(f"[AutoTrader] Scan complete: No actionable signals")

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
