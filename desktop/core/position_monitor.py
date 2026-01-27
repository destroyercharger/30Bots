"""
Position Monitor - Watches positions for stop loss, take profit, and trailing stops
"""

from typing import Dict, List, Optional
from datetime import datetime
from PyQt6.QtCore import QThread, pyqtSignal, QMutex

from .trade_logger import get_trade_logger


class PositionMonitor(QThread):
    """
    Background worker that monitors positions and triggers exits.

    Watches for:
    - Stop loss hits
    - Take profit hits
    - Trailing stop activation and execution
    """

    # Signals
    stop_loss_triggered = pyqtSignal(str, float, str)  # symbol, price, reason
    take_profit_triggered = pyqtSignal(str, float, str)  # symbol, price, reason
    trailing_stop_triggered = pyqtSignal(str, float, str)  # symbol, price, reason
    position_updated = pyqtSignal(str, dict)  # symbol, updated position data
    trade_closed = pyqtSignal(dict)  # trade result from logger

    def __init__(self, broker=None, parent=None):
        super().__init__(parent)
        self.broker = broker
        self.running = False
        self.check_interval = 2000  # 2 seconds
        self.mutex = QMutex()

        # Tracked positions with their risk params
        # {symbol: {entry_price, shares, stop_loss, take_profit, trailing_activation,
        #           trailing_stop_pct, high_water_mark, trailing_active, trade_id, ai_model}}
        self.positions: Dict[str, Dict] = {}

        self.trade_logger = get_trade_logger()

    def set_broker(self, broker):
        """Set the broker for executing stops."""
        self.broker = broker

    def add_position(self, symbol: str, entry_price: float, shares: int,
                     stop_loss: float, take_profit: float,
                     trailing_activation_pct: float = 0.015,
                     trailing_stop_pct: float = 0.01,
                     ai_model: str = None, ai_confidence: float = None,
                     trade_type: str = 'day'):
        """Add a position to monitor."""
        self.mutex.lock()

        # Log entry to database
        trade_id = self.trade_logger.log_entry(
            symbol=symbol,
            side='buy',
            price=entry_price,
            shares=shares,
            ai_model=ai_model,
            ai_confidence=ai_confidence,
            stop_loss=stop_loss,
            take_profit=take_profit,
            trade_type=trade_type
        )

        # Determine if this is a long or short position
        # For shorts, stop_loss > entry_price; for longs, stop_loss < entry_price
        is_short = stop_loss > entry_price

        self.positions[symbol] = {
            'entry_price': entry_price,
            'shares': shares,
            'stop_loss': stop_loss,
            'take_profit': take_profit,
            'trailing_activation_pct': trailing_activation_pct,
            'trailing_stop_pct': trailing_stop_pct,
            'high_water_mark': entry_price,
            'trailing_active': False,
            'trailing_stop_price': None,
            'trade_id': trade_id,
            'ai_model': ai_model,
            'current_price': entry_price,
            'is_short': is_short
        }

        self.mutex.unlock()
        side_str = "SHORT" if is_short else "LONG"
        print(f"[Monitor] Added {symbol} ({side_str}): entry=${entry_price:.2f}, SL=${stop_loss:.2f}, TP=${take_profit:.2f}")

    def remove_position(self, symbol: str):
        """Remove a position from monitoring."""
        self.mutex.lock()
        if symbol in self.positions:
            del self.positions[symbol]
            print(f"[Monitor] Removed {symbol}")
        self.mutex.unlock()

    def update_price(self, symbol: str, price: float):
        """Update current price for a position."""
        self.mutex.lock()
        if symbol in self.positions:
            self.positions[symbol]['current_price'] = price
        self.mutex.unlock()

    def check_position(self, symbol: str, current_price: float) -> Optional[str]:
        """
        Check a position against its stop/target levels.
        Returns exit reason if triggered, None otherwise.
        Handles both LONG and SHORT positions correctly.
        """
        if symbol not in self.positions:
            return None

        pos = self.positions[symbol]
        entry_price = pos['entry_price']
        stop_loss = pos['stop_loss']
        take_profit = pos['take_profit']
        is_short = pos.get('is_short', False)

        if is_short:
            # SHORT position: lose when price goes UP, win when price goes DOWN
            # Stop loss is ABOVE entry, take profit is BELOW entry
            if current_price >= stop_loss:
                return 'stop_loss'
            if current_price <= take_profit:
                return 'take_profit'
        else:
            # LONG position: lose when price goes DOWN, win when price goes UP
            # Stop loss is BELOW entry, take profit is ABOVE entry
            if current_price <= stop_loss:
                return 'stop_loss'
            if current_price >= take_profit:
                return 'take_profit'

        # Check trailing stop (works same for both - trailing follows profitable direction)
        if pos['trailing_active']:
            if is_short:
                # For shorts, trailing stop goes DOWN with price
                if current_price >= pos['trailing_stop_price']:
                    return 'trailing_stop'
                # Update low water mark and trailing stop (follows price down)
                if current_price < pos['high_water_mark']:
                    pos['high_water_mark'] = current_price
                    pos['trailing_stop_price'] = current_price * (1 + pos['trailing_stop_pct'])
            else:
                # For longs, trailing stop goes UP with price
                if current_price <= pos['trailing_stop_price']:
                    return 'trailing_stop'
                # Update high water mark and trailing stop (follows price up)
                if current_price > pos['high_water_mark']:
                    pos['high_water_mark'] = current_price
                    pos['trailing_stop_price'] = current_price * (1 - pos['trailing_stop_pct'])
        else:
            # Check if trailing should activate
            if is_short:
                # For shorts, gain is when price drops
                gain_pct = (entry_price - current_price) / entry_price
            else:
                # For longs, gain is when price rises
                gain_pct = (current_price - entry_price) / entry_price

            if gain_pct >= pos['trailing_activation_pct']:
                pos['trailing_active'] = True
                pos['high_water_mark'] = current_price
                if is_short:
                    pos['trailing_stop_price'] = current_price * (1 + pos['trailing_stop_pct'])
                else:
                    pos['trailing_stop_price'] = current_price * (1 - pos['trailing_stop_pct'])
                print(f"[Monitor] {symbol} trailing stop activated at ${current_price:.2f}")

        return None

    def execute_exit(self, symbol: str, price: float, reason: str):
        """Execute a position exit."""
        if symbol not in self.positions:
            return

        pos = self.positions[symbol]

        # Log exit to database
        result = self.trade_logger.log_exit(pos['trade_id'], price, reason)

        if result:
            self.trade_closed.emit(result)
            pnl = result['pnl']
            pnl_pct = result['pnl_pct']
            status = "WIN" if pnl > 0 else "LOSS"
            print(f"[Monitor] {symbol} {reason}: {status} ${pnl:+.2f} ({pnl_pct:+.1f}%)")

        # Execute exit order through broker
        # For LONG positions: SELL to close
        # For SHORT positions: BUY to close
        if self.broker:
            try:
                from data.broker_adapter import OrderSide, OrderType
                is_short = pos.get('is_short', False)
                exit_side = OrderSide.BUY if is_short else OrderSide.SELL
                self.broker.place_order(
                    symbol=symbol,
                    qty=pos['shares'],
                    side=exit_side,
                    type=OrderType.MARKET
                )
            except Exception as e:
                print(f"[Monitor] Failed to execute exit for {symbol}: {e}")

        # Emit appropriate signal
        if reason == 'stop_loss':
            self.stop_loss_triggered.emit(symbol, price, reason)
        elif reason == 'take_profit':
            self.take_profit_triggered.emit(symbol, price, reason)
        elif reason == 'trailing_stop':
            self.trailing_stop_triggered.emit(symbol, price, reason)

        # Remove from monitoring
        self.remove_position(symbol)

    def run(self):
        """Main monitoring loop."""
        self.running = True
        print("[Monitor] Position monitor started")

        while self.running:
            self.mutex.lock()
            positions_snapshot = dict(self.positions)
            self.mutex.unlock()

            for symbol, pos in positions_snapshot.items():
                # FETCH LIVE PRICE from broker
                current_price = pos.get('current_price', pos['entry_price'])
                if self.broker:
                    try:
                        snapshot = self.broker.get_snapshot(symbol)
                        if snapshot and snapshot.get('price'):
                            current_price = snapshot['price']
                            # Update stored price
                            self.mutex.lock()
                            if symbol in self.positions:
                                self.positions[symbol]['current_price'] = current_price
                            self.mutex.unlock()
                    except Exception as e:
                        pass  # Use last known price

                # Check for exit conditions
                exit_reason = self.check_position(symbol, current_price)

                if exit_reason:
                    print(f"[Monitor] {symbol} EXIT triggered: {exit_reason} at ${current_price:.2f}")
                    print(f"[Monitor]   Entry: ${pos['entry_price']:.2f}, SL: ${pos['stop_loss']:.2f}, TP: ${pos['take_profit']:.2f}")
                    self.execute_exit(symbol, current_price, exit_reason)
                else:
                    # Emit position update
                    pnl = (current_price - pos['entry_price']) * pos['shares']
                    pnl_pct = ((current_price - pos['entry_price']) / pos['entry_price']) * 100
                    self.position_updated.emit(symbol, {
                        'current_price': current_price,
                        'pnl': pnl,
                        'pnl_pct': pnl_pct,
                        'trailing_active': pos['trailing_active'],
                        'trailing_stop': pos.get('trailing_stop_price')
                    })

            self.msleep(self.check_interval)

        print("[Monitor] Position monitor stopped")

    def stop(self):
        """Stop the monitor."""
        self.running = False

    def get_monitored_positions(self) -> Dict:
        """Get all monitored positions."""
        self.mutex.lock()
        positions = dict(self.positions)
        self.mutex.unlock()
        return positions

    def sync_with_broker(self, default_stop_pct: float = 0.02, default_profit_pct: float = 0.05):
        """Sync monitored positions with actual broker positions."""
        if not self.broker:
            return

        try:
            broker_positions = self.broker.get_positions()
            broker_symbols = {p.symbol for p in broker_positions}

            self.mutex.lock()
            monitored_symbols = set(self.positions.keys())

            # Remove positions that are no longer in broker
            for symbol in monitored_symbols - broker_symbols:
                if symbol in self.positions:
                    # Position was closed externally
                    pos = self.positions[symbol]
                    self.trade_logger.log_exit(pos['trade_id'], pos['current_price'], 'external')
                    del self.positions[symbol]
                    print(f"[Monitor] {symbol} removed (closed externally)")

            self.mutex.unlock()

            # ADD untracked positions with default stop/profit levels
            for pos in broker_positions:
                if pos.symbol not in monitored_symbols:
                    entry_price = float(pos.avg_entry_price)
                    shares = float(pos.qty)

                    # Determine direction
                    if shares > 0:  # Long position
                        stop_loss = entry_price * (1 - default_stop_pct)
                        take_profit = entry_price * (1 + default_profit_pct)
                    else:  # Short position
                        stop_loss = entry_price * (1 + default_stop_pct)
                        take_profit = entry_price * (1 - default_profit_pct)

                    self.add_position(
                        symbol=pos.symbol,
                        entry_price=entry_price,
                        shares=abs(shares),
                        stop_loss=stop_loss,
                        take_profit=take_profit,
                        trailing_activation_pct=default_stop_pct * 0.75,
                        trailing_stop_pct=default_stop_pct * 0.5,
                        ai_model='existing_position',
                        trade_type='synced'
                    )
                    print(f"[Monitor] Added existing position: {pos.symbol} @ ${entry_price:.2f}")

        except Exception as e:
            print(f"[Monitor] Sync error: {e}")
