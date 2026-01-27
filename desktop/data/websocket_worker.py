"""
Alpaca WebSocket Worker
Real-time price streaming using Alpaca's market data WebSocket
"""

import json
import threading
from typing import Optional, List, Set
from datetime import datetime

from PyQt6.QtCore import QThread, pyqtSignal, QMutex

try:
    import websocket
    WEBSOCKET_AVAILABLE = True
except ImportError:
    WEBSOCKET_AVAILABLE = False
    print("websocket-client not installed. Install with: pip install websocket-client")


class AlpacaWebSocketWorker(QThread):
    """
    WebSocket worker for real-time Alpaca market data streaming.

    Connects to Alpaca's market data WebSocket and emits signals
    for trades, quotes, and bars.
    """

    # Signals
    connected = pyqtSignal()
    disconnected = pyqtSignal()
    error = pyqtSignal(str)

    # Price updates
    trade_received = pyqtSignal(str, float, float, str)  # symbol, price, size, timestamp
    quote_received = pyqtSignal(str, float, float, float, float)  # symbol, bid, ask, bid_size, ask_size
    bar_received = pyqtSignal(str, float, float, float, float, float, str)  # symbol, o, h, l, c, v, timestamp

    # Alpaca WebSocket URLs
    IEX_URL = "wss://stream.data.alpaca.markets/v2/iex"  # Free tier
    SIP_URL = "wss://stream.data.alpaca.markets/v2/sip"  # Paid tier

    def __init__(
        self,
        api_key: str,
        secret_key: str,
        use_sip: bool = False,
        parent=None
    ):
        super().__init__(parent)
        self.api_key = api_key
        self.secret_key = secret_key
        self.url = self.SIP_URL if use_sip else self.IEX_URL

        self.ws: Optional[websocket.WebSocketApp] = None
        self.subscribed_symbols: Set[str] = set()
        self.running = False
        self.authenticated = False
        self.mutex = QMutex()

        # Pending subscriptions (before auth)
        self._pending_subs: List[str] = []

        # Retry control
        self._retry_count = 0
        self._max_retries = 3
        self._connection_limit_hit = False

    def run(self):
        """Main thread loop"""
        if not WEBSOCKET_AVAILABLE:
            self.error.emit("websocket-client not installed")
            return

        self.running = True
        self._retry_count = 0

        while self.running and self._retry_count < self._max_retries and not self._connection_limit_hit:
            try:
                self.ws = websocket.WebSocketApp(
                    self.url,
                    on_open=self._on_open,
                    on_message=self._on_message,
                    on_error=self._on_error,
                    on_close=self._on_close
                )

                # Run WebSocket (blocks until closed)
                self.ws.run_forever(
                    ping_interval=30,
                    ping_timeout=10
                )

            except Exception as e:
                self.error.emit(f"WebSocket error: {e}")

            # If still running, reconnect after delay
            if self.running and not self._connection_limit_hit:
                self._retry_count += 1
                if self._retry_count < self._max_retries:
                    self.msleep(10000)  # 10 second reconnect delay
                else:
                    print("[WebSocket] Max retries reached, stopping reconnection attempts")

    def stop(self):
        """Stop the WebSocket connection"""
        self.running = False
        if self.ws:
            self.ws.close()

    def subscribe(self, symbols: List[str], trades: bool = True, quotes: bool = True, bars: bool = False):
        """Subscribe to market data for symbols"""
        self.mutex.lock()
        try:
            new_symbols = [s.upper() for s in symbols if s.upper() not in self.subscribed_symbols]

            if not new_symbols:
                return

            if self.authenticated and self.ws:
                # Send subscription message
                msg = {"action": "subscribe"}
                if trades:
                    msg["trades"] = new_symbols
                if quotes:
                    msg["quotes"] = new_symbols
                if bars:
                    msg["bars"] = new_symbols

                self.ws.send(json.dumps(msg))
                self.subscribed_symbols.update(new_symbols)
            else:
                # Queue for after authentication
                self._pending_subs.extend(new_symbols)
        finally:
            self.mutex.unlock()

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from market data"""
        self.mutex.lock()
        try:
            symbols = [s.upper() for s in symbols if s.upper() in self.subscribed_symbols]

            if not symbols or not self.ws:
                return

            msg = {
                "action": "unsubscribe",
                "trades": symbols,
                "quotes": symbols
            }
            self.ws.send(json.dumps(msg))
            self.subscribed_symbols -= set(symbols)
        finally:
            self.mutex.unlock()

    def _on_open(self, ws):
        """Handle WebSocket connection opened"""
        # Authenticate
        auth_msg = {
            "action": "auth",
            "key": self.api_key,
            "secret": self.secret_key
        }
        ws.send(json.dumps(auth_msg))

    def _on_message(self, ws, message):
        """Handle incoming WebSocket message"""
        try:
            data = json.loads(message)

            if isinstance(data, list):
                for item in data:
                    self._process_message(item)
            else:
                self._process_message(data)

        except json.JSONDecodeError as e:
            self.error.emit(f"JSON decode error: {e}")

    def _process_message(self, msg: dict):
        """Process a single message"""
        msg_type = msg.get("T")

        if msg_type == "success":
            # Authentication or subscription success
            if msg.get("msg") == "authenticated":
                self.authenticated = True
                self.connected.emit()

                # Process pending subscriptions
                if self._pending_subs:
                    self.subscribe(self._pending_subs)
                    self._pending_subs.clear()

        elif msg_type == "error":
            error_msg = msg.get('msg', 'Unknown error')
            self.error.emit(f"Alpaca error: {error_msg}")

            # Stop retrying on connection limit errors
            if "connection limit" in error_msg.lower():
                self._connection_limit_hit = True

        elif msg_type == "t":
            # Trade
            symbol = msg.get("S", "")
            price = float(msg.get("p", 0))
            size = float(msg.get("s", 0))
            timestamp = msg.get("t", "")
            self.trade_received.emit(symbol, price, size, timestamp)

        elif msg_type == "q":
            # Quote
            symbol = msg.get("S", "")
            bid = float(msg.get("bp", 0))
            ask = float(msg.get("ap", 0))
            bid_size = float(msg.get("bs", 0))
            ask_size = float(msg.get("as", 0))
            self.quote_received.emit(symbol, bid, ask, bid_size, ask_size)

        elif msg_type == "b":
            # Bar (1-minute)
            symbol = msg.get("S", "")
            o = float(msg.get("o", 0))
            h = float(msg.get("h", 0))
            l = float(msg.get("l", 0))
            c = float(msg.get("c", 0))
            v = float(msg.get("v", 0))
            timestamp = msg.get("t", "")
            self.bar_received.emit(symbol, o, h, l, c, v, timestamp)

    def _on_error(self, ws, error):
        """Handle WebSocket error"""
        error_str = str(error)
        self.error.emit(error_str)

        # Stop retrying on connection limit errors
        if "connection limit" in error_str.lower():
            self._connection_limit_hit = True
            print("[WebSocket] Connection limit exceeded - disabling real-time streaming")
            print("[WebSocket] Trading will continue using REST API for prices")

    def _on_close(self, ws, close_status_code, close_msg):
        """Handle WebSocket connection closed"""
        self.authenticated = False
        self.disconnected.emit()


class PriceUpdateManager:
    """
    Manages real-time price updates for the application.

    Handles WebSocket connection, subscriptions, and price caching.
    """

    def __init__(self, api_key: str, secret_key: str):
        self.api_key = api_key
        self.secret_key = secret_key
        self.worker: Optional[AlpacaWebSocketWorker] = None
        self.prices: dict = {}  # symbol -> {price, bid, ask, timestamp}
        self.callbacks: List[callable] = []

    def start(self):
        """Start the WebSocket connection"""
        if not WEBSOCKET_AVAILABLE:
            print("WebSocket not available")
            return False

        if self.worker and self.worker.isRunning():
            return True

        self.worker = AlpacaWebSocketWorker(
            api_key=self.api_key,
            secret_key=self.secret_key
        )

        # Connect signals
        self.worker.trade_received.connect(self._on_trade)
        self.worker.quote_received.connect(self._on_quote)
        self.worker.connected.connect(self._on_connected)
        self.worker.disconnected.connect(self._on_disconnected)
        self.worker.error.connect(self._on_error)

        self.worker.start()
        return True

    def stop(self):
        """Stop the WebSocket connection"""
        if self.worker:
            self.worker.stop()
            self.worker.wait(5000)
            self.worker = None

    def subscribe(self, symbols: List[str]):
        """Subscribe to price updates for symbols"""
        if self.worker:
            self.worker.subscribe(symbols)

    def unsubscribe(self, symbols: List[str]):
        """Unsubscribe from symbols"""
        if self.worker:
            self.worker.unsubscribe(symbols)

    def get_price(self, symbol: str) -> Optional[float]:
        """Get the latest price for a symbol"""
        data = self.prices.get(symbol.upper())
        return data.get('price') if data else None

    def get_quote(self, symbol: str) -> Optional[dict]:
        """Get the latest quote for a symbol"""
        return self.prices.get(symbol.upper())

    def add_callback(self, callback: callable):
        """Add a callback for price updates: callback(symbol, price, bid, ask)"""
        self.callbacks.append(callback)

    def remove_callback(self, callback: callable):
        """Remove a callback"""
        if callback in self.callbacks:
            self.callbacks.remove(callback)

    def _on_trade(self, symbol: str, price: float, size: float, timestamp: str):
        """Handle trade update"""
        if symbol not in self.prices:
            self.prices[symbol] = {}
        self.prices[symbol]['price'] = price
        self.prices[symbol]['last_trade'] = timestamp

        # Notify callbacks
        data = self.prices[symbol]
        for cb in self.callbacks:
            try:
                cb(symbol, price, data.get('bid', price), data.get('ask', price))
            except Exception as e:
                print(f"Callback error: {e}")

    def _on_quote(self, symbol: str, bid: float, ask: float, bid_size: float, ask_size: float):
        """Handle quote update"""
        if symbol not in self.prices:
            self.prices[symbol] = {}
        self.prices[symbol]['bid'] = bid
        self.prices[symbol]['ask'] = ask
        self.prices[symbol]['bid_size'] = bid_size
        self.prices[symbol]['ask_size'] = ask_size

        # Use mid price if no trade price
        if 'price' not in self.prices[symbol]:
            self.prices[symbol]['price'] = (bid + ask) / 2

    def _on_connected(self):
        """Handle connection"""
        print("WebSocket connected to Alpaca")

    def _on_disconnected(self):
        """Handle disconnection"""
        print("WebSocket disconnected from Alpaca")

    def _on_error(self, error: str):
        """Handle error"""
        print(f"WebSocket error: {error}")
