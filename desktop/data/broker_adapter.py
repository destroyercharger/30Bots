"""
Broker Adapter
Alpaca integration for paper and live trading
"""

import os
import time
from typing import Optional, Dict, List, Any
from datetime import datetime, timedelta
from enum import Enum
from dataclasses import dataclass
from abc import ABC, abstractmethod

import requests
import numpy as np

# Try to import Alpaca SDK for better data access
try:
    from alpaca.data.historical import StockHistoricalDataClient
    from alpaca.data.requests import StockBarsRequest
    from alpaca.data.timeframe import TimeFrame, TimeFrameUnit
    ALPACA_SDK_AVAILABLE = True
except ImportError:
    ALPACA_SDK_AVAILABLE = False


class OrderSide(Enum):
    BUY = "buy"
    SELL = "sell"


class OrderType(Enum):
    MARKET = "market"
    LIMIT = "limit"
    STOP = "stop"
    STOP_LIMIT = "stop_limit"
    TRAILING_STOP = "trailing_stop"


class OrderStatus(Enum):
    NEW = "new"
    PENDING_NEW = "pending_new"
    ACCEPTED = "accepted"
    PARTIALLY_FILLED = "partially_filled"
    FILLED = "filled"
    CANCELED = "canceled"
    REJECTED = "rejected"
    EXPIRED = "expired"


class TimeInForce(Enum):
    DAY = "day"
    GTC = "gtc"
    IOC = "ioc"
    FOK = "fok"
    OPG = "opg"
    CLS = "cls"


@dataclass
class Position:
    symbol: str
    qty: float
    market_value: float
    cost_basis: float
    unrealized_pl: float
    unrealized_plpc: float
    current_price: float
    avg_entry_price: float
    side: str

    @classmethod
    def from_alpaca(cls, data: Dict) -> 'Position':
        return cls(
            symbol=data['symbol'],
            qty=float(data['qty']),
            market_value=float(data['market_value']),
            cost_basis=float(data['cost_basis']),
            unrealized_pl=float(data['unrealized_pl']),
            unrealized_plpc=float(data['unrealized_plpc']),
            current_price=float(data['current_price']),
            avg_entry_price=float(data['avg_entry_price']),
            side=data['side']
        )


@dataclass
class Order:
    id: str
    client_order_id: str
    symbol: str
    qty: float
    filled_qty: float
    side: OrderSide
    type: OrderType
    status: OrderStatus
    time_in_force: TimeInForce
    limit_price: Optional[float]
    stop_price: Optional[float]
    filled_avg_price: Optional[float]
    created_at: datetime
    updated_at: datetime

    @classmethod
    def from_alpaca(cls, data: Dict) -> 'Order':
        return cls(
            id=data['id'],
            client_order_id=data['client_order_id'],
            symbol=data['symbol'],
            qty=float(data['qty']),
            filled_qty=float(data.get('filled_qty', 0) or 0),
            side=OrderSide(data['side']),
            type=OrderType(data['type']),
            status=OrderStatus(data['status']),
            time_in_force=TimeInForce(data['time_in_force']),
            limit_price=float(data['limit_price']) if data.get('limit_price') else None,
            stop_price=float(data['stop_price']) if data.get('stop_price') else None,
            filled_avg_price=float(data['filled_avg_price']) if data.get('filled_avg_price') else None,
            created_at=datetime.fromisoformat(data['created_at'].replace('Z', '+00:00')),
            updated_at=datetime.fromisoformat(data['updated_at'].replace('Z', '+00:00'))
        )


@dataclass
class Account:
    id: str
    equity: float
    cash: float
    buying_power: float
    portfolio_value: float
    currency: str
    pattern_day_trader: bool
    trading_blocked: bool
    transfers_blocked: bool
    account_blocked: bool
    daytrade_count: int
    status: str

    @classmethod
    def from_alpaca(cls, data: Dict) -> 'Account':
        return cls(
            id=data['id'],
            equity=float(data['equity']),
            cash=float(data['cash']),
            buying_power=float(data['buying_power']),
            portfolio_value=float(data['portfolio_value']),
            currency=data['currency'],
            pattern_day_trader=data['pattern_day_trader'],
            trading_blocked=data['trading_blocked'],
            transfers_blocked=data['transfers_blocked'],
            account_blocked=data['account_blocked'],
            daytrade_count=int(data['daytrade_count']),
            status=data['status']
        )


class BaseBroker(ABC):
    """Abstract base class for broker implementations"""

    @abstractmethod
    def get_account(self) -> Account:
        """Get account information"""
        pass

    @abstractmethod
    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        pass

    @abstractmethod
    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        pass

    @abstractmethod
    def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        extended_hours: bool = False
    ) -> Order:
        """Place an order"""
        pass

    @abstractmethod
    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        pass

    @abstractmethod
    def get_order(self, order_id: str) -> Order:
        """Get order by ID"""
        pass

    @abstractmethod
    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        """Get orders"""
        pass


class AlpacaBroker(BaseBroker):
    """Alpaca broker implementation"""

    # API URLs
    PAPER_URL = "https://paper-api.alpaca.markets"
    LIVE_URL = "https://api.alpaca.markets"
    DATA_URL = "https://data.alpaca.markets"

    def __init__(
        self,
        api_key: str = None,
        secret_key: str = None,
        paper: bool = True
    ):
        self.api_key = api_key or os.environ.get('ALPACA_API_KEY', '')
        self.secret_key = secret_key or os.environ.get('ALPACA_SECRET_KEY', '')
        self.paper = paper

        self.base_url = self.PAPER_URL if paper else self.LIVE_URL

        self.session = requests.Session()
        self.session.headers.update({
            'APCA-API-KEY-ID': self.api_key,
            'APCA-API-SECRET-KEY': self.secret_key,
            'Content-Type': 'application/json'
        })

        # Initialize Alpaca SDK client for better data access
        self.data_client = None
        if ALPACA_SDK_AVAILABLE:
            try:
                self.data_client = StockHistoricalDataClient(self.api_key, self.secret_key)
            except Exception as e:
                print(f"Warning: Could not initialize Alpaca SDK: {e}")

    def _request(
        self,
        method: str,
        endpoint: str,
        data: Dict = None,
        params: Dict = None,
        use_data_api: bool = False
    ) -> Dict:
        """Make API request"""
        base = self.DATA_URL if use_data_api else self.base_url
        url = f"{base}{endpoint}"

        response = self.session.request(
            method=method,
            url=url,
            json=data,
            params=params
        )

        response.raise_for_status()
        return response.json() if response.text else {}

    def get_account(self) -> Account:
        """Get account information"""
        data = self._request('GET', '/v2/account')
        return Account.from_alpaca(data)

    def get_positions(self) -> List[Position]:
        """Get all open positions"""
        data = self._request('GET', '/v2/positions')
        return [Position.from_alpaca(p) for p in data]

    def get_position(self, symbol: str) -> Optional[Position]:
        """Get position for a specific symbol"""
        try:
            data = self._request('GET', f'/v2/positions/{symbol}')
            return Position.from_alpaca(data)
        except requests.HTTPError as e:
            if e.response.status_code == 404:
                return None
            raise

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        trail_price: Optional[float] = None,
        extended_hours: bool = False
    ) -> Order:
        """Place an order"""
        order_data = {
            'symbol': symbol,
            'qty': str(qty),
            'side': side.value,
            'type': type.value,
            'time_in_force': time_in_force.value,
            'extended_hours': extended_hours
        }

        if limit_price is not None:
            order_data['limit_price'] = str(limit_price)

        if stop_price is not None:
            order_data['stop_price'] = str(stop_price)

        if trail_percent is not None:
            order_data['trail_percent'] = str(trail_percent)

        if trail_price is not None:
            order_data['trail_price'] = str(trail_price)

        data = self._request('POST', '/v2/orders', data=order_data)
        return Order.from_alpaca(data)

    def place_bracket_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        take_profit_price: float,
        stop_loss_price: float,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Order:
        """Place a bracket order (entry + take profit + stop loss)"""
        order_type = OrderType.LIMIT if limit_price else OrderType.MARKET

        order_data = {
            'symbol': symbol,
            'qty': str(qty),
            'side': side.value,
            'type': order_type.value,
            'time_in_force': time_in_force.value,
            'order_class': 'bracket',
            'take_profit': {
                'limit_price': str(take_profit_price)
            },
            'stop_loss': {
                'stop_price': str(stop_loss_price)
            }
        }

        if limit_price:
            order_data['limit_price'] = str(limit_price)

        data = self._request('POST', '/v2/orders', data=order_data)
        return Order.from_alpaca(data)

    def place_oto_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        stop_loss_price: float,
        limit_price: Optional[float] = None,
        time_in_force: TimeInForce = TimeInForce.DAY
    ) -> Order:
        """Place a one-triggers-other order (entry + stop loss)"""
        order_type = OrderType.LIMIT if limit_price else OrderType.MARKET

        order_data = {
            'symbol': symbol,
            'qty': str(qty),
            'side': side.value,
            'type': order_type.value,
            'time_in_force': time_in_force.value,
            'order_class': 'oto',
            'stop_loss': {
                'stop_price': str(stop_loss_price)
            }
        }

        if limit_price:
            order_data['limit_price'] = str(limit_price)

        data = self._request('POST', '/v2/orders', data=order_data)
        return Order.from_alpaca(data)

    def cancel_order(self, order_id: str) -> bool:
        """Cancel an order"""
        try:
            self._request('DELETE', f'/v2/orders/{order_id}')
            return True
        except requests.HTTPError:
            return False

    def cancel_all_orders(self) -> bool:
        """Cancel all open orders"""
        try:
            self._request('DELETE', '/v2/orders')
            return True
        except requests.HTTPError:
            return False

    def get_order(self, order_id: str) -> Order:
        """Get order by ID"""
        data = self._request('GET', f'/v2/orders/{order_id}')
        return Order.from_alpaca(data)

    def get_orders(
        self,
        status: Optional[str] = None,
        limit: int = 100,
        after: Optional[datetime] = None,
        until: Optional[datetime] = None,
        direction: str = 'desc'
    ) -> List[Order]:
        """Get orders"""
        params = {
            'limit': limit,
            'direction': direction
        }

        if status:
            params['status'] = status
        if after:
            params['after'] = after.isoformat()
        if until:
            params['until'] = until.isoformat()

        data = self._request('GET', '/v2/orders', params=params)
        return [Order.from_alpaca(o) for o in data]

    def close_position(self, symbol: str) -> Order:
        """Close a position"""
        data = self._request('DELETE', f'/v2/positions/{symbol}')
        return Order.from_alpaca(data)

    def close_all_positions(self) -> List[Order]:
        """Close all positions"""
        data = self._request('DELETE', '/v2/positions')
        return [Order.from_alpaca(o) for o in data]

    def wait_for_fill(
        self,
        order_id: str,
        timeout: float = 60,
        poll_interval: float = 0.5
    ) -> Order:
        """Wait for an order to fill"""
        start_time = time.time()

        while time.time() - start_time < timeout:
            order = self.get_order(order_id)

            if order.status in [OrderStatus.FILLED, OrderStatus.CANCELED, OrderStatus.REJECTED, OrderStatus.EXPIRED]:
                return order

            time.sleep(poll_interval)

        raise TimeoutError(f"Order {order_id} did not fill within {timeout} seconds")

    def get_clock(self) -> Dict:
        """Get market clock"""
        return self._request('GET', '/v2/clock')

    def get_calendar(
        self,
        start: Optional[str] = None,
        end: Optional[str] = None
    ) -> List[Dict]:
        """Get market calendar"""
        params = {}
        if start:
            params['start'] = start
        if end:
            params['end'] = end

        return self._request('GET', '/v2/calendar', params=params)

    def get_bars(
        self,
        symbol: str,
        timeframe: str = '1Day',
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 100
    ) -> List[Dict]:
        """Get historical bars using Alpaca SDK for better data access"""

        # Use SDK if available (handles auth and data feed properly)
        if self.data_client and ALPACA_SDK_AVAILABLE:
            try:
                # Map timeframe to SDK TimeFrame
                tf_map = {
                    '1Min': TimeFrame.Minute,
                    '5Min': TimeFrame(5, TimeFrameUnit.Minute),
                    '15Min': TimeFrame(15, TimeFrameUnit.Minute),
                    '30Min': TimeFrame(30, TimeFrameUnit.Minute),
                    '1Hour': TimeFrame.Hour,
                    '4Hour': TimeFrame(4, TimeFrameUnit.Hour),
                    '1Day': TimeFrame.Day,
                    '1Week': TimeFrame.Week,
                }
                sdk_timeframe = tf_map.get(timeframe, TimeFrame.Day)

                # Calculate date range
                end_dt = datetime.now()
                if timeframe in ['1Min', '5Min', '15Min', '30Min']:
                    start_dt = end_dt - timedelta(days=7)
                elif timeframe in ['1Hour', '4Hour']:
                    start_dt = end_dt - timedelta(days=30)
                else:
                    start_dt = end_dt - timedelta(days=120)

                request = StockBarsRequest(
                    symbol_or_symbols=symbol,
                    timeframe=sdk_timeframe,
                    start=start_dt,
                    end=end_dt
                )

                bars_data = self.data_client.get_stock_bars(request)

                # BarSet has .data property containing the actual data
                data_dict = bars_data.data if hasattr(bars_data, 'data') else bars_data

                if symbol in data_dict:
                    bars = []
                    for bar in data_dict[symbol]:
                        bars.append({
                            't': bar.timestamp.isoformat(),
                            'o': float(bar.open),
                            'h': float(bar.high),
                            'l': float(bar.low),
                            'c': float(bar.close),
                            'v': int(bar.volume),
                            'vw': float(bar.vwap) if bar.vwap else float(bar.close)
                        })
                    return bars[-limit:] if len(bars) > limit else bars
                return []

            except Exception as e:
                print(f"SDK bar fetch error: {e}, falling back to REST")

        # Fallback to REST API
        params = {
            'timeframe': timeframe,
            'limit': limit
        }

        if start:
            params['start'] = start
        if end:
            params['end'] = end

        try:
            data = self._request(
                'GET',
                f'/v2/stocks/{symbol}/bars',
                params=params,
                use_data_api=True
            )
            return data.get('bars', [])
        except Exception as e:
            print(f"REST bar fetch error for {symbol}: {e}")
            return []

    def get_latest_trade(self, symbol: str) -> Dict:
        """Get latest trade for a symbol"""
        return self._request(
            'GET',
            f'/v2/stocks/{symbol}/trades/latest',
            use_data_api=True
        )

    def get_latest_quote(self, symbol: str) -> Dict:
        """Get latest quote for a symbol"""
        return self._request(
            'GET',
            f'/v2/stocks/{symbol}/quotes/latest',
            use_data_api=True
        )

    def is_market_open(self) -> bool:
        """Check if market is currently open"""
        clock = self.get_clock()
        return clock.get('is_open', False)

    def get_bars_numpy(
        self,
        symbol: str,
        timeframe: str = '5Min',
        start: Optional[str] = None,
        end: Optional[str] = None,
        limit: int = 200
    ) -> 'np.ndarray':
        """
        Get historical bars as numpy array for charting

        Args:
            symbol: Stock symbol
            timeframe: Alpaca timeframe (1Min, 5Min, 15Min, 30Min, 1Hour, 1Day)
            start: Start date (RFC3339 or YYYY-MM-DD)
            end: End date (RFC3339 or YYYY-MM-DD)
            limit: Max bars to return

        Returns:
            numpy array [time, open, high, low, close, volume]
        """
        import numpy as np

        bars = self.get_bars(
            symbol=symbol,
            timeframe=timeframe,
            start=start,
            end=end,
            limit=limit
        )

        if not bars:
            return np.array([])

        data = []
        for bar in bars:
            # Parse timestamp
            ts = bar.get('t', '')
            if isinstance(ts, str):
                dt = datetime.fromisoformat(ts.replace('Z', '+00:00'))
                timestamp = dt.timestamp()
            else:
                timestamp = ts

            data.append([
                timestamp,
                float(bar.get('o', 0)),
                float(bar.get('h', 0)),
                float(bar.get('l', 0)),
                float(bar.get('c', 0)),
                float(bar.get('v', 0))
            ])

        return np.array(data)

    def get_snapshot(self, symbol: str) -> Dict:
        """Get current snapshot for a symbol"""
        return self._request(
            'GET',
            f'/v2/stocks/{symbol}/snapshot',
            use_data_api=True
        )


def timeframe_to_alpaca(timeframe: str) -> str:
    """
    Convert simple timeframe string to Alpaca format

    Args:
        timeframe: '1m', '5m', '15m', '30m', '1h', '4h', '1d'

    Returns:
        Alpaca timeframe string
    """
    mapping = {
        '1m': '1Min',
        '5m': '5Min',
        '15m': '15Min',
        '30m': '30Min',
        '1h': '1Hour',
        '4h': '4Hour',
        '1d': '1Day',
        '1w': '1Week',
    }
    return mapping.get(timeframe, '5Min')


def get_lookback_days(timeframe: str) -> int:
    """
    Get appropriate lookback days for a timeframe

    Args:
        timeframe: '1m', '5m', '15m', '30m', '1h', '4h', '1d'

    Returns:
        Number of days to look back
    """
    mapping = {
        '1m': 1,
        '5m': 5,
        '15m': 10,
        '30m': 20,
        '1h': 30,
        '4h': 60,
        '1d': 365,
        '1w': 730,
    }
    return mapping.get(timeframe, 5)


class MockBroker(BaseBroker):
    """Mock broker for testing"""

    def __init__(self, initial_cash: float = 100000.0):
        self.cash = initial_cash
        self.positions: Dict[str, Position] = {}
        self.orders: Dict[str, Order] = {}
        self.order_counter = 0

    def get_account(self) -> Account:
        portfolio_value = self.cash + sum(p.market_value for p in self.positions.values())

        return Account(
            id='mock-account',
            equity=portfolio_value,
            cash=self.cash,
            buying_power=self.cash * 4,  # 4x margin
            portfolio_value=portfolio_value,
            currency='USD',
            pattern_day_trader=False,
            trading_blocked=False,
            transfers_blocked=False,
            account_blocked=False,
            daytrade_count=0,
            status='ACTIVE'
        )

    def get_positions(self) -> List[Position]:
        return list(self.positions.values())

    def get_position(self, symbol: str) -> Optional[Position]:
        return self.positions.get(symbol)

    def place_order(
        self,
        symbol: str,
        qty: float,
        side: OrderSide,
        type: OrderType = OrderType.MARKET,
        time_in_force: TimeInForce = TimeInForce.DAY,
        limit_price: Optional[float] = None,
        stop_price: Optional[float] = None,
        trail_percent: Optional[float] = None,
        trail_price: Optional[float] = None,
        extended_hours: bool = False
    ) -> Order:
        self.order_counter += 1
        order_id = f'mock-order-{self.order_counter}'

        # Simulate immediate fill for market orders
        fill_price = limit_price or 100.0  # Default price for testing

        order = Order(
            id=order_id,
            client_order_id=order_id,
            symbol=symbol,
            qty=qty,
            filled_qty=qty if type == OrderType.MARKET else 0,
            side=side,
            type=type,
            status=OrderStatus.FILLED if type == OrderType.MARKET else OrderStatus.NEW,
            time_in_force=time_in_force,
            limit_price=limit_price,
            stop_price=stop_price,
            filled_avg_price=fill_price if type == OrderType.MARKET else None,
            created_at=datetime.now(),
            updated_at=datetime.now()
        )

        self.orders[order_id] = order

        # Update position for filled orders
        if order.status == OrderStatus.FILLED:
            self._update_position(symbol, qty, side, fill_price)

        return order

    def _update_position(self, symbol: str, qty: float, side: OrderSide, price: float):
        if side == OrderSide.BUY:
            if symbol in self.positions:
                pos = self.positions[symbol]
                new_qty = pos.qty + qty
                new_cost = pos.cost_basis + (qty * price)
                pos.qty = new_qty
                pos.cost_basis = new_cost
                pos.avg_entry_price = new_cost / new_qty
                pos.current_price = price
                pos.market_value = new_qty * price
            else:
                self.positions[symbol] = Position(
                    symbol=symbol,
                    qty=qty,
                    market_value=qty * price,
                    cost_basis=qty * price,
                    unrealized_pl=0,
                    unrealized_plpc=0,
                    current_price=price,
                    avg_entry_price=price,
                    side='long'
                )
            self.cash -= qty * price
        else:
            if symbol in self.positions:
                pos = self.positions[symbol]
                pos.qty -= qty
                if pos.qty <= 0:
                    del self.positions[symbol]
                else:
                    pos.market_value = pos.qty * price
            self.cash += qty * price

    def cancel_order(self, order_id: str) -> bool:
        if order_id in self.orders:
            self.orders[order_id].status = OrderStatus.CANCELED
            return True
        return False

    def get_order(self, order_id: str) -> Order:
        return self.orders[order_id]

    def get_orders(self, status: Optional[str] = None) -> List[Order]:
        orders = list(self.orders.values())
        if status:
            orders = [o for o in orders if o.status.value == status]
        return orders
