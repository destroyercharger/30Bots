"""Analyze why positions closed - stops or profits"""
import sys
sys.path.insert(0, '.')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import QueryOrderStatus
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from collections import defaultdict

print("="*70)
print("ANALYZING POSITION EXITS - STOPS OR PROFITS?")
print("="*70)

client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

request = GetOrdersRequest(status=QueryOrderStatus.ALL, limit=100)
orders = client.get_orders(request)

# Group orders by symbol to find entry/exit pairs
symbol_orders = defaultdict(list)
for order in orders:
    if order.status.value == 'filled':
        symbol_orders[order.symbol].append({
            'time': order.filled_at,
            'side': order.side.value,
            'qty': float(order.filled_qty),
            'price': float(order.filled_avg_price)
        })

print("\nAnalyzing entry/exit pairs:\n")

for symbol in sorted(symbol_orders.keys()):
    orders_list = sorted(symbol_orders[symbol], key=lambda x: x['time'])

    # Find BUY then SELL pairs (long trades)
    buys = [o for o in orders_list if o['side'] == 'buy']
    sells = [o for o in orders_list if o['side'] == 'sell']

    if buys and sells:
        # Get most recent buy and sell
        last_buy = buys[-1]
        last_sell = sells[-1] if sells else None

        if last_sell and last_sell['time'] > last_buy['time']:
            entry = last_buy['price']
            exit_price = last_sell['price']
            pnl_pct = (exit_price - entry) / entry * 100

            # Determine if stop or profit
            if pnl_pct > 1.5:
                reason = "TAKE PROFIT"
            elif pnl_pct < -1.0:
                reason = "STOP LOSS"
            else:
                reason = "SMALL MOVE"

            print(f"{symbol}:")
            print(f"  Entry: ${entry:.2f} -> Exit: ${exit_price:.2f}")
            print(f"  P/L: {pnl_pct:+.2f}% = {reason}")
            print()

print("="*70)
