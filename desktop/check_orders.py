"""Check recent orders from Alpaca"""
import sys
sys.path.insert(0, '.')

from alpaca.trading.client import TradingClient
from alpaca.trading.requests import GetOrdersRequest
from alpaca.trading.enums import OrderSide, QueryOrderStatus
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from datetime import datetime, timedelta

print("="*50)
print("RECENT ORDERS CHECK")
print("="*50)

client = TradingClient(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Get orders from today
request = GetOrdersRequest(
    status=QueryOrderStatus.ALL,
    limit=50
)
orders = client.get_orders(request)

print(f"Found {len(orders)} orders\n")

# Group by time
today_orders = []
for order in orders:
    created = order.created_at
    if created.date() >= datetime.now().date() - timedelta(days=1):
        today_orders.append(order)

print(f"Recent orders ({len(today_orders)}):")
for order in sorted(today_orders, key=lambda x: x.created_at, reverse=True)[:20]:
    side = order.side.value.upper()
    status = order.status.value
    time_str = order.created_at.strftime('%H:%M:%S')
    filled_qty = order.filled_qty or 0
    filled_price = order.filled_avg_price or 0
    print(f"  {time_str} {side:4} {order.qty:>4} {order.symbol:<6} @ ${float(filled_price):>8.2f} - {status}")

print("\n" + "="*50)
