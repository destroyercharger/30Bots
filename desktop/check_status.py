"""Quick status check script"""
import sys
sys.path.insert(0, '.')

from data.broker_adapter import AlpacaBroker
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

print("="*50)
print("TRADING STATUS")
print("="*50)

broker = AlpacaBroker(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
account = broker.get_account()
equity = float(account.equity)
cash = float(account.cash)
bp = float(account.buying_power)

print(f"Account Equity: ${equity:,.2f}")
print(f"Cash: ${cash:,.2f}")
print(f"Buying Power: ${bp:,.2f}")

positions = broker.get_positions()
print(f"\nOpen Positions: {len(positions)}")

total_pnl = 0
long_count = 0
short_count = 0
long_pnl = 0
short_pnl = 0

for pos in positions:
    pnl = float(pos.unrealized_pl)
    total_pnl += pnl
    qty = float(pos.qty)
    if qty > 0:
        long_count += 1
        long_pnl += pnl
    else:
        short_count += 1
        short_pnl += pnl

print(f"\nLong positions: {long_count} (P/L: ${long_pnl:+,.2f})")
print(f"Short positions: {short_count} (P/L: ${short_pnl:+,.2f})")
print(f"\nTotal Unrealized P/L: ${total_pnl:+,.2f}")

# Show biggest winners/losers
sorted_pos = sorted(positions, key=lambda p: float(p.unrealized_pl), reverse=True)

print("\nTop Winners:")
for pos in sorted_pos[:3]:
    pnl = float(pos.unrealized_pl)
    pnl_pct = float(pos.unrealized_plpc) * 100
    print(f"  {pos.symbol}: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")

print("\nTop Losers:")
for pos in sorted_pos[-3:]:
    pnl = float(pos.unrealized_pl)
    pnl_pct = float(pos.unrealized_plpc) * 100
    print(f"  {pos.symbol}: ${pnl:+,.2f} ({pnl_pct:+.1f}%)")

print("\n" + "="*50)
