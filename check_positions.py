"""Check current positions and P&L"""
import sys
sys.path.insert(0, r'D:\Projects\Stock-Trading-AI')

from dotenv import load_dotenv
import os
load_dotenv(r'D:\Projects\Stock-Trading-AI\.env')

from alpaca.trading.client import TradingClient

client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)

# Get account
account = client.get_account()
print('=' * 60)
print('ACCOUNT SUMMARY')
print('=' * 60)
print(f'Equity:        ${float(account.equity):,.2f}')
print(f'Cash:          ${float(account.cash):,.2f}')
print(f'Buying Power:  ${float(account.buying_power):,.2f}')
day_pnl = float(account.equity) - float(account.last_equity)
print(f'Day P&L:       ${day_pnl:+,.2f}')
print()

# Get positions
positions = client.get_all_positions()
print('=' * 60)
print('CURRENT POSITIONS')
print('=' * 60)
print(f"{'Symbol':<8} {'Side':<6} {'Qty':<8} {'Entry':>10} {'Current':>10} {'P&L':>12} {'P&L %':>8}")
print('-' * 60)

total_pnl = 0
for p in sorted(positions, key=lambda x: abs(float(x.unrealized_pl)), reverse=True):
    qty = float(p.qty)
    entry = float(p.avg_entry_price)
    current = float(p.current_price)
    pnl = float(p.unrealized_pl)
    pnl_pct = float(p.unrealized_plpc) * 100
    side = 'LONG' if qty > 0 else 'SHORT'
    total_pnl += pnl
    print(f'{p.symbol:<8} {side:<6} {abs(qty):<8.0f} ${entry:>9,.2f} ${current:>9,.2f} ${pnl:>+11,.2f} {pnl_pct:>+7.2f}%')

print('-' * 60)
print(f'Total Unrealized P&L: ${total_pnl:+,.2f}')
print(f'Positions: {len(positions)}')
