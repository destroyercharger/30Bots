"""
Check positions grouped by AI strategy
"""
import sys
sys.path.insert(0, r'D:\Projects\Stock-Trading-AI')
from dotenv import load_dotenv
import os
load_dotenv(r'D:\Projects\Stock-Trading-AI\.env')

from alpaca.trading.client import TradingClient
import sqlite3

client = TradingClient(os.getenv('ALPACA_API_KEY'), os.getenv('ALPACA_SECRET_KEY'), paper=True)
positions = client.get_all_positions()

# Try to load position monitor data for strategy info
open_trades = {}
try:
    db_path = r'D:\Projects\30Bots\desktop\data\trades.db'
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()

    # Get recent trades with model info (both open and recent)
    cursor.execute('''
        SELECT symbol, ai_model, side, entry_price, entry_time
        FROM trades
        ORDER BY entry_time DESC
        LIMIT 50
    ''')
    for row in cursor.fetchall():
        if row[0] not in open_trades:  # Keep most recent
            open_trades[row[0]] = {'model': row[1] or 'Unknown', 'side': row[2]}
    conn.close()
except Exception as e:
    print(f'Note: Could not load trade history: {e}')

print('=' * 70)
print('POSITIONS BY STRATEGY')
print('=' * 70)

# Group by strategy
strategies = {}
for p in positions:
    symbol = p.symbol
    qty = float(p.qty)
    side = 'LONG' if qty > 0 else 'SHORT'
    pnl = float(p.unrealized_pl)
    entry = float(p.avg_entry_price)
    current = float(p.current_price)

    # Get model/strategy from trade history
    trade_info = open_trades.get(symbol, {})
    model = trade_info.get('model', 'Unknown')

    # Extract strategy from model name
    if model and '_' in model:
        strategy = model.rsplit('_', 1)[0]
    else:
        strategy = model if model and model != 'Unknown' else 'Manual/Unknown'

    if strategy not in strategies:
        strategies[strategy] = []
    strategies[strategy].append({
        'symbol': symbol,
        'side': side,
        'model': model,
        'pnl': pnl,
        'entry': entry,
        'current': current
    })

# Print by strategy
for strategy, positions_list in sorted(strategies.items()):
    total_pnl = sum(p['pnl'] for p in positions_list)
    print(f'\n{strategy} ({len(positions_list)} positions, P&L: ${total_pnl:+.2f})')
    print('-' * 70)
    print(f'  {"Symbol":<8} {"Side":<6} {"Model":<22} {"Entry":>10} {"Current":>10} {"P&L":>10}')
    print(f'  {"-"*66}')
    for p in positions_list:
        model_short = p["model"][:22] if p["model"] else "Unknown"
        print(f'  {p["symbol"]:<8} {p["side"]:<6} {model_short:<22} ${p["entry"]:>9.2f} ${p["current"]:>9.2f} ${p["pnl"]:>+9.2f}')

print('\n' + '=' * 70)

# Summary
print('\nSTRATEGY SUMMARY')
print('-' * 40)
for strategy, positions_list in sorted(strategies.items()):
    total_pnl = sum(p['pnl'] for p in positions_list)
    print(f'{strategy:<20} {len(positions_list)} pos  ${total_pnl:>+10.2f}')

print('-' * 40)
total_positions = sum(len(p) for p in strategies.values())
total_pnl = sum(sum(pos['pnl'] for pos in p) for p in strategies.values())
print(f'{"TOTAL":<20} {total_positions} pos  ${total_pnl:>+10.2f}')
print('=' * 70)
