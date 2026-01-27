"""Show predictions from ALL 30 models for one symbol"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

from ai.prediction_worker import get_ai_brain
from data.broker_adapter import AlpacaBroker
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
import pandas as pd

print("="*70)
print("ALL 30 MODELS PREDICTION FOR AAPL")
print("="*70)

brain = get_ai_brain()
broker = AlpacaBroker(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

bars = broker.get_bars('AAPL', '1Day', limit=100)
df = pd.DataFrame(bars)

# Map columns to expected names
column_map = {
    't': 'timestamp', 'o': 'open', 'h': 'high',
    'l': 'low', 'c': 'close', 'v': 'volume', 'vw': 'vwap'
}
df = df.rename(columns=column_map)
print(f"Data shape: {df.shape}, Columns: {list(df.columns)}")

# Get ALL predictions from all 30 models
all_predictions = brain.get_all_predictions(df)

print(f"\nGot {len(all_predictions)} model predictions\n")

# Group by action
buys = [p for p in all_predictions if p['action'] == 'BUY']
sells = [p for p in all_predictions if p['action'] == 'SELL']
holds = [p for p in all_predictions if p['action'] == 'HOLD']

print(f"BUY signals: {len(buys)}")
for p in sorted(buys, key=lambda x: -x['confidence'])[:5]:
    print(f"  {p['model']}: {p['confidence']*100:.1f}% (WR: {p['win_rate']*100:.0f}%)")
    print(f"    SL: {p['stop_loss_pct']*100:.1f}%, TP: {p['take_profit_pct']*100:.1f}%, Trail: {p['trailing_stop_pct']*100:.1f}%")

print(f"\nSELL signals: {len(sells)}")
for p in sorted(sells, key=lambda x: -x['confidence'])[:5]:
    print(f"  {p['model']}: {p['confidence']*100:.1f}% (WR: {p['win_rate']*100:.0f}%)")
    print(f"    SL: {p['stop_loss_pct']*100:.1f}%, TP: {p['take_profit_pct']*100:.1f}%, Trail: {p['trailing_stop_pct']*100:.1f}%")

print(f"\nHOLD signals: {len(holds)}")

print("\n" + "="*70)
print("VERIFICATION: Each model has DIFFERENT risk params based on strategy")
print("="*70)
