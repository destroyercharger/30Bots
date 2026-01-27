"""Full watchlist scan to find BUY signals"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

from data.broker_adapter import AlpacaBroker
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from ai.prediction_worker import AIPredictionWorker, get_ai_brain
from core.auto_trader import DEFAULT_DAY_WATCHLIST

print("="*50)
print("FULL WATCHLIST SCAN")
print("="*50)

# Setup
broker = AlpacaBroker(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
brain = get_ai_brain()
worker = AIPredictionWorker()
worker.set_broker(broker)

print(f"Scanning {len(DEFAULT_DAY_WATCHLIST)} symbols...")
print("Looking for BUY signals >= 65% confidence\n")

buy_signals = []
sell_signals = []

for symbol in DEFAULT_DAY_WATCHLIST:
    try:
        prediction = worker.get_prediction_for_symbol(symbol)
        if prediction:
            action = prediction.get('action', 'HOLD')
            confidence = prediction.get('confidence', 0)
            model = prediction.get('model', 'Unknown')

            if action == 'BUY' and confidence >= 0.65:
                buy_signals.append((symbol, confidence, model))
                print(f"  BUY  {symbol}: {confidence*100:.1f}% ({model})")
            elif action == 'SELL' and confidence >= 0.65:
                sell_signals.append((symbol, confidence, model))
                print(f"  SELL {symbol}: {confidence*100:.1f}% ({model})")
    except Exception as e:
        print(f"  ERROR {symbol}: {e}")

print(f"\n{'='*50}")
print(f"SUMMARY")
print(f"{'='*50}")
print(f"BUY signals (>= 65%): {len(buy_signals)}")
for s, c, m in sorted(buy_signals, key=lambda x: -x[1]):
    print(f"  {s}: {c*100:.1f}% - {m}")

print(f"\nSELL signals (>= 65%): {len(sell_signals)}")
for s, c, m in sorted(sell_signals, key=lambda x: -x[1]):
    print(f"  {s}: {c*100:.1f}% - {m}")

if buy_signals:
    top = buy_signals[0]
    print(f"\n** TOP BUY: {top[0]} at {top[1]*100:.1f}% confidence **")
else:
    print("\n** No actionable BUY signals found **")
