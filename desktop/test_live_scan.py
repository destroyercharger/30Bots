#!/usr/bin/env python3
"""
Live scan for AI trading signals
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from data.broker_adapter import AlpacaBroker
from ai.prediction_worker import AIPredictionWorker
from core.trade_logger import get_trade_logger
from core.auto_trader import DEFAULT_DAY_WATCHLIST, DEFAULT_SWING_WATCHLIST

print("=" * 70)
print(f"30Bots Live AI Scan - {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
print("=" * 70)

# Connect
broker = AlpacaBroker(api_key=ALPACA_API_KEY, secret_key=ALPACA_SECRET_KEY, paper=True)
worker = AIPredictionWorker(broker=broker)

account = broker.get_account()
print(f"\nAccount: ${float(account.equity):,.2f} | Cash: ${float(account.cash):,.2f}")
print(f"Market Open: {broker.is_market_open()}")

# Extended watchlist
extended_watchlist = list(set(DEFAULT_DAY_WATCHLIST + DEFAULT_SWING_WATCHLIST + [
    'BA', 'DIS', 'INTC', 'MU', 'UBER', 'LYFT', 'SNAP', 'PINS',
    'SQ', 'PYPL', 'SHOP', 'ZM', 'DOCU', 'CRWD', 'NET', 'DDOG',
    'ABNB', 'DASH', 'RBLX', 'ROKU', 'BABA', 'NIO', 'XPEV', 'LI'
]))

print(f"\nScanning {len(extended_watchlist)} symbols for signals...\n")

signals = []
for i, symbol in enumerate(extended_watchlist):
    print(f"[{i+1:2}/{len(extended_watchlist)}] {symbol:6}", end=" ")

    try:
        pred = worker.get_prediction_for_symbol(symbol)

        if pred:
            action = pred.get('action', 'HOLD')
            conf = pred.get('confidence', 0)
            model = pred.get('model', 'Unknown')

            if action in ['BUY', 'SELL']:
                stop_pct = pred.get('stop_loss_pct', 0.02) * 100
                target_pct = pred.get('take_profit_pct', 0.06) * 100
                rr = target_pct / stop_pct if stop_pct > 0 else 0

                print(f">>> {action} {conf*100:.1f}% ({model}) SL:{stop_pct:.1f}% TP:{target_pct:.1f}% R:R={rr:.1f}")
                signals.append(pred)
            else:
                print(f"HOLD (signal: {pred.get('signal', 0):.2f})")
        else:
            print("HOLD (no data)")
    except Exception as e:
        print(f"Error: {str(e)[:30]}")

print("\n" + "=" * 70)
print(f"SCAN COMPLETE - Found {len(signals)} actionable signals")
print("=" * 70)

if signals:
    print("\nActionable Signals:")
    print("-" * 70)
    for sig in sorted(signals, key=lambda x: x.get('confidence', 0), reverse=True):
        symbol = sig.get('symbol')
        action = sig.get('action')
        conf = sig.get('confidence', 0) * 100
        model = sig.get('model', 'Unknown')[:25]
        stop = sig.get('stop_loss_pct', 0.02) * 100
        target = sig.get('take_profit_pct', 0.06) * 100

        print(f"  {action:4} {symbol:6} | {conf:5.1f}% | {model:25} | SL:{stop:.1f}% TP:{target:.1f}%")
else:
    print("\nNo signals - All models say HOLD")
    print("This means the AI is being selective and waiting for better entries.")
