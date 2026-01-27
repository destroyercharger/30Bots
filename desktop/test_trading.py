#!/usr/bin/env python3
"""
Test trading system without GUI
"""

import sys
import os
sys.path.insert(0, os.path.dirname(__file__))

from datetime import datetime

# Load config
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY

print("=" * 60)
print("30Bots Trading System Test")
print("=" * 60)

# Test broker connection
print("\n[1] Testing Alpaca Connection...")
from data.broker_adapter import AlpacaBroker

if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    broker = AlpacaBroker(
        api_key=ALPACA_API_KEY,
        secret_key=ALPACA_SECRET_KEY,
        paper=True
    )

    account = broker.get_account()
    print(f"    ✓ Connected to Alpaca")
    print(f"    Account Equity: ${float(account.equity):,.2f}")
    print(f"    Cash: ${float(account.cash):,.2f}")
    print(f"    Buying Power: ${float(account.buying_power):,.2f}")

    # Get positions
    positions = broker.get_positions()
    print(f"    Positions: {len(positions)}")
    for pos in positions[:5]:
        pnl = (float(pos.current_price) - float(pos.avg_entry_price)) * float(pos.qty)
        print(f"      - {pos.symbol}: {int(float(pos.qty))} shares, P&L: ${pnl:+.2f}")
else:
    print("    ✗ No API keys found")
    broker = None

# Test AI Brain
print("\n[2] Testing AI Brain...")
from ai.prediction_worker import get_ai_brain, AI_AVAILABLE

if AI_AVAILABLE:
    brain = get_ai_brain()
    if brain and brain.loaded:
        print(f"    ✓ AI Brain loaded")
        print(f"    Models: {len(brain.models)}")
        print(f"    Top 5 models by win rate:")
        for name in brain.model_priority[:5]:
            model = brain.models.get(name, {})
            wr = model.get('win_rate', 0) * 100
            print(f"      - {name}: {wr:.1f}%")
    else:
        print("    ✗ AI Brain not loaded")
else:
    print("    ✗ AI module not available")

# Test Trade Logger
print("\n[3] Testing Trade Logger...")
from core.trade_logger import get_trade_logger

logger = get_trade_logger()
today_stats = logger.get_today_stats()
overall_stats = logger.get_overall_stats()

print(f"    ✓ Trade logger initialized")
print(f"    Today: {today_stats['total_trades']} trades, ${today_stats['total_pnl']:+.2f} P&L")
print(f"    Overall: {overall_stats['total_trades']} trades, {overall_stats['win_rate']:.1f}% win rate")

# Test AI Prediction
print("\n[4] Testing AI Prediction on AAPL...")
if broker and AI_AVAILABLE:
    from ai.prediction_worker import AIPredictionWorker

    worker = AIPredictionWorker(broker=broker)
    prediction = worker.get_prediction_for_symbol("AAPL")

    if prediction:
        print(f"    ✓ Prediction received")
        print(f"    Symbol: {prediction.get('symbol')}")
        print(f"    Action: {prediction.get('action')}")
        print(f"    Confidence: {prediction.get('confidence', 0)*100:.1f}%")
        print(f"    Model: {prediction.get('model')}")
        print(f"    Stop Loss: {prediction.get('stop_loss_pct', 0)*100:.1f}%")
        print(f"    Take Profit: {prediction.get('take_profit_pct', 0)*100:.1f}%")
    else:
        print("    - No prediction (likely HOLD)")
else:
    print("    ✗ Skipped (no broker or AI)")

# Test Auto Trader config
print("\n[5] Auto Trader Configuration...")
from core.auto_trader import AutoTrader, DEFAULT_DAY_WATCHLIST

trader = AutoTrader(broker=broker)
print(f"    Mode: {trader.config['mode'].upper()}")
print(f"    Max Positions: {trader.config['max_positions']}")
print(f"    Min Confidence: {trader.config['min_confidence']*100:.0f}%")
print(f"    Day Trading Settings:")
print(f"      Stop Loss: {trader.day_settings['stop_loss_pct']*100:.1f}%")
print(f"      Take Profit: {trader.day_settings['take_profit_pct']*100:.1f}%")
print(f"      Trailing Activation: {trader.day_settings['trailing_activation_pct']*100:.1f}%")
print(f"    Watchlist: {', '.join(DEFAULT_DAY_WATCHLIST[:8])}...")

# Scan multiple symbols
print("\n[6] Scanning Watchlist for Signals...")
if broker and AI_AVAILABLE:
    worker = AIPredictionWorker(broker=broker)
    signals = []

    for symbol in DEFAULT_DAY_WATCHLIST[:6]:
        print(f"    Scanning {symbol}...", end=" ")
        pred = worker.get_prediction_for_symbol(symbol)
        if pred and pred.get('action') in ['BUY', 'SELL']:
            action = pred.get('action')
            conf = pred.get('confidence', 0) * 100
            model = pred.get('model', 'Unknown')
            print(f"{action} ({conf:.1f}% - {model})")
            signals.append(pred)
        else:
            print("HOLD")

    print(f"\n    Found {len(signals)} actionable signals")
else:
    print("    ✗ Skipped")

print("\n" + "=" * 60)
print("Test Complete!")
print("=" * 60)
print("\nTo run the GUI app, use Windows Python:")
print("  cd D:\\Projects\\30Bots\\desktop")
print("  python main.py")
