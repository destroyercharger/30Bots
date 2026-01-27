"""Test that model-specific risk params are being used"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

from ai.prediction_worker import get_ai_brain
from data.broker_adapter import AlpacaBroker
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
import pandas as pd

print("="*60)
print("VERIFYING MODEL-SPECIFIC RISK PARAMETERS")
print("="*60)

brain = get_ai_brain()
broker = AlpacaBroker(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)

# Test symbols that have signals
test_symbols = ['RIVN', 'COIN', 'NFLX', 'GOOGL', 'AMD', 'META']

for symbol in test_symbols:
    bars = broker.get_bars(symbol, '5Min', limit=100)
    if not bars:
        continue

    df = pd.DataFrame(bars)
    # Use lower confidence to see more signals
    prediction = brain.get_prediction(df, min_confidence=0.3)

    if prediction and prediction.get('action') != 'HOLD':
        model = prediction.get('model', 'Unknown')
        print(f"\n{symbol} - {prediction.get('action')}")
        print(f"  Model: {model}")
        print(f"  Confidence: {prediction.get('confidence', 0)*100:.1f}%")
        print(f"  Model Win Rate: {prediction.get('win_rate', 0)*100:.1f}%")
        print(f"  ----")
        print(f"  Stop Loss: {prediction.get('stop_loss_pct', 0)*100:.2f}%")
        print(f"  Take Profit: {prediction.get('take_profit_pct', 0)*100:.2f}%")
        print(f"  Trailing Activation: {prediction.get('trailing_activation_pct', 0)*100:.2f}%")
        print(f"  Trailing Stop: {prediction.get('trailing_stop_pct', 0)*100:.2f}%")
        print(f"  R:R Ratio: {prediction.get('risk_reward_ratio', 0):.1f}:1")
    else:
        action = prediction.get('action', 'NONE') if prediction else 'NONE'
        print(f"\n{symbol}: {action}")

print("\n" + "="*60)
print("Expected params from MODEL_RISK_PARAMS:")
print("  MeanReversion_Selective: SL=1.5%, TP=4%, Trail=0.8%")
print("  MeanReversion_Moderate:  SL=2.0%, TP=5%, Trail=1.0%")
print("  Momentum_Moderate:       SL=2.5%, TP=6.5%, Trail=1.5%")
print("  Momentum_Aggressive:     SL=3.5%, TP=8.5%, Trail=2.0%")
print("="*60)
