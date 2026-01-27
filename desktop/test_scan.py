"""Test AI scanning directly"""
import sys
sys.path.insert(0, '.')
sys.path.insert(0, '..')

from data.broker_adapter import AlpacaBroker
from config import ALPACA_API_KEY, ALPACA_SECRET_KEY
from ai.prediction_worker import AIPredictionWorker, get_ai_brain

print("="*50)
print("AI SCAN TEST")
print("="*50)

# Setup broker
broker = AlpacaBroker(ALPACA_API_KEY, ALPACA_SECRET_KEY, paper=True)
print(f"Broker connected: ${float(broker.get_account().equity):,.2f}")

# Get AI brain
brain = get_ai_brain()
if brain and brain.loaded:
    print(f"AI Brain loaded: {len(brain.models)} models")
else:
    print("AI Brain NOT loaded!")
    sys.exit(1)

# Create AI worker
worker = AIPredictionWorker()
worker.set_broker(broker)
print(f"AI Worker ready: {worker.ai_brain.loaded if worker.ai_brain else 'No brain'}")

# Test scan a few symbols
test_symbols = ['AAPL', 'NVDA', 'TSLA', 'MSFT', 'AMD']
print(f"\nScanning {len(test_symbols)} symbols...")

for symbol in test_symbols:
    print(f"\n--- {symbol} ---")
    try:
        prediction = worker.get_prediction_for_symbol(symbol)
        if prediction:
            action = prediction.get('action', 'NONE')
            confidence = prediction.get('confidence', 0)
            model = prediction.get('model', 'Unknown')
            print(f"  Action: {action}")
            print(f"  Confidence: {confidence*100:.1f}%")
            print(f"  Model: {model}")
            if action == 'BUY' and confidence >= 0.70:
                print(f"  ** ACTIONABLE BUY SIGNAL! **")
        else:
            print(f"  No prediction returned")
    except Exception as e:
        print(f"  Error: {e}")

print("\n" + "="*50)
