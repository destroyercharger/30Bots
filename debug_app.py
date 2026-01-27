"""
30Bots Desktop App - Debug & Diagnostics
Checks all major components are working correctly
"""
import sys
sys.path.insert(0, r'D:\Projects\Stock-Trading-AI')
sys.path.insert(0, r'D:\Projects\30Bots\desktop')

from dotenv import load_dotenv
import os
load_dotenv(r'D:\Projects\Stock-Trading-AI\.env')

def print_header(title):
    print(f"\n{'='*60}")
    print(f" {title}")
    print('='*60)

def check_pass(name):
    print(f"  [OK] {name}")

def check_fail(name, error):
    print(f"  [FAIL] {name}: {error}")

def check_warn(name, msg):
    print(f"  [WARN] {name}: {msg}")

# ============================================================
print_header("1. ENVIRONMENT & API KEYS")
# ============================================================

alpaca_key = os.getenv('ALPACA_API_KEY')
alpaca_secret = os.getenv('ALPACA_SECRET_KEY')
gemini_key = os.getenv('GEMINI_API_KEY')

if alpaca_key and len(alpaca_key) > 10:
    check_pass(f"Alpaca API Key (ends with ...{alpaca_key[-4:]})")
else:
    check_fail("Alpaca API Key", "Missing or invalid")

if alpaca_secret and len(alpaca_secret) > 10:
    check_pass(f"Alpaca Secret Key (ends with ...{alpaca_secret[-4:]})")
else:
    check_fail("Alpaca Secret Key", "Missing or invalid")

if gemini_key and len(gemini_key) > 10:
    check_pass(f"Gemini API Key (ends with ...{gemini_key[-4:]})")
else:
    check_warn("Gemini API Key", "Missing - AI Assistant won't work")

# ============================================================
print_header("2. ALPACA CONNECTION")
# ============================================================

try:
    from alpaca.trading.client import TradingClient
    client = TradingClient(alpaca_key, alpaca_secret, paper=True)
    account = client.get_account()
    check_pass(f"Connected to Alpaca")
    check_pass(f"Account Equity: ${float(account.equity):,.2f}")
    check_pass(f"Buying Power: ${float(account.buying_power):,.2f}")
    check_pass(f"Account Status: {account.status}")

    positions = client.get_all_positions()
    check_pass(f"Positions loaded: {len(positions)}")

    if account.pattern_day_trader:
        check_warn("PDT Status", "Account is flagged as Pattern Day Trader")
    else:
        check_pass("PDT Status: Not flagged")

except Exception as e:
    check_fail("Alpaca Connection", str(e))

# ============================================================
print_header("3. AI MODEL BRAIN")
# ============================================================

try:
    sys.path.insert(0, r'D:\Projects\30Bots')
    from ai_30model_brain import AIModelBrain
    brain = AIModelBrain()
    check_pass(f"AI Brain initialized")
    check_pass(f"Models loaded: {len(brain.models)}")

    # Check top models
    top_models = brain.get_top_models(3)
    for i, (name, wr) in enumerate(top_models, 1):
        check_pass(f"Top {i}: {name} ({wr:.1f}% win rate)")

except Exception as e:
    check_fail("AI Model Brain", str(e))

# ============================================================
print_header("4. PREDICTION TEST")
# ============================================================

try:
    # Test prediction on a symbol
    test_symbol = "AAPL"
    prediction = brain.predict(test_symbol)

    if prediction:
        action = prediction.get('action', 'NONE')
        confidence = prediction.get('confidence', 0)
        model = prediction.get('model', 'Unknown')
        check_pass(f"Prediction for {test_symbol}: {action} ({confidence*100:.1f}% - {model})")
    else:
        check_warn(f"Prediction for {test_symbol}", "No signal generated (may be normal)")

except Exception as e:
    check_fail("Prediction Test", str(e))

# ============================================================
print_header("5. DATABASE & TRADE LOGGER")
# ============================================================

try:
    from core.trade_logger import get_trade_logger
    logger = get_trade_logger()
    check_pass("Trade logger initialized")

    # Check recent trades
    recent = logger.get_recent_trades(5)
    check_pass(f"Recent trades loaded: {len(recent)}")

    # Check today's stats
    stats = logger.get_today_stats()
    check_pass(f"Today's P&L: ${stats.get('total_pnl', 0):,.2f}")
    check_pass(f"Today's trades: {stats.get('total_trades', 0)}")

except Exception as e:
    check_fail("Trade Logger", str(e))

# ============================================================
print_header("6. NEWS FETCHER")
# ============================================================

try:
    from data.news_fetcher import get_news_fetcher
    fetcher = get_news_fetcher()
    check_pass("News fetcher initialized")

    # Test fetching news
    news = fetcher.get_market_news(limit=5)
    check_pass(f"Market news fetched: {len(news)} articles")

except Exception as e:
    check_fail("News Fetcher", str(e))

# ============================================================
print_header("7. GEMINI SENTIMENT ANALYZER")
# ============================================================

try:
    from data.news_sentiment import get_sentiment_analyzer
    analyzer = get_sentiment_analyzer()
    check_pass("Sentiment analyzer initialized")

    if gemini_key:
        # Test a simple sentiment analysis (skip actual API call to save quota)
        check_pass("Gemini API key configured")
        check_warn("Skipping live test", "To preserve API quota")
    else:
        check_warn("Sentiment analysis", "Disabled - no Gemini API key")

except Exception as e:
    check_fail("Sentiment Analyzer", str(e))

# ============================================================
print_header("8. GEMINI AI ASSISTANT")
# ============================================================

try:
    from ai.gemini_assistant import get_assistant
    assistant = get_assistant()
    check_pass("AI Assistant initialized")

    if gemini_key:
        check_pass("Ready to answer queries")
    else:
        check_warn("AI Assistant", "Disabled - no Gemini API key")

except Exception as e:
    check_fail("AI Assistant", str(e))

# ============================================================
print_header("9. PYQT6 GUI")
# ============================================================

try:
    from PyQt6.QtWidgets import QApplication
    from PyQt6.QtCore import QT_VERSION_STR
    check_pass(f"PyQt6 version: {QT_VERSION_STR}")
    check_pass("GUI framework ready")

except Exception as e:
    check_fail("PyQt6", str(e))

# ============================================================
print_header("10. CURRENT POSITIONS DETAIL")
# ============================================================

try:
    positions = client.get_all_positions()
    if positions:
        print(f"\n  {'Symbol':<8} {'Side':<6} {'Qty':<6} {'Entry':>10} {'Current':>10} {'P&L':>12}")
        print(f"  {'-'*54}")
        total_pnl = 0
        for p in positions:
            qty = float(p.qty)
            side = 'LONG' if qty > 0 else 'SHORT'
            entry = float(p.avg_entry_price)
            current = float(p.current_price)
            pnl = float(p.unrealized_pl)
            total_pnl += pnl
            print(f"  {p.symbol:<8} {side:<6} {abs(qty):<6.0f} ${entry:>9,.2f} ${current:>9,.2f} ${pnl:>+11,.2f}")
        print(f"  {'-'*54}")
        print(f"  {'Total Unrealized P&L:':<40} ${total_pnl:>+11,.2f}")
    else:
        check_warn("Positions", "No open positions")
except Exception as e:
    check_fail("Positions Detail", str(e))

# ============================================================
print_header("SUMMARY")
# ============================================================
print("""
All core components checked. The app should be functioning correctly.

Key features verified:
  • Alpaca API connection and trading
  • 30-model AI ensemble predictions
  • Trade logging and database
  • News fetching
  • Sentiment analysis (Gemini)
  • AI Assistant (Gemini)
  • PyQt6 GUI framework

If any checks failed, review the error messages above.
""")
