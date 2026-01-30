# CLAUDE.md - 30Bots Trading Application

## Quick Commands

```bash
# Navigate to project
cd /mnt/d/Projects/30Bots

# Run the desktop application
python desktop/main.py

# Run with virtual environment (recommended)
cd /mnt/d/Projects/30Bots/desktop
source venv/bin/activate  # Linux/WSL
venv\Scripts\activate     # Windows
python main.py

# Install dependencies
pip install -r desktop/requirements.txt

# Run analysis scripts
python scripts/check_positions.py
python scripts/check_status.py
python scripts/analyze_trades.py
```

## Project Overview

30Bots is a PyQt6 desktop trading terminal with AI-powered predictions using a 30-model ensemble system. It connects to Alpaca Markets for live/paper trading and uses Google Gemini for AI assistant and news sentiment analysis.

### Key Features
- **30-Model AI Ensemble**: XGBoost, LightGBM models with priority-based prediction system
- **Dual-Mode Trading**: Day trading (5-min candles) + Swing trading (daily candles)
- **Dual Account Support**: 50K account (day trading) + 100K account (swing trading)
- **Real-time Position Monitoring**: Stop-loss, take-profit, trailing stops
- **News Sentiment**: Gemini-powered analysis of Alpaca news feed
- **Crypto Tab**: Portfolio tracking and volatility scanner

## Architecture

```
30Bots/
├── desktop/                    # Main PyQt6 application
│   ├── main.py                 # Entry point
│   ├── config.py               # Configuration and API keys
│   ├── ai/                     # AI components
│   │   ├── gemini_assistant.py # Chat assistant
│   │   └── prediction_worker.py# Background predictions
│   ├── core/                   # Trading logic
│   │   ├── auto_trader.py      # Automated trading
│   │   ├── position_monitor.py # SL/TP monitoring
│   │   └── trade_logger.py     # SQLite trade database
│   ├── data/                   # Data layer
│   │   ├── broker_adapter.py   # Alpaca API wrapper
│   │   ├── news_fetcher.py     # News data
│   │   ├── news_sentiment.py   # Sentiment analysis
│   │   └── websocket_worker.py # Real-time streaming
│   ├── ui/                     # User interface
│   │   ├── main_window.py      # Main window
│   │   ├── styles.py           # Dark theme CSS
│   │   ├── tabs/               # Tab widgets
│   │   │   ├── dashboard_tab.py
│   │   │   ├── ai_trading_tab.py
│   │   │   ├── news_trading_tab.py
│   │   │   ├── crypto_tab.py
│   │   │   ├── analytics_tab.py
│   │   │   └── settings_tab.py
│   │   └── widgets/            # Reusable widgets
│   └── trades.db               # SQLite trade history
├── models/                     # ML models
│   ├── ai_30model_brain.py     # 30-model ensemble predictor
│   ├── multi_strategy_predictor.py
│   └── multi_strategy_30_system.pkl  # Trained model weights
├── scripts/                    # Utility scripts
│   ├── check_positions.py
│   ├── check_status.py
│   ├── analyze_trades.py
│   └── ...
└── legacy/                     # Older code versions
```

## Configuration

### Environment Variables (.env file)
```env
# Alpaca API (required)
ALPACA_API_KEY=your_key
ALPACA_SECRET_KEY=your_secret
ALPACA_PAPER=true

# Dual Account Setup (optional)
ALPACA_50K_API_KEY=day_trading_key
ALPACA_50K_SECRET_KEY=day_trading_secret
ALPACA_100K_API_KEY=swing_trading_key
ALPACA_100K_SECRET_KEY=swing_trading_secret

# Gemini API (for AI features)
GEMINI_API_KEY=your_gemini_key
```

The app searches for `.env` in these locations:
1. `desktop/.env`
2. `30Bots/.env`
3. `/mnt/d/Projects/Stock-Trading-AI/.env`

### Trading Parameters (config.py)
- `MAX_POSITIONS`: 6
- `POSITION_SIZE_PCT`: 10% per position
- `STOP_LOSS_PCT`: 2% default
- `PROFIT_TARGET_PCT`: 6% default
- `TRAILING_STOP_ACTIVATION`: 1.5% gain
- `TRAILING_STOP_PCT`: 1% trail
- `ALLOW_SHORTING`: True

## AI Model System

The 30-model brain uses a **priority-based prediction system**:
1. Models are ranked by win rate
2. Highest win rate model gets first prediction attempt
3. Falls back to next model if confidence is low

### Model Strategies (10 strategies x 3 modes each = 30 models)
- **Momentum**: Selective (94.8% WR), Moderate (90.9% WR), Aggressive (86.8% WR)
- **Mean Reversion**: Quick reversals to mean
- **Breakout**: Volatility breakout detection
- **Trend Following**: Extended trend capture
- **VWAP**: Volume-weighted price analysis
- And 5 more strategy types...

Each model has optimized risk parameters:
- Stop loss: 1.5% - 3.5% depending on strategy
- Take profit: 4% - 8.5% (minimum 2:1 R:R)
- Trailing stops with activation thresholds

## UI Components

### Tabs
1. **Dashboard**: Portfolio overview, positions, P&L, activity log
2. **AI Trading**: Predictions, model performance, watchlists, auto-trading controls
3. **News Trading**: Alpaca news feed with Gemini sentiment analysis
4. **Crypto**: Cryptocurrency portfolio and volatility scanner
5. **Analytics**: Equity curves, trade calendar heatmap, time analysis
6. **Settings**: API config, risk parameters, notifications

### Keyboard Shortcuts
- `Ctrl+Shift+A`: Toggle AI Assistant panel
- `Ctrl+Q`: Quit application

## Database

SQLite database at `desktop/trades.db` stores:
- Trade history with entry/exit prices
- Position tracking
- Model performance metrics

## Dependencies

Key packages (see `desktop/requirements.txt`):
- PyQt6 >= 6.5.0 (GUI)
- pyqtgraph >= 0.13.0 (Charts)
- pandas >= 2.0.0, numpy >= 1.24.0 (Data)
- xgboost >= 1.7.0, scikit-learn >= 1.2.0 (ML)
- websocket-client >= 1.5.0 (Real-time data)

## API Rate Limits

Gemini API rate limiting is built-in:
- 12 requests/minute (under 15 RPM free tier)
- Exponential backoff on 429 errors
- Up to 3 retry attempts

## Related Projects

- **Stock-Trading-AI**: `/mnt/d/Projects/Stock-Trading-AI` - Main training infrastructure
- **HybridTrader**: `/mnt/d/Projects/HybridTrader` - Alternative trading system
