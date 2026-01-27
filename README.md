# 30Bots Desktop Trading Application

A professional PyQt6 desktop trading application with AI-powered predictions, featuring a Bloomberg/TradeStation-inspired dark theme.

![Python](https://img.shields.io/badge/Python-3.10+-blue.svg)
![PyQt6](https://img.shields.io/badge/PyQt6-6.0+-green.svg)
![License](https://img.shields.io/badge/License-MIT-yellow.svg)

## Features

### AI-Powered Trading
- **30-Model Ensemble** - Combines predictions from 30 trained ML models (XGBoost, LightGBM, etc.)
- **Dual-Mode Trading** - Simultaneous day trading (5-min candles) and swing trading (daily candles)
- **Automatic Position Management** - Stop-loss and take-profit monitoring with real-time execution

### Dashboard
- Real-time portfolio overview with P&L tracking
- Interactive stock charts with multiple timeframes
- Live position monitoring with entry/exit indicators
- Activity log for trade events

### AI Trading Tab
- Real-time AI predictions with confidence scores
- Model performance comparison
- Watchlist management for day and swing trading
- Auto-trading controls (Start/Stop/Pause)

### News Trading
- Alpaca news feed integration
- Gemini-powered sentiment analysis
- Trading signals based on news sentiment
- Real-time news alerts

### Crypto Tab
- Cryptocurrency portfolio tracking
- Volatility scanner
- Market overview (Fear & Greed index, BTC dominance)
- 24-hour price changes

### Analytics
- Equity curve visualization
- Model performance comparison charts
- Trade calendar heatmap
- Time-based analysis (best/worst trading hours)
- Symbol performance breakdown

### Settings
- Trading configuration (position size, stop-loss %, etc.)
- API connection management
- Notification preferences
- Theme and appearance settings
- Persistent settings via QSettings

### AI Assistant (Ctrl+Shift+A)
- Gemini-powered chat interface
- Trading analysis and insights
- Error debugging assistance
- Quick actions for common queries

## Installation

### Prerequisites
- Python 3.10+
- Alpaca Trading Account (Paper or Live)
- Google Gemini API Key (for AI features)

### Setup

1. Clone the repository:
```bash
git clone https://github.com/destroyercharger/30Bots.git
cd 30Bots
```

2. Create virtual environment:
```bash
python -m venv venv
source venv/bin/activate  # On Windows: venv\Scripts\activate
```

3. Install dependencies:
```bash
pip install -r desktop/requirements.txt
```

4. Configure environment variables:
Create a `.env` file in the project root (or parent Stock-Trading-AI folder):
```env
ALPACA_API_KEY=your_alpaca_api_key
ALPACA_SECRET_KEY=your_alpaca_secret_key
ALPACA_BASE_URL=https://paper-api.alpaca.markets  # or live URL
GEMINI_API_KEY=your_gemini_api_key
```

5. Run the application:
```bash
python desktop/main.py
```

## Project Structure

```
30Bots/
├── desktop/
│   ├── ai/
│   │   ├── gemini_assistant.py    # AI chat assistant
│   │   └── prediction_worker.py   # Background AI predictions
│   ├── core/
│   │   ├── auto_trader.py         # Automated trading logic
│   │   ├── position_monitor.py    # SL/TP monitoring
│   │   └── trade_logger.py        # Trade history database
│   ├── data/
│   │   ├── broker_adapter.py      # Alpaca API wrapper
│   │   ├── news_fetcher.py        # News data fetching
│   │   ├── news_sentiment.py      # Gemini sentiment analysis
│   │   └── websocket_worker.py    # Real-time data streaming
│   ├── ui/
│   │   ├── tabs/
│   │   │   ├── dashboard_tab.py   # Main dashboard
│   │   │   ├── ai_trading_tab.py  # AI trading interface
│   │   │   ├── news_trading_tab.py
│   │   │   ├── crypto_tab.py
│   │   │   ├── analytics_tab.py
│   │   │   └── settings_tab.py
│   │   ├── widgets/
│   │   │   ├── ai_terminal.py     # AI chat widget
│   │   │   ├── stock_chart.py     # Interactive charts
│   │   │   ├── positions_table.py
│   │   │   └── ...
│   │   ├── main_window.py         # Main application window
│   │   └── styles.py              # Dark theme stylesheet
│   ├── config.py                  # Configuration loader
│   └── main.py                    # Application entry point
├── ai_30model_brain.py            # 30-model ensemble predictor
├── multi_strategy_predictor.py    # Strategy prediction logic
└── README.md
```

## Usage

### Starting the Application
```bash
python desktop/main.py
```

The application will:
1. Connect to Alpaca API
2. Load the 30-model AI brain
3. Initialize dual-mode traders (Day + Swing)
4. Start position monitoring
5. Auto-start trading during market hours

### Keyboard Shortcuts
- `Ctrl+Shift+A` - Toggle AI Assistant panel
- `Ctrl+Q` - Quit application

### Trading Modes
- **Day Trading**: 5-minute candles, 2% stop-loss, 4% take-profit
- **Swing Trading**: Daily candles, 5% stop-loss, 12% take-profit

## API Rate Limiting

The application includes built-in rate limiting for the Gemini API:
- 12 requests per minute (stays under 15 RPM free tier limit)
- Exponential backoff on 429 errors
- Automatic retry with up to 3 attempts

## Contributing

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

## License

This project is licensed under the MIT License - see the LICENSE file for details.

## Disclaimer

This software is for educational and research purposes only. Trading stocks and cryptocurrencies involves substantial risk of loss. Past performance is not indicative of future results. Always do your own research and consider consulting with a financial advisor before making investment decisions.

## Acknowledgments

- [Alpaca Markets](https://alpaca.markets/) - Trading API
- [Google Gemini](https://ai.google.dev/) - AI Assistant
- [PyQt6](https://www.riverbankcomputing.com/software/pyqt/) - GUI Framework
