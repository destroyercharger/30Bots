"""
30Bots Desktop Trading Application - Configuration
"""

import os
from pathlib import Path

# =============================================================================
# APPLICATION INFO
# =============================================================================
APP_NAME = "30Bots Trading Terminal"
APP_VERSION = "1.0.0"

# =============================================================================
# PATHS
# =============================================================================
BASE_DIR = Path(__file__).parent
DATA_DIR = BASE_DIR / "data"
DB_PATH = BASE_DIR / "trades.db"

# =============================================================================
# API KEYS
# =============================================================================
# Option 1: Set your keys directly here (uncomment and fill in):
# ALPACA_API_KEY = "your-api-key-here"
# ALPACA_SECRET_KEY = "your-secret-key-here"

# Option 2: Load from .env file
def load_env_file():
    """Try to load API keys from .env files"""
    env_paths = [
        BASE_DIR / ".env",                                    # desktop/.env
        BASE_DIR.parent / ".env",                             # 30Bots/.env
        Path("/mnt/d/Projects/Stock-Trading-AI/.env"),        # Stock-Trading-AI/.env
        Path("D:/Projects/Stock-Trading-AI/.env"),            # Windows path
    ]

    for env_path in env_paths:
        if env_path.exists():
            try:
                with open(env_path) as f:
                    for line in f:
                        line = line.strip()
                        if line and not line.startswith('#') and '=' in line:
                            key, value = line.split('=', 1)
                            key = key.strip()
                            value = value.strip().strip('"').strip("'")
                            if key and value and key not in os.environ:
                                os.environ[key] = value
                print(f"Loaded env from: {env_path}")
                return True
            except Exception as e:
                print(f"Error loading {env_path}: {e}")
    return False

# Try to load .env file
load_env_file()

# Load from environment (or direct setting above)
ALPACA_API_KEY = os.environ.get("ALPACA_API_KEY", "")
ALPACA_SECRET_KEY = os.environ.get("ALPACA_SECRET_KEY", "")
ALPACA_PAPER = os.environ.get("ALPACA_PAPER", "true").lower() == "true"

# Gemini API for news sentiment analysis
GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")

# Print status
if ALPACA_API_KEY and ALPACA_SECRET_KEY:
    print(f"Alpaca API keys loaded (key ends with: ...{ALPACA_API_KEY[-4:]})")
else:
    print("WARNING: Alpaca API keys not found. Set them in config.py or .env file")

# =============================================================================
# TRADING CONFIGURATION
# =============================================================================
STARTING_CAPITAL = 100000.0
MAX_POSITIONS = 6
POSITION_SIZE_PCT = 0.10  # 10% per position

# Risk Management
STOP_LOSS_PCT = 0.02          # 2% default stop loss
PROFIT_TARGET_PCT = 0.06      # 6% default profit target
TRAILING_STOP_ACTIVATION = 0.015  # Activate at 1.5% gain
TRAILING_STOP_PCT = 0.01      # Trail by 1%
MAX_STOP_PCT = 0.05           # Maximum stop loss (emergency)

# After Hours
AFTER_HOURS_STOP_LOSS_PCT = 0.015
AFTER_HOURS_PROFIT_TARGET_PCT = 0.04

# =============================================================================
# MARKET HOURS (Eastern Time)
# =============================================================================
MARKET_OPEN_HOUR = 9
MARKET_OPEN_MINUTE = 30
MARKET_CLOSE_HOUR = 16
MARKET_CLOSE_MINUTE = 0

PREMARKET_START_HOUR = 4
AFTERHOURS_END_HOUR = 20

# =============================================================================
# UI CONFIGURATION
# =============================================================================
WINDOW_MIN_WIDTH = 1400
WINDOW_MIN_HEIGHT = 900

# Update intervals (milliseconds)
PRICE_UPDATE_INTERVAL = 1000      # 1 second
POSITION_UPDATE_INTERVAL = 5000   # 5 seconds
SCANNER_INTERVAL = 60000          # 1 minute

# =============================================================================
# CHART CONFIGURATION
# =============================================================================
DEFAULT_CHART_INTERVAL = "5m"
CHART_INTERVALS = ["1m", "5m", "15m", "30m", "1h", "4h", "1d", "1wk"]

# Default indicators to show
DEFAULT_INDICATORS = ["sma_20", "sma_50", "volume"]

# =============================================================================
# LOGGING
# =============================================================================
LOG_LEVEL = "INFO"
MAX_ACTIVITY_LOG_ENTRIES = 500
MAX_DECISION_LOG_ENTRIES = 100
