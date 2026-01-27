"""
AI Prediction Worker
Background thread for running AI model predictions
"""

import sys
from pathlib import Path
from typing import Optional, Dict, List
from datetime import datetime, timedelta
import threading

import numpy as np
import pandas as pd

from PyQt6.QtCore import QThread, pyqtSignal, QTimer, QMutex

# Add parent directory for imports
PARENT_DIR = Path(__file__).parent.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Try to import the AI brain
try:
    from ai_30model_brain import AIModelBrain, get_model_risk_params, MODEL_RISK_PARAMS
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    AIModelBrain = None

# Global AI brain instance
_ai_brain = None
_ai_lock = threading.Lock()


def get_ai_brain() -> Optional['AIModelBrain']:
    """Get or create the global AI brain instance."""
    global _ai_brain

    if not AI_AVAILABLE:
        return None

    with _ai_lock:
        if _ai_brain is None:
            _ai_brain = AIModelBrain()
        return _ai_brain


class AIPredictionWorker(QThread):
    """
    Worker thread for running AI predictions.

    Fetches data and runs the 30-model brain to generate trading signals.
    """

    # Signals
    prediction_ready = pyqtSignal(dict)  # Single prediction result
    all_predictions_ready = pyqtSignal(list)  # All 30 model predictions
    model_status = pyqtSignal(str, bool)  # Model name, loaded status
    error = pyqtSignal(str)

    def __init__(self, broker=None, parent=None, trading_mode='day'):
        super().__init__(parent)
        self.broker = broker
        self.ai_brain = None
        self.running = False
        self.symbols_to_scan: List[str] = []
        self.scan_interval = 60000  # 1 minute default
        self.mutex = QMutex()

        # Trading mode configuration
        self.trading_mode = trading_mode  # 'day' or 'swing'
        self.timeframe_config = {
            'day': {
                'timeframe': '5Min',
                'limit': 100,  # ~8 hours of 5-min data
                'min_bars': 50,
                'description': 'Day Trading (5-minute candles)'
            },
            'swing': {
                'timeframe': '1Day',
                'limit': 100,  # 100 days
                'min_bars': 50,
                'description': 'Swing Trading (daily candles)'
            }
        }

        # Initialize AI brain
        self._init_brain()

    def _init_brain(self):
        """Initialize the AI brain."""
        if AI_AVAILABLE:
            self.ai_brain = get_ai_brain()
            if self.ai_brain and self.ai_brain.loaded:
                config = self.timeframe_config[self.trading_mode]
                print(f"[AI Worker] Brain loaded with {len(self.ai_brain.models)} models")
                print(f"[AI Worker] Mode: {config['description']}")
            else:
                print("[AI Worker] Brain failed to load models")
        else:
            print("[AI Worker] AI brain not available")

    def set_trading_mode(self, mode: str):
        """Set trading mode ('day' or 'swing')."""
        if mode in self.timeframe_config:
            self.trading_mode = mode
            config = self.timeframe_config[mode]
            print(f"[AI Worker] Switched to {config['description']}")

    def set_broker(self, broker):
        """Set the broker for data fetching."""
        self.broker = broker

    def set_symbols(self, symbols: List[str]):
        """Set symbols to scan."""
        self.mutex.lock()
        self.symbols_to_scan = list(symbols)
        self.mutex.unlock()

    def add_symbol(self, symbol: str):
        """Add a symbol to scan."""
        self.mutex.lock()
        if symbol not in self.symbols_to_scan:
            self.symbols_to_scan.append(symbol)
        self.mutex.unlock()

    def get_prediction_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get AI prediction for a single symbol."""
        if not self.ai_brain or not self.ai_brain.loaded:
            return None

        if not self.broker:
            return None

        try:
            # Get timeframe config based on trading mode
            config = self.timeframe_config[self.trading_mode]

            # Fetch historical data from Alpaca
            bars = self.broker.get_bars(
                symbol=symbol,
                timeframe=config['timeframe'],
                limit=config['limit']
            )

            if not bars or len(bars) < config['min_bars']:
                return None

            # Convert to DataFrame
            df = pd.DataFrame(bars)

            # Rename columns to match expected format
            column_map = {
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap',
                'n': 'trades'
            }
            df = df.rename(columns=column_map)

            # Get prediction from AI brain
            prediction = self.ai_brain.get_prediction(df, min_confidence=0.6)

            if prediction:
                prediction['symbol'] = symbol
                prediction['timestamp'] = datetime.now().isoformat()
                prediction['trading_mode'] = self.trading_mode
                prediction['timeframe'] = config['timeframe']

            return prediction

        except Exception as e:
            self.error.emit(f"Prediction error for {symbol}: {e}")
            return None

    def get_all_model_predictions(self, symbol: str) -> List[Dict]:
        """Get predictions from all 30 models for a symbol."""
        if not self.ai_brain or not self.ai_brain.loaded:
            return []

        if not self.broker:
            return []

        try:
            # Get timeframe config based on trading mode
            config = self.timeframe_config[self.trading_mode]

            # Fetch historical data
            bars = self.broker.get_bars(
                symbol=symbol,
                timeframe=config['timeframe'],
                limit=config['limit']
            )

            if not bars or len(bars) < config['min_bars']:
                return []

            # Convert to DataFrame
            df = pd.DataFrame(bars)
            column_map = {
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap'
            }
            df = df.rename(columns=column_map)

            # Get predictions from all models
            predictions = self.ai_brain.get_all_predictions(df)

            for pred in predictions:
                pred['symbol'] = symbol

            return predictions

        except Exception as e:
            self.error.emit(f"All predictions error for {symbol}: {e}")
            return []

    def get_best_signal_per_strategy(self, symbol: str, min_confidence: float = 0.65) -> Dict[str, Dict]:
        """
        Get the best signal from EACH strategy category.

        Returns a dict mapping strategy name to best signal:
        {
            'Momentum': {'action': 'BUY', 'confidence': 0.85, 'model': 'Momentum_Selective', ...},
            'MeanReversion': {'action': 'SELL', 'confidence': 0.78, 'model': 'MeanReversion_Moderate', ...},
            ...
        }

        Each strategy can independently trigger trades.
        """
        if not self.ai_brain or not self.ai_brain.loaded:
            return {}

        if not self.broker:
            return {}

        try:
            # Get timeframe config based on trading mode
            config = self.timeframe_config[self.trading_mode]

            # Fetch historical data
            bars = self.broker.get_bars(
                symbol=symbol,
                timeframe=config['timeframe'],
                limit=config['limit']
            )

            if not bars or len(bars) < config['min_bars']:
                return {}

            # Convert to DataFrame
            df = pd.DataFrame(bars)
            column_map = {
                't': 'timestamp',
                'o': 'open',
                'h': 'high',
                'l': 'low',
                'c': 'close',
                'v': 'volume',
                'vw': 'vwap'
            }
            df = df.rename(columns=column_map)

            # Get predictions from all models
            all_predictions = self.ai_brain.get_all_predictions(df)

            # Group by strategy and find the best signal in each
            strategy_signals = {}

            for pred in all_predictions:
                model_name = pred.get('model', '')
                action = pred.get('action', 'HOLD')
                confidence = pred.get('confidence', 0)

                # Skip HOLD signals or low confidence
                if action == 'HOLD' or confidence < min_confidence:
                    continue

                # Extract strategy from model name (e.g., "Momentum_Selective" -> "Momentum")
                strategy = model_name.rsplit('_', 1)[0] if '_' in model_name else model_name

                # Keep the highest confidence signal for each strategy
                if strategy not in strategy_signals or confidence > strategy_signals[strategy].get('confidence', 0):
                    pred['symbol'] = symbol
                    pred['strategy'] = strategy
                    pred['timestamp'] = datetime.now().isoformat()
                    pred['trading_mode'] = self.trading_mode
                    strategy_signals[strategy] = pred

            return strategy_signals

        except Exception as e:
            self.error.emit(f"Strategy signals error for {symbol}: {e}")
            return {}

    def run(self):
        """Main worker loop - scans symbols periodically."""
        self.running = True

        while self.running:
            self.mutex.lock()
            symbols = list(self.symbols_to_scan)
            self.mutex.unlock()

            for symbol in symbols:
                if not self.running:
                    break

                prediction = self.get_prediction_for_symbol(symbol)
                if prediction:
                    self.prediction_ready.emit(prediction)

            # Wait before next scan
            self.msleep(self.scan_interval)

    def stop(self):
        """Stop the worker."""
        self.running = False

    def scan_once(self, symbol: str):
        """Run a single prediction scan (non-blocking)."""
        prediction = self.get_prediction_for_symbol(symbol)
        if prediction:
            self.prediction_ready.emit(prediction)

    def get_model_list(self) -> List[Dict]:
        """Get list of all available models with their stats."""
        if not self.ai_brain or not self.ai_brain.loaded:
            return []

        models = []
        for name in self.ai_brain.model_priority:
            model_data = self.ai_brain.models.get(name, {})
            risk_params = get_model_risk_params(name) if AI_AVAILABLE else {}

            models.append({
                'name': name,
                'strategy': model_data.get('config', {}).get('strategy', 'Unknown'),
                'win_rate': model_data.get('win_rate', 0),
                'threshold': model_data.get('config', {}).get('signal_threshold', 0),
                'stop_loss_pct': risk_params.get('stop_loss_pct', 0.02),
                'take_profit_pct': risk_params.get('take_profit_pct', 0.06),
                'risk_reward': risk_params.get('take_profit_pct', 0.06) / max(risk_params.get('stop_loss_pct', 0.02), 0.001),
                'description': risk_params.get('description', '')
            })

        return models


def create_sample_prediction(symbol: str = "AAPL") -> Dict:
    """Create a sample prediction for testing without models."""
    return {
        'symbol': symbol,
        'model': 'Momentum_Selective',
        'strategy': 'Momentum',
        'action': 'BUY',
        'signal': 0.75,
        'confidence': 0.948,
        'buy_prob': 0.95,
        'sell_prob': 0.02,
        'win_rate': 0.948,
        'stop_loss_pct': 0.018,
        'take_profit_pct': 0.054,
        'trailing_activation_pct': 0.02,
        'trailing_stop_pct': 0.012,
        'risk_reward_ratio': 3.0,
        'risk_description': 'High-confidence momentum, tight risk management',
        'timestamp': datetime.now().isoformat()
    }
