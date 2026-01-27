"""
AI Model Brain - Priority-Based Trading System
===============================================
Loads trained models from checkpoint and makes predictions
using a priority system based on win rate.

The model with the highest win rate gets first priority.
If it can't make a confident prediction, it falls to the next model.
"""

import pickle
import numpy as np
import pandas as pd
from pathlib import Path
from datetime import datetime
import threading
from typing import Dict, List, Optional, Tuple

# Paths - check multiple locations
_BASE_DIR = Path(__file__).parent
CHECKPOINT_PATH = Path("C:/TrainingData5Years/checkpoints_multistrategy/checkpoint_latest.pkl")
FINAL_MODEL_PATH = Path("C:/Users/steve/Desktop/Stocks 7.1/Trade 6.0/models/multi_strategy_30_system.pkl")
LOCAL_MODEL_PATH = _BASE_DIR / "multi_strategy_30_system.pkl"  # Local copy in 30Bots dir


# =============================================================================
# OPTIMAL RISK PARAMETERS FOR EACH MODEL
# =============================================================================
# Derived from strategy characteristics, win rates, and backtesting principles:
#
# STOP LOSS LOGIC:
#   - Higher win rate = tighter stops (fewer false signals to worry about)
#   - Selective = tighter, Aggressive = wider (more trades need room)
#   - Momentum/Trend = wider (ride the trend), MeanReversion = tighter (quick reversals)
#
# TAKE PROFIT LOGIC:
#   - Risk:Reward minimum 2:1 for all models
#   - Momentum/Breakout/Trend = higher targets (capture full moves)
#   - MeanReversion/VWAP = moderate targets (mean reversion is bounded)
#
# TRAILING STOP LOGIC:
#   - Activation: When to start trailing (after X% gain)
#   - Trail %: How tight to trail once activated
#   - Trend strategies = looser trail, Reversal strategies = tighter trail
#
MODEL_RISK_PARAMS = {
    # =========================================================================
    # MOMENTUM STRATEGY - Rides price momentum, needs room for pullbacks
    # =========================================================================
    'Momentum_Selective': {
        'stop_loss_pct': 0.018,        # 1.8% - tight (94.8% win rate)
        'take_profit_pct': 0.054,      # 5.4% - 3:1 R:R
        'trailing_activation_pct': 0.02,  # Activate at 2% gain
        'trailing_stop_pct': 0.012,    # Trail by 1.2%
        'description': 'High-confidence momentum, tight risk management'
    },
    'Momentum_Moderate': {
        'stop_loss_pct': 0.025,        # 2.5% - moderate (90.9% win rate)
        'take_profit_pct': 0.065,      # 6.5% - 2.6:1 R:R
        'trailing_activation_pct': 0.025,
        'trailing_stop_pct': 0.015,
        'description': 'Balanced momentum with moderate risk tolerance'
    },
    'Momentum_Aggressive': {
        'stop_loss_pct': 0.035,        # 3.5% - wider (86.8% win rate)
        'take_profit_pct': 0.085,      # 8.5% - 2.4:1 R:R
        'trailing_activation_pct': 0.03,
        'trailing_stop_pct': 0.02,
        'description': 'Aggressive momentum capture with wider stops'
    },

    # =========================================================================
    # MEAN REVERSION - Quick reversals to mean, tight stops essential
    # =========================================================================
    'MeanReversion_Selective': {
        'stop_loss_pct': 0.015,        # 1.5% - very tight (94.6% win rate)
        'take_profit_pct': 0.04,       # 4% - 2.7:1 R:R
        'trailing_activation_pct': 0.015,
        'trailing_stop_pct': 0.008,    # Tight trail for reversals
        'description': 'High-probability mean reversion with tight risk'
    },
    'MeanReversion_Moderate': {
        'stop_loss_pct': 0.02,         # 2% (90.9% win rate)
        'take_profit_pct': 0.05,       # 5% - 2.5:1 R:R
        'trailing_activation_pct': 0.018,
        'trailing_stop_pct': 0.01,
        'description': 'Balanced mean reversion'
    },
    'MeanReversion_Aggressive': {
        'stop_loss_pct': 0.028,        # 2.8% (87.1% win rate)
        'take_profit_pct': 0.065,      # 6.5% - 2.3:1 R:R
        'trailing_activation_pct': 0.022,
        'trailing_stop_pct': 0.012,
        'description': 'Aggressive mean reversion with extended targets'
    },

    # =========================================================================
    # BREAKOUT - Volatile entries, needs room for false breakouts
    # =========================================================================
    'Breakout_Selective': {
        'stop_loss_pct': 0.022,        # 2.2% (88.9% win rate)
        'take_profit_pct': 0.07,       # 7% - 3.2:1 R:R (breakouts can run)
        'trailing_activation_pct': 0.025,
        'trailing_stop_pct': 0.015,
        'description': 'High-quality breakouts with extended targets'
    },
    'Breakout_Moderate': {
        'stop_loss_pct': 0.03,         # 3% (79.8% win rate)
        'take_profit_pct': 0.08,       # 8% - 2.7:1 R:R
        'trailing_activation_pct': 0.03,
        'trailing_stop_pct': 0.018,
        'description': 'Balanced breakout trading'
    },
    'Breakout_Aggressive': {
        'stop_loss_pct': 0.04,         # 4% (70.6% win rate - needs room)
        'take_profit_pct': 0.10,       # 10% - 2.5:1 R:R
        'trailing_activation_pct': 0.035,
        'trailing_stop_pct': 0.022,
        'description': 'Aggressive breakouts with wide stops for volatility'
    },

    # =========================================================================
    # TREND FOLLOWING - Long holds, trailing stops critical
    # =========================================================================
    'TrendFollowing_Selective': {
        'stop_loss_pct': 0.02,         # 2% (90.7% win rate)
        'take_profit_pct': 0.07,       # 7% - 3.5:1 R:R
        'trailing_activation_pct': 0.02,
        'trailing_stop_pct': 0.015,    # Moderate trail to ride trends
        'description': 'High-confidence trend following'
    },
    'TrendFollowing_Moderate': {
        'stop_loss_pct': 0.028,        # 2.8% (83.4% win rate)
        'take_profit_pct': 0.075,      # 7.5% - 2.7:1 R:R
        'trailing_activation_pct': 0.025,
        'trailing_stop_pct': 0.018,
        'description': 'Balanced trend following with room to breathe'
    },
    'TrendFollowing_Aggressive': {
        'stop_loss_pct': 0.038,        # 3.8% (76.4% win rate)
        'take_profit_pct': 0.095,      # 9.5% - 2.5:1 R:R
        'trailing_activation_pct': 0.03,
        'trailing_stop_pct': 0.022,
        'description': 'Aggressive trend capture with extended runs'
    },

    # =========================================================================
    # GAP TRADING - Quick intraday moves, tight management
    # =========================================================================
    'GapTrading_Selective': {
        'stop_loss_pct': 0.018,        # 1.8% (89.9% win rate)
        'take_profit_pct': 0.045,      # 4.5% - 2.5:1 R:R
        'trailing_activation_pct': 0.015,
        'trailing_stop_pct': 0.01,
        'description': 'High-probability gap fills'
    },
    'GapTrading_Moderate': {
        'stop_loss_pct': 0.025,        # 2.5% (82.4% win rate)
        'take_profit_pct': 0.055,      # 5.5% - 2.2:1 R:R
        'trailing_activation_pct': 0.02,
        'trailing_stop_pct': 0.012,
        'description': 'Balanced gap trading'
    },
    'GapTrading_Aggressive': {
        'stop_loss_pct': 0.032,        # 3.2% (76.5% win rate)
        'take_profit_pct': 0.07,       # 7% - 2.2:1 R:R
        'trailing_activation_pct': 0.025,
        'trailing_stop_pct': 0.015,
        'description': 'Aggressive gap trading with extended targets'
    },

    # =========================================================================
    # MULTI-INDICATOR - Confluence-based, balanced approach
    # =========================================================================
    'MultiIndicator_Selective': {
        'stop_loss_pct': 0.02,         # 2% (88.1% win rate)
        'take_profit_pct': 0.055,      # 5.5% - 2.75:1 R:R
        'trailing_activation_pct': 0.02,
        'trailing_stop_pct': 0.012,
        'description': 'High-confluence signals with balanced risk'
    },
    'MultiIndicator_Moderate': {
        'stop_loss_pct': 0.028,        # 2.8% (79.8% win rate)
        'take_profit_pct': 0.065,      # 6.5% - 2.3:1 R:R
        'trailing_activation_pct': 0.025,
        'trailing_stop_pct': 0.015,
        'description': 'Balanced multi-indicator approach'
    },
    'MultiIndicator_Aggressive': {
        'stop_loss_pct': 0.038,        # 3.8% (72.2% win rate)
        'take_profit_pct': 0.085,      # 8.5% - 2.2:1 R:R
        'trailing_activation_pct': 0.03,
        'trailing_stop_pct': 0.018,
        'description': 'Aggressive confluence trading'
    },

    # =========================================================================
    # VWAP - Intraday mean reversion to VWAP
    # =========================================================================
    'VWAP_Selective': {
        'stop_loss_pct': 0.015,        # 1.5% (90% win rate)
        'take_profit_pct': 0.04,       # 4% - 2.7:1 R:R
        'trailing_activation_pct': 0.015,
        'trailing_stop_pct': 0.008,
        'description': 'High-probability VWAP reversion'
    },
    'VWAP_Moderate': {
        'stop_loss_pct': 0.022,        # 2.2% (82.3% win rate)
        'take_profit_pct': 0.05,       # 5% - 2.3:1 R:R
        'trailing_activation_pct': 0.018,
        'trailing_stop_pct': 0.01,
        'description': 'Balanced VWAP trading'
    },
    'VWAP_Aggressive': {
        'stop_loss_pct': 0.03,         # 3% (75.9% win rate)
        'take_profit_pct': 0.065,      # 6.5% - 2.2:1 R:R
        'trailing_activation_pct': 0.022,
        'trailing_stop_pct': 0.012,
        'description': 'Aggressive VWAP strategies'
    },

    # =========================================================================
    # RSI DIVERGENCE - Reversal signals, lower win rate = wider stops
    # =========================================================================
    'RSIDivergence_Selective': {
        'stop_loss_pct': 0.035,        # 3.5% (50% win rate - needs room)
        'take_profit_pct': 0.09,       # 9% - 2.6:1 R:R (compensate low win rate)
        'trailing_activation_pct': 0.03,
        'trailing_stop_pct': 0.018,
        'description': 'RSI divergence with risk-adjusted targets'
    },
    'RSIDivergence_Moderate': {
        'stop_loss_pct': 0.04,         # 4% (48.1% win rate)
        'take_profit_pct': 0.10,       # 10% - 2.5:1 R:R
        'trailing_activation_pct': 0.035,
        'trailing_stop_pct': 0.02,
        'description': 'RSI divergence with extended targets'
    },
    'RSIDivergence_Aggressive': {
        'stop_loss_pct': 0.05,         # 5% (45.4% win rate - high risk)
        'take_profit_pct': 0.12,       # 12% - 2.4:1 R:R
        'trailing_activation_pct': 0.04,
        'trailing_stop_pct': 0.025,
        'description': 'Aggressive RSI divergence plays'
    },

    # =========================================================================
    # BOLLINGER BANDS - Band-based mean reversion
    # =========================================================================
    'BollingerBands_Selective': {
        'stop_loss_pct': 0.025,        # 2.5% (76.5% win rate)
        'take_profit_pct': 0.06,       # 6% - 2.4:1 R:R
        'trailing_activation_pct': 0.02,
        'trailing_stop_pct': 0.012,
        'description': 'High-quality Bollinger Band signals'
    },
    'BollingerBands_Moderate': {
        'stop_loss_pct': 0.032,        # 3.2% (68% win rate)
        'take_profit_pct': 0.075,      # 7.5% - 2.3:1 R:R
        'trailing_activation_pct': 0.025,
        'trailing_stop_pct': 0.015,
        'description': 'Balanced Bollinger Band trading'
    },
    'BollingerBands_Aggressive': {
        'stop_loss_pct': 0.042,        # 4.2% (52.4% win rate)
        'take_profit_pct': 0.095,      # 9.5% - 2.3:1 R:R
        'trailing_activation_pct': 0.032,
        'trailing_stop_pct': 0.02,
        'description': 'Aggressive Bollinger breakout/reversal'
    },

    # =========================================================================
    # VOLUME SPIKE - Event-driven, highly volatile
    # =========================================================================
    'VolumeSpike_Selective': {
        'stop_loss_pct': 0.028,        # 2.8% (70.6% win rate)
        'take_profit_pct': 0.07,       # 7% - 2.5:1 R:R
        'trailing_activation_pct': 0.025,
        'trailing_stop_pct': 0.015,
        'description': 'High-quality volume spike entries'
    },
    'VolumeSpike_Moderate': {
        'stop_loss_pct': 0.035,        # 3.5% (64.1% win rate)
        'take_profit_pct': 0.085,      # 8.5% - 2.4:1 R:R
        'trailing_activation_pct': 0.03,
        'trailing_stop_pct': 0.018,
        'description': 'Balanced volume spike trading'
    },
    'VolumeSpike_Aggressive': {
        'stop_loss_pct': 0.045,        # 4.5% (50.2% win rate)
        'take_profit_pct': 0.11,       # 11% - 2.4:1 R:R
        'trailing_activation_pct': 0.038,
        'trailing_stop_pct': 0.022,
        'description': 'Aggressive volume spike plays'
    },
}

# Default risk parameters for unknown models
DEFAULT_RISK_PARAMS = {
    'stop_loss_pct': 0.025,
    'take_profit_pct': 0.06,
    'trailing_activation_pct': 0.02,
    'trailing_stop_pct': 0.012,
    'description': 'Default balanced parameters'
}


def get_model_risk_params(model_name: str) -> Dict:
    """
    Get the optimal risk parameters for a specific model.

    Returns:
        Dict with keys: stop_loss_pct, take_profit_pct,
                       trailing_activation_pct, trailing_stop_pct
    """
    return MODEL_RISK_PARAMS.get(model_name, DEFAULT_RISK_PARAMS).copy()


def calculate_risk_prices(entry_price: float, model_name: str) -> Dict:
    """
    Calculate actual stop loss and take profit prices for a model.

    Args:
        entry_price: The entry price for the position
        model_name: Name of the model (e.g., 'Momentum_Selective')

    Returns:
        Dict with stop_loss_price, take_profit_price, trailing_activation_price,
        and all percentage values
    """
    params = get_model_risk_params(model_name)

    return {
        'model': model_name,
        'entry_price': entry_price,
        'stop_loss_price': round(entry_price * (1 - params['stop_loss_pct']), 4),
        'take_profit_price': round(entry_price * (1 + params['take_profit_pct']), 4),
        'trailing_activation_price': round(entry_price * (1 + params['trailing_activation_pct']), 4),
        'stop_loss_pct': params['stop_loss_pct'],
        'take_profit_pct': params['take_profit_pct'],
        'trailing_activation_pct': params['trailing_activation_pct'],
        'trailing_stop_pct': params['trailing_stop_pct'],
        'risk_reward_ratio': round(params['take_profit_pct'] / params['stop_loss_pct'], 2),
        'description': params['description']
    }


class AIModelBrain:
    """
    Priority-based AI trading brain using trained strategy models.
    """

    def __init__(self):
        self.models = {}
        self.model_priority = []  # Ordered by win rate (highest first)
        self.loaded = False
        self.last_load_time = None
        self._lock = threading.Lock()

        # Load models on init
        self.reload_models()

    def reload_models(self) -> bool:
        """Load or reload models from checkpoint/final file."""
        with self._lock:
            try:
                # Try checkpoint first (for live training), then final model, then local
                model_path = None
                if CHECKPOINT_PATH.exists():
                    model_path = CHECKPOINT_PATH
                elif FINAL_MODEL_PATH.exists():
                    model_path = FINAL_MODEL_PATH
                elif LOCAL_MODEL_PATH.exists():
                    model_path = LOCAL_MODEL_PATH

                if not model_path:
                    print("[AI BRAIN] No model file found")
                    print(f"[AI BRAIN] Checked: {CHECKPOINT_PATH}, {FINAL_MODEL_PATH}, {LOCAL_MODEL_PATH}")
                    return False

                with open(model_path, 'rb') as f:
                    data = pickle.load(f)

                # Handle both checkpoint and final format
                if 'all_models' in data:
                    self.models = data['all_models']
                elif 'trained_models' in data:
                    self.models = data['trained_models']
                else:
                    print("[AI BRAIN] Invalid model format")
                    return False

                # Sort by win rate (highest first) to create priority order
                sorted_models = sorted(
                    self.models.items(),
                    key=lambda x: x[1].get('win_rate', 0),
                    reverse=True
                )

                self.model_priority = [name for name, _ in sorted_models]
                self.loaded = True
                self.last_load_time = datetime.now()

                print(f"[AI BRAIN] Loaded {len(self.models)} models")
                print(f"[AI BRAIN] Top 3 by win rate:")
                for i, name in enumerate(self.model_priority[:3]):
                    wr = self.models[name]['win_rate'] * 100
                    print(f"  #{i+1} {name}: {wr:.1f}%")

                return True

            except Exception as e:
                print(f"[AI BRAIN] Error loading models: {e}")
                return False

    def compute_features(self, df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Compute features for a specific strategy."""
        df = df.copy()

        # Ensure we have required columns
        required = ['open', 'high', 'low', 'close', 'volume']
        for col in required:
            if col not in df.columns:
                # Try lowercase
                lower_col = col.lower()
                if lower_col in df.columns:
                    df[col] = df[lower_col]
                else:
                    return df, []

        # Base features (used by all strategies)
        df['returns'] = df['close'].pct_change()
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)  # Needed by trained models

        # Moving averages
        for period in [5, 10, 20, 50, 100]:
            df[f'sma_{period}'] = df['close'].rolling(period).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period).mean()

        # Trend strength
        df['trend_strength'] = abs(df['sma_20'] - df['sma_50']) / (df['close'] + 1e-10)

        # Volatility
        df['volatility'] = df['returns'].rolling(20).std()
        df['volatility_ratio'] = df['volatility'] / (df['volatility'].rolling(50).mean() + 1e-10)

        # ATR (Average True Range) - used by multiple strategies
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift(1))
        low_close = abs(df['low'] - df['close'].shift(1))
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14).mean()
        df['atr_pct'] = df['atr'] / (df['close'] + 1e-10)

        # Additional features needed by trained models
        df['returns_2'] = df['close'].pct_change(2)
        df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['consolidation'] = df['close'].rolling(10).std() / (df['close'].rolling(50).std() + 1e-10)

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp12 = df['close'].ewm(span=12).mean()
        exp26 = df['close'].ewm(span=26).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_mid = df['close'].rolling(20).mean()
        bb_std = df['close'].rolling(20).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-10)

        # VWAP approximation (intraday would need true VWAP)
        if 'vw' in df.columns:
            # Polygon provides vw (volume-weighted average price)
            df['vwap'] = df['vw']
        else:
            df['vwap'] = (df['close'] * df['volume']).cumsum() / (df['volume'].cumsum() + 1e-10)
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)

        # Volume
        df['volume_sma'] = df['volume'].rolling(20).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

        # Price position
        df['high_20'] = df['high'].rolling(20).max()
        df['low_20'] = df['low'].rolling(20).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-10)

        # Strategy-specific features - these must match training script exactly
        if strategy == 'Momentum':
            # Training used: returns_5, returns_10, returns_20, rsi, macd_hist, volume_ratio, price_position, volatility_ratio
            features = ['returns_5', 'returns_10', 'returns_20', 'rsi', 'macd_hist',
                       'volume_ratio', 'price_position', 'volatility_ratio']

        elif strategy == 'MeanReversion':
            # Training used: rsi, bb_position, price_vs_vwap, returns_5, volatility, volume_ratio, atr_pct, price_position
            features = ['rsi', 'bb_position', 'price_vs_vwap', 'returns_5', 'volatility',
                       'volume_ratio', 'atr_pct', 'price_position']

        elif strategy == 'Breakout':
            # Training used: price_position, bb_width, volume_ratio, atr_pct, volatility_ratio, returns_2, high_low_range, consolidation
            features = ['price_position', 'bb_width', 'volume_ratio', 'atr_pct',
                       'volatility_ratio', 'returns_2', 'high_low_range', 'consolidation']

        elif strategy == 'TrendFollowing':
            # Training used: sma_5_20_cross, sma_10_50_cross, sma_20_100_cross, ema_trend, macd, macd_signal, trend_strength, volume_ratio
            df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(float)
            df['sma_10_50_cross'] = (df['sma_10'] > df['sma_50']).astype(float)
            df['sma_20_100_cross'] = (df['sma_20'] > df['sma_100']).astype(float)
            df['ema_trend'] = (df['ema_10'] > df['ema_20']).astype(float)
            features = ['sma_5_20_cross', 'sma_10_50_cross', 'sma_20_100_cross',
                       'ema_trend', 'macd', 'macd_signal', 'trend_strength', 'volume_ratio']

        elif strategy == 'VWAP':
            df['vwap'] = (df['close'] * df['volume']).cumsum() / df['volume'].cumsum()
            df['price_vs_vwap'] = (df['close'] - df['vwap']) / df['vwap']
            features = ['price_vs_vwap', 'volume_ratio', 'rsi', 'returns',
                       'bb_position', 'macd_hist', 'price_position', 'volatility']

        elif strategy == 'RSIDivergence':
            df['rsi_slope'] = df['rsi'].diff(5)
            df['price_slope'] = df['returns_5']
            df['divergence'] = ((df['rsi_slope'] > 0) & (df['price_slope'] < 0) |
                               (df['rsi_slope'] < 0) & (df['price_slope'] > 0)).astype(float)
            features = ['rsi', 'rsi_slope', 'price_slope', 'divergence',
                       'bb_position', 'volume_ratio', 'macd_hist', 'returns']

        elif strategy == 'BollingerBands':
            df['bb_squeeze'] = (df['bb_upper'] - df['bb_lower']) / bb_mid
            df['bb_breakout'] = ((df['close'] > df['bb_upper']) | (df['close'] < df['bb_lower'])).astype(float)
            features = ['bb_position', 'bb_squeeze', 'bb_breakout', 'rsi',
                       'volume_ratio', 'returns', 'macd_hist', 'volatility']

        elif strategy == 'VolumeSpike':
            df['volume_spike'] = (df['volume_ratio'] > 2).astype(float)
            df['price_volume_trend'] = df['returns'] * df['volume_ratio']
            features = ['volume_ratio', 'volume_spike', 'price_volume_trend',
                       'returns', 'rsi', 'macd_hist', 'bb_position', 'price_position']

        elif strategy == 'GapTrading':
            df['gap'] = (df['open'] - df['close'].shift(1)) / df['close'].shift(1)
            df['gap_up'] = (df['gap'] > 0.01).astype(float)
            df['gap_down'] = (df['gap'] < -0.01).astype(float)
            features = ['gap', 'gap_up', 'gap_down', 'volume_ratio',
                       'rsi', 'returns', 'bb_position', 'price_position']

        elif strategy == 'MultiIndicator':
            df['rsi_bullish'] = (df['rsi'] < 40).astype(float)
            df['macd_bullish'] = (df['macd_hist'] > 0).astype(float)
            df['bb_bullish'] = (df['bb_position'] < 0.3).astype(float)
            df['trend_bullish'] = (df['sma_20'] > df['sma_50']).astype(float)
            df['signal_confluence'] = df['rsi_bullish'] + df['macd_bullish'] + df['bb_bullish'] + df['trend_bullish']
            features = ['signal_confluence', 'rsi', 'macd_hist', 'bb_position',
                       'volume_ratio', 'returns', 'price_position', 'volatility']

        else:
            # Default features
            features = ['returns', 'rsi', 'macd_hist', 'bb_position',
                       'volume_ratio', 'price_position', 'volatility', 'returns_5']

        return df, features

    def predict_single_model(self, model_name: str, df: pd.DataFrame) -> Optional[Dict]:
        """Get prediction from a single model."""
        if model_name not in self.models:
            return None

        model_data = self.models[model_name]
        strategy = model_data['config']['strategy']
        threshold = model_data['config']['signal_threshold']

        try:
            # Compute features
            df_features, feature_names = self.compute_features(df, strategy)
            # Forward fill NaN values to preserve more rows, then drop any remaining NaN in last row
            df_features = df_features.ffill().bfill()

            if len(df_features) < 10:
                return None

            # Get model's expected features
            model_features = model_data.get('features', feature_names)

            # Ensure all features exist
            valid_features = [f for f in model_features if f in df_features.columns]
            if len(valid_features) < 3:
                return None

            # Get latest row
            X = df_features[valid_features].iloc[-1:].values

            # Scale - but only if feature count matches
            scaler = model_data.get('scaler')
            if scaler:
                expected_features = scaler.n_features_in_
                if X.shape[1] == expected_features:
                    X = scaler.transform(X)
                # else: features don't match - use raw values

            # Predict
            model = model_data.get('model')
            if model is None:
                return None

            proba = model.predict_proba(X)[0]
            # Class 0 = hold, Class 1 = buy, Class 2 = sell
            buy_prob = proba[1] if len(proba) > 1 else 0
            sell_prob = proba[2] if len(proba) > 2 else 0

            # Calculate signal strength
            signal = buy_prob - sell_prob

            # Determine action based on threshold
            if signal > threshold:
                action = 'BUY'
                confidence = buy_prob
            elif signal < -threshold:
                action = 'SELL'
                confidence = sell_prob
            else:
                action = 'HOLD'
                confidence = 1 - buy_prob - sell_prob

            # Get optimal risk parameters for this model
            risk_params = get_model_risk_params(model_name)

            # Get current price for calculating actual stop/target prices
            current_price = df['close'].iloc[-1] if 'close' in df.columns else None

            result = {
                'model': model_name,
                'strategy': strategy,
                'action': action,
                'signal': signal,
                'confidence': confidence,
                'buy_prob': buy_prob,
                'sell_prob': sell_prob,
                'win_rate': model_data['win_rate'],
                'threshold': threshold,
                # Risk parameters (percentages)
                'stop_loss_pct': risk_params['stop_loss_pct'],
                'take_profit_pct': risk_params['take_profit_pct'],
                'trailing_activation_pct': risk_params['trailing_activation_pct'],
                'trailing_stop_pct': risk_params['trailing_stop_pct'],
                'risk_reward_ratio': round(risk_params['take_profit_pct'] / risk_params['stop_loss_pct'], 2),
                'risk_description': risk_params['description']
            }

            # Add calculated prices if current price available
            if current_price is not None:
                result['current_price'] = current_price
                result['stop_loss_price'] = round(current_price * (1 - risk_params['stop_loss_pct']), 4)
                result['take_profit_price'] = round(current_price * (1 + risk_params['take_profit_pct']), 4)
                result['trailing_activation_price'] = round(current_price * (1 + risk_params['trailing_activation_pct']), 4)

            return result

        except Exception as e:
            return None

    def get_prediction(self, df: pd.DataFrame, min_confidence: float = 0.6) -> Optional[Dict]:
        """
        Get trading prediction using priority system.

        Goes through models in order of win rate.
        Returns the first model that gives a confident BUY or SELL signal.
        """
        if not self.loaded or not self.models:
            self.reload_models()
            if not self.loaded:
                return None

        # Try each model in priority order
        for model_name in self.model_priority:
            result = self.predict_single_model(model_name, df)

            if result and result['action'] != 'HOLD':
                if result['confidence'] >= min_confidence:
                    result['priority_rank'] = self.model_priority.index(model_name) + 1
                    return result

        # No confident signal from any model
        return {
            'model': 'consensus',
            'strategy': 'all',
            'action': 'HOLD',
            'signal': 0,
            'confidence': 0,
            'reason': 'No model gave confident signal'
        }

    def get_all_predictions(self, df: pd.DataFrame) -> List[Dict]:
        """Get predictions from all models (for analysis)."""
        if not self.loaded:
            self.reload_models()

        results = []
        for model_name in self.model_priority:
            result = self.predict_single_model(model_name, df)
            if result:
                result['priority_rank'] = self.model_priority.index(model_name) + 1
                results.append(result)

        return results

    def get_status(self) -> Dict:
        """Get brain status."""
        return {
            'loaded': self.loaded,
            'model_count': len(self.models),
            'last_load': self.last_load_time.isoformat() if self.last_load_time else None,
            'top_models': [
                {
                    'name': name,
                    'win_rate': self.models[name]['win_rate'] * 100,
                    'strategy': self.models[name]['config']['strategy']
                }
                for name in self.model_priority[:5]
            ] if self.models else []
        }

    def get_risk_params(self, model_name: str) -> Dict:
        """
        Get optimal risk parameters for a specific model.

        Args:
            model_name: Name of the model (e.g., 'Momentum_Selective')

        Returns:
            Dict with stop_loss_pct, take_profit_pct, trailing_activation_pct,
            trailing_stop_pct, and risk_reward_ratio
        """
        return get_model_risk_params(model_name)

    def calculate_position_risk(self, model_name: str, entry_price: float,
                                 shares: int) -> Dict:
        """
        Calculate complete risk profile for a position.

        Args:
            model_name: Name of the model used for this trade
            entry_price: Entry price per share
            shares: Number of shares

        Returns:
            Complete risk calculation including dollar amounts
        """
        params = get_model_risk_params(model_name)
        position_value = entry_price * shares

        stop_loss_price = entry_price * (1 - params['stop_loss_pct'])
        take_profit_price = entry_price * (1 + params['take_profit_pct'])
        trailing_activation_price = entry_price * (1 + params['trailing_activation_pct'])

        return {
            'model': model_name,
            'entry_price': entry_price,
            'shares': shares,
            'position_value': round(position_value, 2),
            # Stop loss
            'stop_loss_price': round(stop_loss_price, 4),
            'stop_loss_pct': params['stop_loss_pct'],
            'max_loss_dollars': round((entry_price - stop_loss_price) * shares, 2),
            # Take profit
            'take_profit_price': round(take_profit_price, 4),
            'take_profit_pct': params['take_profit_pct'],
            'target_profit_dollars': round((take_profit_price - entry_price) * shares, 2),
            # Trailing stop
            'trailing_activation_price': round(trailing_activation_price, 4),
            'trailing_activation_pct': params['trailing_activation_pct'],
            'trailing_stop_pct': params['trailing_stop_pct'],
            # Risk metrics
            'risk_reward_ratio': round(params['take_profit_pct'] / params['stop_loss_pct'], 2),
            'description': params['description']
        }

    def get_all_risk_params(self) -> Dict:
        """
        Get risk parameters for all 30 models.

        Returns:
            Dict mapping model names to their risk parameters
        """
        return MODEL_RISK_PARAMS.copy()

    def print_risk_summary(self):
        """Print a formatted summary of all model risk parameters."""
        print("\n" + "=" * 80)
        print("OPTIMAL RISK PARAMETERS FOR ALL 30 MODELS")
        print("=" * 80)

        strategies = {}
        for name, params in MODEL_RISK_PARAMS.items():
            strategy = name.rsplit('_', 1)[0]
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append((name, params))

        for strategy, models in sorted(strategies.items()):
            print(f"\n{strategy}")
            print("-" * 60)
            print(f"{'Model':<30} {'SL%':>6} {'TP%':>6} {'R:R':>5} {'Trail':>6}")
            print("-" * 60)
            for name, params in models:
                level = name.split('_')[-1]
                sl = params['stop_loss_pct'] * 100
                tp = params['take_profit_pct'] * 100
                rr = tp / sl
                trail = params['trailing_stop_pct'] * 100
                print(f"  {level:<28} {sl:>5.1f}% {tp:>5.1f}% {rr:>5.1f} {trail:>5.1f}%")


# Singleton instance
_brain_instance = None

def get_30model_brain() -> AIModelBrain:
    """Get singleton 30-model AI brain instance."""
    global _brain_instance
    if _brain_instance is None:
        _brain_instance = AIModelBrain()
    return _brain_instance
