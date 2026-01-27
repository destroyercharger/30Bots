"""
Multi-Strategy Model Predictor
==============================
Uses the 21 trained XGBoost models to generate trading signals.
Each model represents a different strategy at a specific selectivity level.

Trained on 6.9 BILLION simulations with win rates from 70.6% to 94.8%.
"""

import numpy as np
import pandas as pd
import pickle
from pathlib import Path
from typing import Dict, List, Optional, Tuple
from datetime import datetime


class StrategyFeatures:
    """Compute features for each strategy - matches training exactly."""

    @staticmethod
    def compute_base_features(df: pd.DataFrame) -> pd.DataFrame:
        """Compute base technical indicators used by all strategies."""
        df = df.copy()

        # Price changes
        df['returns'] = df['close'].pct_change()
        df['returns_2'] = df['close'].pct_change(2)
        df['returns_5'] = df['close'].pct_change(5)
        df['returns_10'] = df['close'].pct_change(10)
        df['returns_20'] = df['close'].pct_change(20)

        # Moving averages
        for period in [5, 10, 20, 50, 100, 200]:
            df[f'sma_{period}'] = df['close'].rolling(period, min_periods=1).mean()
            df[f'ema_{period}'] = df['close'].ewm(span=period, min_periods=1).mean()

        # Volatility
        df['volatility'] = df['returns'].rolling(20, min_periods=1).std()
        df['volatility_ratio'] = df['volatility'] / df['volatility'].rolling(50, min_periods=1).mean()

        # RSI
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(14, min_periods=1).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(14, min_periods=1).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))

        # MACD
        exp12 = df['close'].ewm(span=12, min_periods=1).mean()
        exp26 = df['close'].ewm(span=26, min_periods=1).mean()
        df['macd'] = exp12 - exp26
        df['macd_signal'] = df['macd'].ewm(span=9, min_periods=1).mean()
        df['macd_hist'] = df['macd'] - df['macd_signal']

        # Bollinger Bands
        bb_mid = df['close'].rolling(20, min_periods=1).mean()
        bb_std = df['close'].rolling(20, min_periods=1).std()
        df['bb_upper'] = bb_mid + 2 * bb_std
        df['bb_lower'] = bb_mid - 2 * bb_std
        df['bb_position'] = (df['close'] - df['bb_lower']) / (df['bb_upper'] - df['bb_lower'] + 1e-10)
        df['bb_width'] = (df['bb_upper'] - df['bb_lower']) / (bb_mid + 1e-10)

        # Volume
        df['volume_sma'] = df['volume'].rolling(20, min_periods=1).mean()
        df['volume_ratio'] = df['volume'] / (df['volume_sma'] + 1e-10)

        # ATR
        high_low = df['high'] - df['low']
        high_close = abs(df['high'] - df['close'].shift())
        low_close = abs(df['low'] - df['close'].shift())
        tr = pd.concat([high_low, high_close, low_close], axis=1).max(axis=1)
        df['atr'] = tr.rolling(14, min_periods=1).mean()
        df['atr_pct'] = df['atr'] / (df['close'] + 1e-10)

        # Price position
        df['high_20'] = df['high'].rolling(20, min_periods=1).max()
        df['low_20'] = df['low'].rolling(20, min_periods=1).min()
        df['price_position'] = (df['close'] - df['low_20']) / (df['high_20'] - df['low_20'] + 1e-10)

        # Gaps
        df['gap'] = (df['open'] - df['close'].shift()) / (df['close'].shift() + 1e-10)

        # VWAP
        df['vwap'] = (df['close'] * df['volume']).rolling(20, min_periods=1).sum() / (df['volume'].rolling(20, min_periods=1).sum() + 1e-10)
        df['price_vs_vwap'] = (df['close'] - df['vwap']) / (df['vwap'] + 1e-10)

        return df

    @staticmethod
    def get_momentum_features(df: pd.DataFrame) -> List[str]:
        """Features for Momentum strategy."""
        return [
            'returns_5', 'returns_10', 'returns_20',
            'rsi', 'macd_hist', 'volume_ratio',
            'price_position', 'volatility_ratio'
        ]

    @staticmethod
    def get_mean_reversion_features(df: pd.DataFrame) -> List[str]:
        """Features for Mean Reversion strategy."""
        return [
            'rsi', 'bb_position', 'price_vs_vwap',
            'returns_5', 'volatility', 'volume_ratio',
            'atr_pct', 'price_position'
        ]

    @staticmethod
    def get_breakout_features(df: pd.DataFrame) -> List[str]:
        """Features for Breakout strategy."""
        df['high_low_range'] = (df['high'] - df['low']) / (df['close'] + 1e-10)
        df['consolidation'] = df['bb_width'].rolling(10, min_periods=1).mean() / (df['bb_width'] + 1e-10)
        return [
            'price_position', 'bb_width', 'volume_ratio',
            'atr_pct', 'volatility_ratio', 'returns_2',
            'high_low_range', 'consolidation'
        ]

    @staticmethod
    def get_trend_following_features(df: pd.DataFrame) -> List[str]:
        """Features for Trend Following strategy."""
        df['sma_5_20_cross'] = (df['sma_5'] - df['sma_20']) / (df['close'] + 1e-10)
        df['sma_10_50_cross'] = (df['sma_10'] - df['sma_50']) / (df['close'] + 1e-10)
        df['sma_20_100_cross'] = (df['sma_20'] - df['sma_100']) / (df['close'] + 1e-10)
        df['ema_trend'] = (df['ema_10'] - df['ema_50']) / (df['close'] + 1e-10)
        df['trend_strength'] = abs(df['returns_20']) / (df['volatility'] + 1e-10)
        return [
            'sma_5_20_cross', 'sma_10_50_cross', 'sma_20_100_cross',
            'ema_trend', 'macd', 'macd_signal',
            'trend_strength', 'volume_ratio'
        ]

    @staticmethod
    def get_vwap_features(df: pd.DataFrame) -> List[str]:
        """Features for VWAP strategy."""
        df['vwap_distance'] = (df['close'] - df['vwap']) / (df['atr'] + 1e-10)
        return [
            'price_vs_vwap', 'vwap_distance', 'volume_ratio',
            'returns', 'rsi', 'atr_pct',
            'price_position', 'volatility'
        ]

    @staticmethod
    def get_gap_features(df: pd.DataFrame) -> List[str]:
        """Features for Gap Trading strategy."""
        df['gap_size'] = abs(df['gap'])
        df['gap_direction'] = np.sign(df['gap'])
        df['gap_fill_potential'] = df['gap_size'] * (1 - df['price_position'])
        df['prev_close_distance'] = (df['close'] - df['close'].shift()) / (df['atr'] + 1e-10)
        return [
            'gap', 'gap_size', 'gap_direction',
            'gap_fill_potential', 'volume_ratio', 'rsi',
            'bb_position', 'prev_close_distance'
        ]

    @staticmethod
    def get_multi_indicator_features(df: pd.DataFrame) -> List[str]:
        """Features for Multi-Indicator Confluence strategy."""
        df['rsi_bullish'] = (df['rsi'] < 40).astype(int)
        df['rsi_bearish'] = (df['rsi'] > 60).astype(int)
        df['macd_bullish'] = (df['macd_hist'] > 0).astype(int)
        df['bb_bullish'] = (df['bb_position'] < 0.3).astype(int)
        df['trend_bullish'] = (df['sma_20'] > df['sma_50']).astype(int)
        df['volume_confirm'] = (df['volume_ratio'] > 1.2).astype(int)

        df['bullish_count'] = df['rsi_bullish'] + df['macd_bullish'] + df['bb_bullish'] + df['trend_bullish']
        df['bearish_count'] = df['rsi_bearish'] + (1-df['macd_bullish']) + (1-df['bb_bullish']) + (1-df['trend_bullish'])
        df['signal_confluence'] = df['bullish_count'] - df['bearish_count']

        return [
            'signal_confluence', 'bullish_count', 'bearish_count',
            'volume_confirm', 'rsi', 'macd_hist',
            'bb_position', 'price_position'
        ]

    @staticmethod
    def get_rsi_divergence_features(df: pd.DataFrame) -> List[str]:
        """Features for RSI Divergence strategy."""
        # Fast trend calculation using simple diff
        df['price_trend'] = df['close'].diff(10) / (df['close'].shift(10) + 1e-10)
        df['rsi_trend'] = df['rsi'].diff(10)
        df['divergence'] = (np.sign(df['price_trend']) != np.sign(df['rsi_trend'])).astype(int)
        df['rsi_extreme'] = ((df['rsi'] > 70) | (df['rsi'] < 30)).astype(int)
        return [
            'rsi', 'rsi_trend', 'price_trend',
            'divergence', 'rsi_extreme', 'volume_ratio',
            'returns_5', 'bb_position'
        ]

    @staticmethod
    def get_bollinger_features(df: pd.DataFrame) -> List[str]:
        """Features for Bollinger Bands strategy."""
        bb_width_min = df['bb_width'].rolling(50, min_periods=1).min()
        bb_width_range = df['bb_width'].rolling(50, min_periods=1).max() - bb_width_min
        df['bb_squeeze'] = ((df['bb_width'] - bb_width_min) / (bb_width_range + 1e-10) < 0.2).astype(int)
        df['bb_breakout_up'] = (df['close'] > df['bb_upper']).astype(int)
        df['bb_breakout_down'] = (df['close'] < df['bb_lower']).astype(int)
        df['bb_mean_revert'] = abs(df['bb_position'] - 0.5)
        return [
            'bb_position', 'bb_width', 'bb_squeeze',
            'bb_breakout_up', 'bb_breakout_down', 'bb_mean_revert',
            'volatility_ratio', 'volume_ratio'
        ]

    @staticmethod
    def get_volume_spike_features(df: pd.DataFrame) -> List[str]:
        """Features for Volume Spike strategy."""
        df['volume_spike'] = (df['volume_ratio'] > 2.0).astype(int)
        df['volume_trend'] = df['volume'].rolling(5, min_periods=1).mean() / (df['volume'].rolling(20, min_periods=1).mean() + 1e-10)
        df['price_volume_corr'] = np.sign(df['returns']) * (df['volume_ratio'] - 1)
        df['unusual_volume'] = (df['volume_ratio'] - 1) * np.sign(df['returns'])
        return [
            'volume_ratio', 'volume_spike', 'volume_trend',
            'price_volume_corr', 'unusual_volume', 'returns',
            'atr_pct', 'rsi'
        ]


class MultiStrategyPredictor:
    """
    Uses trained XGBoost models to generate trading signals.

    21 models across 7 strategies x 3 selectivity levels:
    - Momentum, MeanReversion, Breakout, TrendFollowing, GapTrading, VWAP, MultiIndicator
    - Selective (94%+ win rate), Moderate (80-90%), Aggressive (70-80%)
    """

    # Model path
    MODEL_PATH = Path("C:/Users/steve/Desktop/Stocks 7.1/Trade 6.0/models/multi_strategy_30_system.pkl")

    # Strategy to feature method mapping
    STRATEGY_FEATURES = {
        'Momentum': StrategyFeatures.get_momentum_features,
        'MeanReversion': StrategyFeatures.get_mean_reversion_features,
        'Breakout': StrategyFeatures.get_breakout_features,
        'TrendFollowing': StrategyFeatures.get_trend_following_features,
        'VWAP': StrategyFeatures.get_vwap_features,
        'GapTrading': StrategyFeatures.get_gap_features,
        'MultiIndicator': StrategyFeatures.get_multi_indicator_features,
        'RSIDivergence': StrategyFeatures.get_rsi_divergence_features,
        'BollingerBands': StrategyFeatures.get_bollinger_features,
        'VolumeSpike': StrategyFeatures.get_volume_spike_features,
    }

    # Selectivity thresholds (matches training)
    SELECTIVITY = {
        'Selective': {'threshold': 0.80, 'min_confidence': 0.75},
        'Moderate': {'threshold': 0.65, 'min_confidence': 0.60},
        'Aggressive': {'threshold': 0.50, 'min_confidence': 0.45},
    }

    def __init__(self):
        """Initialize the predictor."""
        self.models = {}
        self.model_info = {}
        self.loaded = False
        self.load_models()

    def load_models(self) -> bool:
        """Load trained models from disk."""
        try:
            if not self.MODEL_PATH.exists():
                print(f"[MULTI-STRATEGY] Model file not found: {self.MODEL_PATH}")
                return False

            with open(self.MODEL_PATH, 'rb') as f:
                data = pickle.load(f)

            # Models are stored under 'trained_models' key
            trained_models = data.get('trained_models', {})

            # Build models dict with required structure
            for name, model_data in trained_models.items():
                self.models[name] = {
                    'model': model_data['model'],
                    'scaler': model_data['scaler'],
                    'features': model_data['features'],
                    'strategy': model_data['config']['strategy']
                }
                # Win rate is stored as decimal (0.948 = 94.8%), convert to percentage
                win_rate_decimal = float(model_data['win_rate'])
                win_rate_pct = win_rate_decimal * 100 if win_rate_decimal < 1 else win_rate_decimal

                self.model_info[name] = {
                    'strategy': model_data['config']['strategy'],
                    'win_rate': win_rate_pct,
                    'total_trades': model_data['total_trades'],
                    'simulations': model_data['simulations']
                }

            print(f"[MULTI-STRATEGY] Loaded {len(self.models)} trained models")
            for name, info in sorted(self.model_info.items(), key=lambda x: x[1]['win_rate'], reverse=True)[:5]:
                win_rate = info.get('win_rate', 0)
                trades = info.get('total_trades', 0)
                print(f"  - {name}: {win_rate:.1f}% win rate ({trades:,} trades)")

            self.loaded = True
            return True

        except Exception as e:
            print(f"[MULTI-STRATEGY] Error loading models: {e}")
            import traceback
            traceback.print_exc()
            return False

    def prepare_features(self, df: pd.DataFrame, strategy: str) -> Tuple[pd.DataFrame, List[str]]:
        """Prepare features for a specific strategy."""
        # Compute base features
        df = StrategyFeatures.compute_base_features(df)

        # Get strategy-specific features
        if strategy in self.STRATEGY_FEATURES:
            features = self.STRATEGY_FEATURES[strategy](df)
        else:
            features = StrategyFeatures.get_momentum_features(df)

        return df, features

    def predict_single_model(self, df: pd.DataFrame, model_name: str) -> Optional[Dict]:
        """
        Get prediction from a single model.

        Returns:
            {
                'model': model_name,
                'signal': 'BUY'|'SELL'|'HOLD',
                'probability': float,
                'confidence': float,
                'win_rate': float
            }
        """
        if model_name not in self.models:
            return None

        try:
            model_data = self.models[model_name]
            model = model_data['model']
            scaler = model_data['scaler']
            strategy = model_data['strategy']
            feature_names = model_data['features']  # Use stored feature names

            # Prepare features - compute all base + strategy-specific features
            df_feat, _ = self.prepare_features(df.copy(), strategy)

            # Get latest row
            latest = df_feat.iloc[-1:]

            # Extract feature values - use only the features the model was trained on
            missing_features = [f for f in feature_names if f not in latest.columns]
            if missing_features:
                print(f"[MULTI-STRATEGY] Missing features for {model_name}: {missing_features}")
                return None

            X = latest[feature_names].values

            # Handle NaN
            X = np.nan_to_num(X, nan=0.0)

            # Scale features
            X_scaled = scaler.transform(X)

            # Get prediction probabilities
            proba = model.predict_proba(X_scaled)[0]

            # Probability of winning trade
            win_prob = proba[1] if len(proba) > 1 else proba[0]

            # Get selectivity level from model name
            level = model_name.split('_')[-1]  # 'Selective', 'Moderate', 'Aggressive'
            threshold = self.SELECTIVITY.get(level, {}).get('threshold', 0.65)

            # Determine signal
            if win_prob >= threshold:
                signal = 'BUY'
            elif win_prob <= (1 - threshold):
                signal = 'SELL'
            else:
                signal = 'HOLD'

            # Get historical win rate
            model_win_rate = self.model_info.get(model_name, {}).get('win_rate', 0)

            return {
                'model': model_name,
                'strategy': strategy,
                'level': level,
                'signal': signal,
                'probability': round(float(win_prob), 4),
                'threshold': threshold,
                'confidence': round(float(abs(win_prob - 0.5) * 2), 4),
                'win_rate': model_win_rate
            }

        except Exception as e:
            print(f"[MULTI-STRATEGY] Error predicting {model_name}: {e}")
            return None

    def predict_all(self, df: pd.DataFrame) -> Dict:
        """
        Get predictions from all available models.

        Returns:
            {
                'predictions': [list of model predictions],
                'best_signal': strongest signal,
                'consensus': overall market direction,
                'buy_count': number of buy signals,
                'sell_count': number of sell signals,
                'hold_count': number of hold signals
            }
        """
        if not self.loaded or len(self.models) == 0:
            return {
                'predictions': [],
                'best_signal': None,
                'consensus': 'NEUTRAL',
                'buy_count': 0,
                'sell_count': 0,
                'hold_count': 0,
                'error': 'Models not loaded'
            }

        predictions = []

        for model_name in self.models.keys():
            pred = self.predict_single_model(df, model_name)
            if pred:
                predictions.append(pred)

        # Sort by confidence * win_rate (most reliable signals first)
        predictions.sort(
            key=lambda x: x['confidence'] * (x['win_rate'] / 100),
            reverse=True
        )

        # Count signals
        buy_count = sum(1 for p in predictions if p['signal'] == 'BUY')
        sell_count = sum(1 for p in predictions if p['signal'] == 'SELL')
        hold_count = sum(1 for p in predictions if p['signal'] == 'HOLD')

        # Determine consensus
        if buy_count > sell_count * 2:
            consensus = 'BULLISH'
        elif sell_count > buy_count * 2:
            consensus = 'BEARISH'
        else:
            consensus = 'NEUTRAL'

        # Best signal is the highest confidence BUY or SELL
        best_signal = None
        for pred in predictions:
            if pred['signal'] in ['BUY', 'SELL']:
                best_signal = pred
                break

        return {
            'predictions': predictions,
            'best_signal': best_signal,
            'consensus': consensus,
            'buy_count': buy_count,
            'sell_count': sell_count,
            'hold_count': hold_count,
            'total_models': len(predictions)
        }

    def predict_by_strategy(self, df: pd.DataFrame, strategy: str) -> Dict:
        """
        Get predictions for a specific strategy (all selectivity levels).

        Args:
            df: OHLCV dataframe
            strategy: 'Momentum', 'MeanReversion', etc.
        """
        predictions = []

        for level in ['Selective', 'Moderate', 'Aggressive']:
            model_name = f"{strategy}_{level}"
            pred = self.predict_single_model(df, model_name)
            if pred:
                predictions.append(pred)

        return {
            'strategy': strategy,
            'predictions': predictions,
            'best': predictions[0] if predictions else None
        }

    def get_selective_signals(self, df: pd.DataFrame) -> List[Dict]:
        """Get only Selective (highest win rate) model signals."""
        signals = []

        for model_name in self.models.keys():
            if '_Selective' in model_name:
                pred = self.predict_single_model(df, model_name)
                if pred and pred['signal'] != 'HOLD':
                    signals.append(pred)

        # Sort by win rate
        signals.sort(key=lambda x: x['win_rate'], reverse=True)
        return signals

    def get_trading_recommendation(self, df: pd.DataFrame) -> Dict:
        """
        Get a final trading recommendation based on all models.

        Returns a simplified recommendation suitable for trading decisions.
        """
        all_preds = self.predict_all(df)

        if not all_preds['predictions']:
            return {
                'action': 'HOLD',
                'confidence': 0,
                'reason': 'No model predictions available'
            }

        # Get selective signals (highest quality)
        selective_preds = [p for p in all_preds['predictions'] if p['level'] == 'Selective']
        selective_buys = [p for p in selective_preds if p['signal'] == 'BUY']
        selective_sells = [p for p in selective_preds if p['signal'] == 'SELL']

        # If multiple selective models agree, strong signal
        if len(selective_buys) >= 3:
            avg_prob = sum(p['probability'] for p in selective_buys) / len(selective_buys)
            avg_win = sum(p['win_rate'] for p in selective_buys) / len(selective_buys)
            return {
                'action': 'STRONG_BUY',
                'confidence': round(avg_prob * 100, 1),
                'win_rate': round(avg_win, 1),
                'reason': f"{len(selective_buys)} high-accuracy models agree on BUY",
                'agreeing_models': [p['model'] for p in selective_buys]
            }

        if len(selective_sells) >= 3:
            avg_prob = sum(1 - p['probability'] for p in selective_sells) / len(selective_sells)
            avg_win = sum(p['win_rate'] for p in selective_sells) / len(selective_sells)
            return {
                'action': 'STRONG_SELL',
                'confidence': round(avg_prob * 100, 1),
                'win_rate': round(avg_win, 1),
                'reason': f"{len(selective_sells)} high-accuracy models agree on SELL",
                'agreeing_models': [p['model'] for p in selective_sells]
            }

        # Single best signal
        best = all_preds['best_signal']
        if best:
            return {
                'action': best['signal'],
                'confidence': round(best['probability'] * 100, 1),
                'win_rate': best['win_rate'],
                'reason': f"Best signal from {best['model']}",
                'agreeing_models': [best['model']]
            }

        return {
            'action': 'HOLD',
            'confidence': 50,
            'reason': 'No clear signal from models'
        }

    def get_model_stats(self) -> Dict:
        """Get statistics about loaded models."""
        if not self.loaded:
            return {'loaded': False, 'models': 0}

        strategies = {}
        for name, info in self.model_info.items():
            strategy = info.get('strategy', 'Unknown')
            if strategy not in strategies:
                strategies[strategy] = []
            strategies[strategy].append({
                'name': name,
                'win_rate': info.get('win_rate', 0),
                'trades': info.get('total_trades', 0)
            })

        return {
            'loaded': True,
            'total_models': len(self.models),
            'strategies': strategies,
            'best_model': max(self.model_info.items(),
                            key=lambda x: x[1].get('win_rate', 0))[0] if self.model_info else None,
            'avg_win_rate': sum(m.get('win_rate', 0) for m in self.model_info.values()) / len(self.model_info) if self.model_info else 0
        }


# Singleton instance
_predictor: Optional[MultiStrategyPredictor] = None

def get_multi_strategy_predictor() -> MultiStrategyPredictor:
    """Get singleton predictor instance."""
    global _predictor
    if _predictor is None:
        _predictor = MultiStrategyPredictor()
    return _predictor


if __name__ == "__main__":
    # Test the predictor
    print("=== Multi-Strategy Predictor Test ===\n")

    predictor = get_multi_strategy_predictor()
    stats = predictor.get_model_stats()

    print(f"Models Loaded: {stats['total_models']}")
    print(f"Average Win Rate: {stats['avg_win_rate']:.1f}%")
    print(f"Best Model: {stats['best_model']}")

    print("\nStrategies:")
    for strategy, models in stats.get('strategies', {}).items():
        print(f"\n  {strategy}:")
        for m in models:
            print(f"    - {m['name']}: {m['win_rate']:.1f}% ({m['trades']:,} trades)")
