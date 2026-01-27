"""
AI Module - 30-Model Trading Brain Integration
"""

import sys
from pathlib import Path

# Add parent 30Bots directory to path for importing the AI brain
PARENT_DIR = Path(__file__).parent.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

# Import from the main 30Bots AI brain
try:
    from ai_30model_brain import (
        AIModelBrain,
        MODEL_RISK_PARAMS,
        DEFAULT_RISK_PARAMS,
        get_model_risk_params,
        calculate_risk_prices,
        CHECKPOINT_PATH,
        FINAL_MODEL_PATH
    )
    AI_BRAIN_AVAILABLE = True
except ImportError as e:
    print(f"AI Brain not available: {e}")
    AI_BRAIN_AVAILABLE = False
    AIModelBrain = None
    MODEL_RISK_PARAMS = {}
    DEFAULT_RISK_PARAMS = {}
    get_model_risk_params = None
    calculate_risk_prices = None

from .prediction_worker import AIPredictionWorker, get_ai_brain

__all__ = [
    'AIModelBrain',
    'MODEL_RISK_PARAMS',
    'DEFAULT_RISK_PARAMS',
    'get_model_risk_params',
    'calculate_risk_prices',
    'AIPredictionWorker',
    'get_ai_brain',
    'AI_BRAIN_AVAILABLE'
]
