"""
Gemini AI Assistant
Provides intelligent assistance for trading analysis, debugging, and insights
"""

import os
import sys
import json
import time
import requests
from pathlib import Path
from datetime import datetime
from threading import Lock
from typing import Optional, Dict, List, Any

# Add parent for imports
PARENT_DIR = Path(__file__).parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

try:
    from config import GEMINI_API_KEY
except ImportError:
    GEMINI_API_KEY = os.environ.get("GEMINI_API_KEY", "")


class RateLimiter:
    """Rate limiter to prevent hitting API limits (15 RPM for Gemini free tier)."""

    def __init__(self, requests_per_minute: int = 12):
        self.min_interval = 60.0 / requests_per_minute  # ~5 seconds between requests
        self.last_request_time = 0
        self.lock = Lock()
        self.consecutive_errors = 0
        self.max_retries = 3

    def wait(self):
        """Wait if necessary to respect rate limits."""
        with self.lock:
            now = time.time()
            elapsed = now - self.last_request_time
            if elapsed < self.min_interval:
                sleep_time = self.min_interval - elapsed
                time.sleep(sleep_time)
            self.last_request_time = time.time()

    def backoff_wait(self):
        """Exponential backoff after errors."""
        wait_time = min(60, (2 ** self.consecutive_errors) * 2)
        self.consecutive_errors += 1
        time.sleep(wait_time)
        return wait_time

    def reset_errors(self):
        """Reset error count after successful request."""
        self.consecutive_errors = 0


# Global rate limiter (shared across assistant instances)
_rate_limiter = RateLimiter()


class GeminiAssistant:
    """
    AI Assistant powered by Google's Gemini model.
    Provides trading analysis, debugging help, and intelligent responses.
    """

    API_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

    SYSTEM_PROMPT = """You are an AI trading assistant for the 30Bots trading application.
You help users understand their trading performance, debug issues, and make informed decisions.

You have access to:
- Current portfolio positions and P&L
- Recent trade history
- AI model performance data
- Application error logs

Guidelines:
- Be concise and actionable in your responses
- Use specific numbers and data when available
- Highlight risks and opportunities
- Suggest concrete next steps when appropriate
- Format currency as $X,XXX.XX
- Format percentages as X.XX%

When analyzing trades:
- Consider entry/exit timing
- Evaluate stop loss and take profit levels
- Compare actual vs expected performance
- Identify patterns in winning/losing trades

When debugging:
- Identify the root cause
- Suggest specific fixes
- Explain why the issue occurred
"""

    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        self.conversation_history: List[Dict] = []
        self.trading_context: Dict = {}
        self.max_history = 10  # Keep last 10 exchanges

    def set_trading_context(self, context: Dict):
        """Set the current trading context for the assistant."""
        self.trading_context = context

    def _build_context_string(self) -> str:
        """Build a context string from trading data."""
        if not self.trading_context:
            return ""

        parts = []

        # Account info
        if 'account' in self.trading_context:
            acc = self.trading_context['account']
            parts.append(f"Account: Equity=${acc.get('equity', 0):,.2f}, "
                        f"Cash=${acc.get('cash', 0):,.2f}, "
                        f"Day P&L=${acc.get('day_pnl', 0):+,.2f}")

        # Positions
        if 'positions' in self.trading_context and self.trading_context['positions']:
            pos_str = "Current Positions:\n"
            for pos in self.trading_context['positions']:
                pos_str += f"  - {pos['symbol']}: {pos['qty']} shares, "
                pos_str += f"Entry=${pos['entry']:,.2f}, "
                pos_str += f"Current=${pos['current']:,.2f}, "
                pos_str += f"P&L=${pos['pnl']:+,.2f} ({pos['pnl_pct']:+.2f}%)\n"
            parts.append(pos_str)

        # Recent trades
        if 'recent_trades' in self.trading_context and self.trading_context['recent_trades']:
            trades_str = "Recent Trades:\n"
            for trade in self.trading_context['recent_trades'][:5]:
                trades_str += f"  - {trade['symbol']} {trade['action']} "
                trades_str += f"@ ${trade['price']:,.2f}, "
                trades_str += f"P&L=${trade.get('pnl', 0):+,.2f}\n"
            parts.append(trades_str)

        # Model performance
        if 'model_performance' in self.trading_context:
            perf = self.trading_context['model_performance']
            parts.append(f"Top AI Models: {perf}")

        # Errors
        if 'recent_errors' in self.trading_context and self.trading_context['recent_errors']:
            parts.append(f"Recent Errors: {', '.join(self.trading_context['recent_errors'][:3])}")

        return "\n".join(parts)

    def query(self, user_message: str) -> str:
        """
        Send a query to Gemini and get a response.

        Args:
            user_message: The user's question or request

        Returns:
            The assistant's response
        """
        if not self.api_key:
            return "Error: Gemini API key not configured. Please set GEMINI_API_KEY in your environment."

        # Build the full prompt with context
        context = self._build_context_string()

        # Build conversation history string
        history_str = ""
        for entry in self.conversation_history[-self.max_history:]:
            history_str += f"User: {entry['user']}\nAssistant: {entry['assistant']}\n\n"

        full_prompt = f"""{self.SYSTEM_PROMPT}

Current Trading Context:
{context if context else "No trading context available."}

Previous Conversation:
{history_str if history_str else "This is the start of the conversation."}

User: {user_message}

Please provide a helpful, concise response."""

        # Make API request with retry logic
        headers = {"Content-Type": "application/json"}
        payload = {
            "contents": [{"parts": [{"text": full_prompt}]}],
            "generationConfig": {
                "temperature": 0.7,
                "maxOutputTokens": 1024,
            }
        }

        for attempt in range(_rate_limiter.max_retries):
            try:
                # Wait to respect rate limits
                _rate_limiter.wait()

                response = requests.post(
                    f"{self.API_URL}?key={self.api_key}",
                    headers=headers,
                    json=payload,
                    timeout=30
                )

                # Handle rate limiting (429)
                if response.status_code == 429:
                    wait_time = _rate_limiter.backoff_wait()
                    if attempt < _rate_limiter.max_retries - 1:
                        continue
                    return f"Rate limited. Please wait {wait_time}s and try again."

                if response.status_code == 200:
                    data = response.json()
                    assistant_response = data['candidates'][0]['content']['parts'][0]['text']

                    # Success - reset error counter
                    _rate_limiter.reset_errors()

                    # Store in history
                    self.conversation_history.append({
                        'user': user_message,
                        'assistant': assistant_response,
                        'timestamp': datetime.now().isoformat()
                    })

                    # Trim history if too long
                    if len(self.conversation_history) > self.max_history:
                        self.conversation_history = self.conversation_history[-self.max_history:]

                    return assistant_response
                else:
                    return f"API Error ({response.status_code}): {response.text[:200]}"

            except requests.exceptions.Timeout:
                return "Error: Request timed out. Please try again."
            except requests.exceptions.RequestException as e:
                if "429" in str(e):
                    _rate_limiter.backoff_wait()
                    continue
                return f"Network Error: {str(e)}"
            except Exception as e:
                return f"Error: {str(e)}"

        return "Error: Max retries reached. Please try again later."

    def analyze_positions(self) -> str:
        """Analyze current positions and provide insights."""
        return self.query("Analyze my current positions. What's performing well and what should I watch out for?")

    def debug_errors(self, error_log: str) -> str:
        """Help debug recent errors."""
        # Add error log to context temporarily
        self.trading_context['recent_errors'] = [error_log]
        response = self.query(f"Help me debug this error: {error_log}")
        return response

    def get_trading_summary(self) -> str:
        """Generate a trading summary."""
        return self.query("Give me a brief summary of my trading activity today. Include key metrics and notable events.")

    def suggest_improvements(self) -> str:
        """Suggest trading improvements based on history."""
        return self.query("Based on my recent trading history, what improvements would you suggest?")

    def explain_trade(self, symbol: str) -> str:
        """Explain why a specific trade was made or closed."""
        return self.query(f"Explain the recent trading activity for {symbol}. Why was the position opened/closed?")

    def clear_history(self):
        """Clear conversation history."""
        self.conversation_history = []


# Singleton instance
_assistant_instance = None


def get_assistant() -> GeminiAssistant:
    """Get or create the global assistant instance."""
    global _assistant_instance
    if _assistant_instance is None:
        _assistant_instance = GeminiAssistant()
    return _assistant_instance