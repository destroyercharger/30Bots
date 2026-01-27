"""
News Sentiment Analyzer - Uses Google Gemini for sentiment analysis
"""

import json
import time
import requests
from typing import Dict, List, Optional
from datetime import datetime
from threading import Lock

from config import GEMINI_API_KEY


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
        wait_time = min(60, (2 ** self.consecutive_errors) * 2)  # 2, 4, 8, 16... max 60s
        print(f"[Sentiment] Rate limited, waiting {wait_time}s...")
        time.sleep(wait_time)
        self.consecutive_errors += 1

    def reset_errors(self):
        """Reset error count after successful request."""
        self.consecutive_errors = 0


# Global rate limiter instance
_rate_limiter = RateLimiter()


class GeminiSentimentAnalyzer:
    """Analyzes news sentiment using Google Gemini API."""

    BASE_URL = "https://generativelanguage.googleapis.com/v1beta/models/gemini-2.0-flash-lite:generateContent"

    def __init__(self, api_key: str = None):
        self.api_key = api_key or GEMINI_API_KEY
        if not self.api_key:
            print("[Sentiment] Warning: No Gemini API key configured")

    def analyze_article(self, headline: str, summary: str = "", symbols: List[str] = None) -> Dict:
        """
        Analyze sentiment of a news article.

        Returns:
            Dict with sentiment, confidence, affected_symbols, trading_signal
        """
        if not self.api_key:
            return self._default_response()

        symbols_str = ", ".join(symbols) if symbols else "general market"

        prompt = f"""Analyze this financial news for trading sentiment.

Headline: {headline}
Summary: {summary}
Related Symbols: {symbols_str}

Respond in JSON format only:
{{
    "sentiment": "bullish" | "bearish" | "neutral",
    "confidence": 0.0-1.0,
    "magnitude": "high" | "medium" | "low",
    "affected_symbols": ["SYM1", "SYM2"],
    "trading_signal": "strong_buy" | "buy" | "hold" | "sell" | "strong_sell",
    "reasoning": "brief explanation",
    "price_impact": "positive" | "negative" | "neutral",
    "urgency": "immediate" | "short_term" | "long_term"
}}"""

        # Retry loop with rate limiting
        for attempt in range(_rate_limiter.max_retries):
            try:
                # Wait to respect rate limits
                _rate_limiter.wait()

                response = requests.post(
                    f"{self.BASE_URL}?key={self.api_key}",
                    headers={"Content-Type": "application/json"},
                    json={
                        "contents": [{"parts": [{"text": prompt}]}],
                        "generationConfig": {
                            "temperature": 0.1,
                            "maxOutputTokens": 500
                        }
                    },
                    timeout=15
                )

                # Handle rate limiting (429)
                if response.status_code == 429:
                    print(f"[Sentiment] Rate limited (attempt {attempt + 1}/{_rate_limiter.max_retries})")
                    _rate_limiter.backoff_wait()
                    continue

                response.raise_for_status()
                data = response.json()

                # Success - reset error counter
                _rate_limiter.reset_errors()

                # Extract text from response
                text = data.get("candidates", [{}])[0].get("content", {}).get("parts", [{}])[0].get("text", "")

                # Parse JSON from response
                # Find JSON in the response (handle markdown code blocks)
                if "```json" in text:
                    text = text.split("```json")[1].split("```")[0]
                elif "```" in text:
                    text = text.split("```")[1].split("```")[0]

                result = json.loads(text.strip())

                # Add metadata
                result["headline"] = headline
                result["analyzed_at"] = datetime.now().isoformat()
                result["source_symbols"] = symbols or []

                return result

            except json.JSONDecodeError as e:
                print(f"[Sentiment] JSON parse error: {e}")
                return self._default_response(headline, symbols)
            except requests.exceptions.RequestException as e:
                if "429" in str(e):
                    print(f"[Sentiment] Rate limited (attempt {attempt + 1})")
                    _rate_limiter.backoff_wait()
                    continue
                print(f"[Sentiment] API error: {e}")
                return self._default_response(headline, symbols)
            except Exception as e:
                print(f"[Sentiment] Unexpected error: {e}")
                return self._default_response(headline, symbols)

        # All retries exhausted
        print("[Sentiment] Max retries reached, returning default response")
        return self._default_response(headline, symbols)

    def _default_response(self, headline: str = "", symbols: List[str] = None) -> Dict:
        """Return default neutral response."""
        return {
            "sentiment": "neutral",
            "confidence": 0.5,
            "magnitude": "low",
            "affected_symbols": symbols or [],
            "trading_signal": "hold",
            "reasoning": "Unable to analyze - using default neutral",
            "price_impact": "neutral",
            "urgency": "long_term",
            "headline": headline,
            "analyzed_at": datetime.now().isoformat()
        }

    def analyze_batch(self, articles: List[Dict]) -> List[Dict]:
        """Analyze multiple articles and return sentiment for each."""
        results = []
        for article in articles:
            sentiment = self.analyze_article(
                headline=article.get("headline", ""),
                summary=article.get("summary", ""),
                symbols=article.get("symbols", [])
            )
            sentiment["article_id"] = article.get("id")
            sentiment["source"] = article.get("source")
            sentiment["url"] = article.get("url")
            results.append(sentiment)
        return results

    def get_symbol_sentiment(self, articles: List[Dict], symbol: str) -> Dict:
        """
        Get aggregated sentiment for a specific symbol from multiple articles.
        """
        symbol_articles = [a for a in articles if symbol in a.get("symbols", [])]

        if not symbol_articles:
            return {
                "symbol": symbol,
                "overall_sentiment": "neutral",
                "confidence": 0.0,
                "article_count": 0,
                "bullish_count": 0,
                "bearish_count": 0,
                "recommendation": "hold"
            }

        sentiments = self.analyze_batch(symbol_articles)

        bullish = sum(1 for s in sentiments if s["sentiment"] == "bullish")
        bearish = sum(1 for s in sentiments if s["sentiment"] == "bearish")
        total = len(sentiments)

        # Calculate aggregate
        if bullish > bearish:
            overall = "bullish"
            confidence = bullish / total
            recommendation = "buy" if confidence > 0.6 else "hold"
        elif bearish > bullish:
            overall = "bearish"
            confidence = bearish / total
            recommendation = "sell" if confidence > 0.6 else "hold"
        else:
            overall = "neutral"
            confidence = 0.5
            recommendation = "hold"

        return {
            "symbol": symbol,
            "overall_sentiment": overall,
            "confidence": confidence,
            "article_count": total,
            "bullish_count": bullish,
            "bearish_count": bearish,
            "neutral_count": total - bullish - bearish,
            "recommendation": recommendation,
            "analyzed_articles": sentiments
        }


# Singleton instance
_sentiment_analyzer = None


def get_sentiment_analyzer() -> GeminiSentimentAnalyzer:
    """Get singleton sentiment analyzer instance."""
    global _sentiment_analyzer
    if _sentiment_analyzer is None:
        _sentiment_analyzer = GeminiSentimentAnalyzer()
    return _sentiment_analyzer
