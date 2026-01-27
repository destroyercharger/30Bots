"""
News Trading Tab - Real-time news analysis and trading signals
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QTableWidget, QTableWidgetItem, QTextEdit, QGroupBox,
    QSplitter, QComboBox, QSpinBox, QHeaderView, QFrame
)
from PyQt6.QtCore import Qt, QTimer, pyqtSignal
from PyQt6.QtGui import QColor, QFont
from datetime import datetime
import sys
from pathlib import Path

# Add parent paths
sys.path.insert(0, str(Path(__file__).parent.parent.parent))

from data.news_fetcher import get_news_fetcher
from data.news_sentiment import get_sentiment_analyzer


class NewsTradingTab(QWidget):
    """News trading tab with sentiment analysis."""

    trade_signal = pyqtSignal(dict)  # Emit trade signals

    def __init__(self, broker=None, parent=None):
        super().__init__(parent)
        self.broker = broker
        self.news_fetcher = get_news_fetcher()
        self.sentiment_analyzer = get_sentiment_analyzer()
        self.current_articles = []
        self.analyzed_articles = []

        self.setup_ui()
        self.setup_timers()

    def setup_ui(self):
        """Setup the UI components."""
        layout = QVBoxLayout(self)
        layout.setSpacing(10)

        # Header
        header = self._create_header()
        layout.addWidget(header)

        # Main content splitter
        splitter = QSplitter(Qt.Orientation.Horizontal)

        # Left: News feed
        news_panel = self._create_news_panel()
        splitter.addWidget(news_panel)

        # Right: Sentiment analysis and signals
        analysis_panel = self._create_analysis_panel()
        splitter.addWidget(analysis_panel)

        splitter.setSizes([600, 400])
        layout.addWidget(splitter)

    def _create_header(self) -> QWidget:
        """Create header with controls."""
        frame = QFrame()
        frame.setStyleSheet("background-color: #1a1a2e; border-radius: 5px; padding: 10px;")
        layout = QHBoxLayout(frame)

        # Title
        title = QLabel("News Trading")
        title.setFont(QFont("Arial", 14, QFont.Weight.Bold))
        title.setStyleSheet("color: #00d4ff;")
        layout.addWidget(title)

        layout.addStretch()

        # Refresh interval
        layout.addWidget(QLabel("Refresh:"))
        self.refresh_combo = QComboBox()
        self.refresh_combo.addItems(["1 min", "5 min", "15 min", "Manual"])
        self.refresh_combo.setCurrentIndex(1)
        self.refresh_combo.currentIndexChanged.connect(self.on_refresh_changed)
        layout.addWidget(self.refresh_combo)

        # Symbol filter
        layout.addWidget(QLabel("Symbol:"))
        self.symbol_filter = QComboBox()
        self.symbol_filter.addItems(["All", "AAPL", "NVDA", "TSLA", "MSFT", "AMD", "GOOGL", "AMZN", "META"])
        self.symbol_filter.setEditable(True)
        self.symbol_filter.currentTextChanged.connect(self.on_filter_changed)
        layout.addWidget(self.symbol_filter)

        # Refresh button
        self.refresh_btn = QPushButton("Refresh News")
        self.refresh_btn.clicked.connect(self.refresh_news)
        self.refresh_btn.setStyleSheet("""
            QPushButton {
                background-color: #00d4ff;
                color: #000;
                padding: 8px 15px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00a8cc;
            }
        """)
        layout.addWidget(self.refresh_btn)

        # Auto-trade toggle
        self.auto_trade_btn = QPushButton("Auto-Trade: OFF")
        self.auto_trade_btn.setCheckable(True)
        self.auto_trade_btn.clicked.connect(self.toggle_auto_trade)
        self.auto_trade_btn.setStyleSheet("""
            QPushButton {
                background-color: #333;
                color: #fff;
                padding: 8px 15px;
                border-radius: 4px;
            }
            QPushButton:checked {
                background-color: #00c853;
                color: #000;
            }
        """)
        layout.addWidget(self.auto_trade_btn)

        return frame

    def _create_news_panel(self) -> QWidget:
        """Create news feed panel."""
        group = QGroupBox("Live News Feed")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #00d4ff;
            }
        """)
        layout = QVBoxLayout(group)

        # News table
        self.news_table = QTableWidget()
        self.news_table.setColumnCount(5)
        self.news_table.setHorizontalHeaderLabels(["Time", "Symbols", "Headline", "Sentiment", "Signal"])
        self.news_table.horizontalHeader().setSectionResizeMode(2, QHeaderView.ResizeMode.Stretch)
        self.news_table.setSelectionBehavior(QTableWidget.SelectionBehavior.SelectRows)
        self.news_table.itemSelectionChanged.connect(self.on_news_selected)
        self.news_table.setStyleSheet("""
            QTableWidget {
                background-color: #0d0d1a;
                color: #fff;
                gridline-color: #333;
            }
            QTableWidget::item:selected {
                background-color: #1a3a5c;
            }
            QHeaderView::section {
                background-color: #1a1a2e;
                color: #00d4ff;
                padding: 5px;
                border: 1px solid #333;
            }
        """)
        layout.addWidget(self.news_table)

        # Article preview
        self.article_preview = QTextEdit()
        self.article_preview.setReadOnly(True)
        self.article_preview.setMaximumHeight(150)
        self.article_preview.setStyleSheet("""
            QTextEdit {
                background-color: #0d0d1a;
                color: #ccc;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        self.article_preview.setPlaceholderText("Select an article to view details...")
        layout.addWidget(self.article_preview)

        return group

    def _create_analysis_panel(self) -> QWidget:
        """Create sentiment analysis panel."""
        group = QGroupBox("Sentiment Analysis")
        group.setStyleSheet("""
            QGroupBox {
                font-weight: bold;
                border: 1px solid #333;
                border-radius: 5px;
                margin-top: 10px;
                padding-top: 10px;
            }
            QGroupBox::title {
                color: #00d4ff;
            }
        """)
        layout = QVBoxLayout(group)

        # Overall sentiment
        sentiment_frame = QFrame()
        sentiment_frame.setStyleSheet("background-color: #1a1a2e; border-radius: 5px; padding: 15px;")
        sentiment_layout = QHBoxLayout(sentiment_frame)

        self.sentiment_label = QLabel("NEUTRAL")
        self.sentiment_label.setFont(QFont("Arial", 24, QFont.Weight.Bold))
        self.sentiment_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.sentiment_label.setStyleSheet("color: #888;")
        sentiment_layout.addWidget(self.sentiment_label)

        self.confidence_label = QLabel("0%")
        self.confidence_label.setFont(QFont("Arial", 16))
        self.confidence_label.setAlignment(Qt.AlignmentFlag.AlignCenter)
        self.confidence_label.setStyleSheet("color: #666;")
        sentiment_layout.addWidget(self.confidence_label)

        layout.addWidget(sentiment_frame)

        # Symbol sentiments table
        layout.addWidget(QLabel("Symbol Sentiments:"))
        self.symbol_table = QTableWidget()
        self.symbol_table.setColumnCount(4)
        self.symbol_table.setHorizontalHeaderLabels(["Symbol", "Sentiment", "Articles", "Action"])
        self.symbol_table.horizontalHeader().setSectionResizeMode(QHeaderView.ResizeMode.Stretch)
        self.symbol_table.setStyleSheet("""
            QTableWidget {
                background-color: #0d0d1a;
                color: #fff;
                gridline-color: #333;
            }
            QHeaderView::section {
                background-color: #1a1a2e;
                color: #00d4ff;
                padding: 5px;
                border: 1px solid #333;
            }
        """)
        layout.addWidget(self.symbol_table)

        # Trading signals
        layout.addWidget(QLabel("Recent Trading Signals:"))
        self.signals_log = QTextEdit()
        self.signals_log.setReadOnly(True)
        self.signals_log.setMaximumHeight(150)
        self.signals_log.setStyleSheet("""
            QTextEdit {
                background-color: #0d0d1a;
                color: #0f0;
                font-family: monospace;
                border: 1px solid #333;
                border-radius: 4px;
                padding: 10px;
            }
        """)
        layout.addWidget(self.signals_log)

        # Manual trade buttons
        btn_layout = QHBoxLayout()

        self.buy_btn = QPushButton("BUY Selected")
        self.buy_btn.clicked.connect(lambda: self.execute_manual_trade("BUY"))
        self.buy_btn.setStyleSheet("""
            QPushButton {
                background-color: #00c853;
                color: #000;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #00a843;
            }
        """)
        btn_layout.addWidget(self.buy_btn)

        self.sell_btn = QPushButton("SELL Selected")
        self.sell_btn.clicked.connect(lambda: self.execute_manual_trade("SELL"))
        self.sell_btn.setStyleSheet("""
            QPushButton {
                background-color: #ff1744;
                color: #fff;
                padding: 10px 20px;
                border-radius: 4px;
                font-weight: bold;
            }
            QPushButton:hover {
                background-color: #d50000;
            }
        """)
        btn_layout.addWidget(self.sell_btn)

        layout.addLayout(btn_layout)

        return group

    def setup_timers(self):
        """Setup refresh timers."""
        self.refresh_timer = QTimer()
        self.refresh_timer.timeout.connect(self.refresh_news)
        self.refresh_timer.start(300000)  # 5 minutes default

    def on_refresh_changed(self, index: int):
        """Handle refresh interval change."""
        intervals = [60000, 300000, 900000, 0]  # 1, 5, 15 min, manual
        interval = intervals[index]
        if interval > 0:
            self.refresh_timer.start(interval)
        else:
            self.refresh_timer.stop()

    def on_filter_changed(self, text: str):
        """Handle symbol filter change."""
        self.refresh_news()

    def toggle_auto_trade(self, checked: bool):
        """Toggle auto-trading based on news."""
        if checked:
            self.auto_trade_btn.setText("Auto-Trade: ON")
            self.log_signal("Auto-trading ENABLED - will execute on strong signals")
        else:
            self.auto_trade_btn.setText("Auto-Trade: OFF")
            self.log_signal("Auto-trading DISABLED")

    def refresh_news(self):
        """Fetch and analyze latest news."""
        self.refresh_btn.setEnabled(False)
        self.refresh_btn.setText("Loading...")

        try:
            # Get symbol filter
            symbol_filter = self.symbol_filter.currentText()
            if symbol_filter == "All":
                articles = self.news_fetcher.get_recent_news(hours=4, limit=30)
            else:
                articles = self.news_fetcher.get_symbol_news(symbol_filter, limit=20)

            self.current_articles = articles
            self.update_news_table(articles)

            # Analyze sentiment
            if articles:
                self.analyze_news(articles)

        except Exception as e:
            print(f"[News] Error refreshing: {e}")
            self.log_signal(f"Error: {e}")

        finally:
            self.refresh_btn.setEnabled(True)
            self.refresh_btn.setText("Refresh News")

    def update_news_table(self, articles: list):
        """Update news table with articles."""
        self.news_table.setRowCount(len(articles))

        for i, article in enumerate(articles):
            # Time
            created = article.get("created_at", "")
            if created:
                try:
                    dt = datetime.fromisoformat(created.replace("Z", "+00:00"))
                    time_str = dt.strftime("%H:%M")
                except:
                    time_str = created[:5]
            else:
                time_str = ""

            self.news_table.setItem(i, 0, QTableWidgetItem(time_str))

            # Symbols
            symbols = ", ".join(article.get("symbols", [])[:3])
            self.news_table.setItem(i, 1, QTableWidgetItem(symbols))

            # Headline
            headline = article.get("headline", "")[:80]
            self.news_table.setItem(i, 2, QTableWidgetItem(headline))

            # Sentiment (will be updated after analysis)
            self.news_table.setItem(i, 3, QTableWidgetItem("..."))

            # Signal (will be updated after analysis)
            self.news_table.setItem(i, 4, QTableWidgetItem("..."))

    def analyze_news(self, articles: list):
        """Analyze sentiment for all articles."""
        self.analyzed_articles = self.sentiment_analyzer.analyze_batch(articles[:10])  # Limit for API

        # Update table with sentiments
        for i, sentiment in enumerate(self.analyzed_articles):
            if i >= self.news_table.rowCount():
                break

            # Sentiment
            sent = sentiment.get("sentiment", "neutral")
            sent_item = QTableWidgetItem(sent.upper())
            if sent == "bullish":
                sent_item.setForeground(QColor("#00c853"))
            elif sent == "bearish":
                sent_item.setForeground(QColor("#ff1744"))
            else:
                sent_item.setForeground(QColor("#888"))
            self.news_table.setItem(i, 3, sent_item)

            # Signal
            signal = sentiment.get("trading_signal", "hold")
            signal_item = QTableWidgetItem(signal.upper())
            if "buy" in signal:
                signal_item.setForeground(QColor("#00c853"))
            elif "sell" in signal:
                signal_item.setForeground(QColor("#ff1744"))
            self.news_table.setItem(i, 4, signal_item)

            # Check for auto-trade signals
            if self.auto_trade_btn.isChecked():
                self.check_auto_trade(sentiment)

        # Update overall sentiment
        self.update_overall_sentiment()

        # Update symbol sentiments
        self.update_symbol_sentiments()

    def update_overall_sentiment(self):
        """Update overall market sentiment display."""
        if not self.analyzed_articles:
            return

        bullish = sum(1 for a in self.analyzed_articles if a.get("sentiment") == "bullish")
        bearish = sum(1 for a in self.analyzed_articles if a.get("sentiment") == "bearish")
        total = len(self.analyzed_articles)

        if bullish > bearish:
            self.sentiment_label.setText("BULLISH")
            self.sentiment_label.setStyleSheet("color: #00c853;")
            confidence = bullish / total * 100
        elif bearish > bullish:
            self.sentiment_label.setText("BEARISH")
            self.sentiment_label.setStyleSheet("color: #ff1744;")
            confidence = bearish / total * 100
        else:
            self.sentiment_label.setText("NEUTRAL")
            self.sentiment_label.setStyleSheet("color: #888;")
            confidence = 50

        self.confidence_label.setText(f"{confidence:.0f}%")

    def update_symbol_sentiments(self):
        """Update symbol sentiment table."""
        # Get unique symbols from articles
        all_symbols = set()
        for article in self.current_articles:
            all_symbols.update(article.get("symbols", []))

        self.symbol_table.setRowCount(len(all_symbols))

        for i, symbol in enumerate(sorted(all_symbols)):
            # Get sentiment for symbol
            symbol_sent = self.sentiment_analyzer.get_symbol_sentiment(
                self.current_articles, symbol
            )

            self.symbol_table.setItem(i, 0, QTableWidgetItem(symbol))

            sent = symbol_sent.get("overall_sentiment", "neutral")
            sent_item = QTableWidgetItem(sent.upper())
            if sent == "bullish":
                sent_item.setForeground(QColor("#00c853"))
            elif sent == "bearish":
                sent_item.setForeground(QColor("#ff1744"))
            self.symbol_table.setItem(i, 1, sent_item)

            self.symbol_table.setItem(i, 2, QTableWidgetItem(str(symbol_sent.get("article_count", 0))))

            action = symbol_sent.get("recommendation", "hold")
            action_item = QTableWidgetItem(action.upper())
            if action == "buy":
                action_item.setForeground(QColor("#00c853"))
            elif action == "sell":
                action_item.setForeground(QColor("#ff1744"))
            self.symbol_table.setItem(i, 3, action_item)

    def on_news_selected(self):
        """Handle news article selection."""
        row = self.news_table.currentRow()
        if row >= 0 and row < len(self.current_articles):
            article = self.current_articles[row]
            preview = f"<b>{article.get('headline', '')}</b><br><br>"
            preview += f"<i>Source: {article.get('source', 'Unknown')}</i><br>"
            preview += f"<i>Symbols: {', '.join(article.get('symbols', []))}</i><br><br>"
            preview += article.get('summary', article.get('content', '')[:500])
            self.article_preview.setHtml(preview)

    def check_auto_trade(self, sentiment: dict):
        """Check if sentiment warrants auto-trade."""
        signal = sentiment.get("trading_signal", "hold")
        confidence = sentiment.get("confidence", 0)
        symbols = sentiment.get("affected_symbols", [])
        urgency = sentiment.get("urgency", "long_term")

        # Only trade on strong signals with high confidence
        if confidence >= 0.7 and urgency in ("immediate", "short_term"):
            if signal in ("strong_buy", "buy") and symbols:
                for symbol in symbols[:1]:  # Only first symbol
                    self.log_signal(f"AUTO-TRADE: BUY {symbol} (confidence: {confidence*100:.0f}%)")
                    self.trade_signal.emit({
                        "action": "BUY",
                        "symbol": symbol,
                        "confidence": confidence,
                        "source": "news_sentiment",
                        "headline": sentiment.get("headline", "")
                    })

            elif signal in ("strong_sell", "sell") and symbols:
                for symbol in symbols[:1]:
                    self.log_signal(f"AUTO-TRADE: SELL {symbol} (confidence: {confidence*100:.0f}%)")
                    self.trade_signal.emit({
                        "action": "SELL",
                        "symbol": symbol,
                        "confidence": confidence,
                        "source": "news_sentiment",
                        "headline": sentiment.get("headline", "")
                    })

    def execute_manual_trade(self, action: str):
        """Execute manual trade for selected symbol."""
        row = self.symbol_table.currentRow()
        if row >= 0:
            symbol = self.symbol_table.item(row, 0).text()
            self.log_signal(f"MANUAL TRADE: {action} {symbol}")
            self.trade_signal.emit({
                "action": action,
                "symbol": symbol,
                "confidence": 0.8,
                "source": "manual_news"
            })

    def log_signal(self, message: str):
        """Log a signal message."""
        timestamp = datetime.now().strftime("%H:%M:%S")
        self.signals_log.append(f"[{timestamp}] {message}")
