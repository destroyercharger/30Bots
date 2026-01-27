"""
AI Terminal Widget
Dockable chat interface for the Gemini AI Assistant
"""

from PyQt6.QtWidgets import (
    QDockWidget, QWidget, QVBoxLayout, QHBoxLayout,
    QTextEdit, QLineEdit, QPushButton, QLabel, QFrame,
    QScrollArea, QSizePolicy
)
from PyQt6.QtCore import Qt, pyqtSignal, QThread, QTimer
from PyQt6.QtGui import QFont, QTextCursor, QKeyEvent

import sys
from pathlib import Path
from datetime import datetime

# Add parent for imports
PARENT_DIR = Path(__file__).parent.parent.parent
if str(PARENT_DIR) not in sys.path:
    sys.path.insert(0, str(PARENT_DIR))

from ui.styles import COLORS

# Import assistant
try:
    from ai.gemini_assistant import get_assistant, GeminiAssistant
    AI_AVAILABLE = True
except ImportError:
    AI_AVAILABLE = False
    GeminiAssistant = None


class AIQueryWorker(QThread):
    """Background thread for AI queries."""

    response_ready = pyqtSignal(str)
    error = pyqtSignal(str)

    def __init__(self, assistant, query: str, parent=None):
        super().__init__(parent)
        self.assistant = assistant
        self.query = query

    def run(self):
        try:
            response = self.assistant.query(self.query)
            self.response_ready.emit(response)
        except Exception as e:
            self.error.emit(str(e))


class ChatBubble(QFrame):
    """Individual chat message bubble."""

    def __init__(self, message: str, is_user: bool = False, parent=None):
        super().__init__(parent)
        self.is_user = is_user
        self.setup_ui(message)

    def setup_ui(self, message: str):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 8, 10, 8)
        layout.setSpacing(4)

        # Sender label
        sender = QLabel("You" if self.is_user else "Assistant")
        sender.setFont(QFont("Segoe UI", 9, QFont.Weight.Bold))
        sender.setStyleSheet(f"color: {COLORS['accent_blue'] if self.is_user else COLORS['profit']};")
        layout.addWidget(sender)

        # Message text
        msg_label = QLabel(message)
        msg_label.setWordWrap(True)
        msg_label.setTextInteractionFlags(Qt.TextInteractionFlag.TextSelectableByMouse)
        msg_label.setStyleSheet(f"color: {COLORS['text_primary']}; font-size: 12px;")
        layout.addWidget(msg_label)

        # Timestamp
        time_label = QLabel(datetime.now().strftime("%H:%M"))
        time_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 9px;")
        time_label.setAlignment(Qt.AlignmentFlag.AlignRight)
        layout.addWidget(time_label)

        # Style the bubble
        bg_color = COLORS['bg_light'] if self.is_user else COLORS['bg_card']
        self.setStyleSheet(f"""
            QFrame {{
                background-color: {bg_color};
                border-radius: 10px;
                margin: {'0 0 0 50px' if self.is_user else '0 50px 0 0'};
            }}
        """)


class AITerminalWidget(QWidget):
    """Main AI terminal chat interface."""

    def __init__(self, parent=None):
        super().__init__(parent)
        self.assistant = get_assistant() if AI_AVAILABLE else None
        self.worker = None
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(10, 10, 10, 10)
        layout.setSpacing(10)

        # Header
        header = QHBoxLayout()
        title = QLabel("AI Assistant")
        title.setFont(QFont("Segoe UI", 14, QFont.Weight.Bold))
        title.setStyleSheet(f"color: {COLORS['text_primary']};")
        header.addWidget(title)

        header.addStretch()

        # Status indicator
        self.status_label = QLabel("Ready")
        self.status_label.setStyleSheet(f"color: {COLORS['profit']}; font-size: 10px;")
        header.addWidget(self.status_label)

        # Clear button
        clear_btn = QPushButton("Clear")
        clear_btn.setFixedWidth(60)
        clear_btn.setToolTip("Clear conversation history")
        clear_btn.clicked.connect(self.clear_chat)
        header.addWidget(clear_btn)

        layout.addLayout(header)

        # Chat display area
        self.chat_scroll = QScrollArea()
        self.chat_scroll.setWidgetResizable(True)
        self.chat_scroll.setStyleSheet(f"""
            QScrollArea {{
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                background-color: {COLORS['bg_dark']};
            }}
        """)

        self.chat_container = QWidget()
        self.chat_layout = QVBoxLayout(self.chat_container)
        self.chat_layout.setContentsMargins(10, 10, 10, 10)
        self.chat_layout.setSpacing(10)
        self.chat_layout.addStretch()

        self.chat_scroll.setWidget(self.chat_container)
        layout.addWidget(self.chat_scroll, stretch=1)

        # Welcome message
        self.add_message(
            "Hello! I'm your AI trading assistant. I can help you:\n"
            "- Analyze your positions and trades\n"
            "- Debug errors and issues\n"
            "- Provide trading insights\n\n"
            "Try the quick actions below or ask me anything!",
            is_user=False
        )

        # Quick action buttons
        actions_layout = QHBoxLayout()
        actions_layout.setSpacing(8)

        analyze_btn = QPushButton("Analyze Positions")
        analyze_btn.setToolTip("Get AI analysis of your current positions")
        analyze_btn.clicked.connect(self.analyze_positions)
        actions_layout.addWidget(analyze_btn)

        debug_btn = QPushButton("Debug Errors")
        debug_btn.setToolTip("Get help with recent errors")
        debug_btn.clicked.connect(self.debug_errors)
        actions_layout.addWidget(debug_btn)

        summary_btn = QPushButton("Daily Summary")
        summary_btn.setToolTip("Get a summary of today's trading")
        summary_btn.clicked.connect(self.get_summary)
        actions_layout.addWidget(summary_btn)

        layout.addLayout(actions_layout)

        # Input area
        input_layout = QHBoxLayout()
        input_layout.setSpacing(8)

        self.input_field = QLineEdit()
        self.input_field.setPlaceholderText("Type a message...")
        self.input_field.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
                padding: 12px;
                color: {COLORS['text_primary']};
                font-size: 13px;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['accent_blue']};
            }}
        """)
        self.input_field.returnPressed.connect(self.send_message)
        input_layout.addWidget(self.input_field)

        send_btn = QPushButton("Send")
        send_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: white;
                border: none;
                border-radius: 8px;
                padding: 12px 20px;
                font-weight: bold;
            }}
            QPushButton:hover {{
                background-color: {COLORS['accent_blue']}cc;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_muted']};
            }}
        """)
        send_btn.clicked.connect(self.send_message)
        self.send_btn = send_btn
        input_layout.addWidget(send_btn)

        layout.addLayout(input_layout)

    def add_message(self, message: str, is_user: bool = False):
        """Add a message bubble to the chat."""
        # Remove the stretch at the end
        if self.chat_layout.count() > 0:
            item = self.chat_layout.itemAt(self.chat_layout.count() - 1)
            if item.spacerItem():
                self.chat_layout.removeItem(item)

        # Add the bubble
        bubble = ChatBubble(message, is_user)
        self.chat_layout.addWidget(bubble)

        # Re-add stretch
        self.chat_layout.addStretch()

        # Scroll to bottom
        QTimer.singleShot(100, self._scroll_to_bottom)

    def _scroll_to_bottom(self):
        """Scroll chat to bottom."""
        scrollbar = self.chat_scroll.verticalScrollBar()
        scrollbar.setValue(scrollbar.maximum())

    def send_message(self):
        """Send user message and get response."""
        message = self.input_field.text().strip()
        if not message:
            return

        if not self.assistant:
            self.add_message("AI Assistant is not available. Please check your Gemini API key.", False)
            return

        # Add user message
        self.add_message(message, is_user=True)
        self.input_field.clear()

        # Show loading state
        self.status_label.setText("Thinking...")
        self.status_label.setStyleSheet(f"color: {COLORS['accent_yellow']}; font-size: 10px;")
        self.send_btn.setEnabled(False)

        # Start worker thread
        self.worker = AIQueryWorker(self.assistant, message)
        self.worker.response_ready.connect(self.on_response)
        self.worker.error.connect(self.on_error)
        self.worker.start()

    def on_response(self, response: str):
        """Handle AI response."""
        self.add_message(response, is_user=False)
        self.status_label.setText("Ready")
        self.status_label.setStyleSheet(f"color: {COLORS['profit']}; font-size: 10px;")
        self.send_btn.setEnabled(True)

    def on_error(self, error: str):
        """Handle AI error."""
        self.add_message(f"Error: {error}", is_user=False)
        self.status_label.setText("Error")
        self.status_label.setStyleSheet(f"color: {COLORS['loss']}; font-size: 10px;")
        self.send_btn.setEnabled(True)

    def analyze_positions(self):
        """Quick action: Analyze positions."""
        self.input_field.setText("Analyze my current positions")
        self.send_message()

    def debug_errors(self):
        """Quick action: Debug errors."""
        self.input_field.setText("Help me debug recent errors in the trading bot")
        self.send_message()

    def get_summary(self):
        """Quick action: Get daily summary."""
        self.input_field.setText("Give me a summary of today's trading activity")
        self.send_message()

    def clear_chat(self):
        """Clear the chat history."""
        # Clear UI
        while self.chat_layout.count() > 0:
            item = self.chat_layout.takeAt(0)
            if item.widget():
                item.widget().deleteLater()

        # Clear assistant history
        if self.assistant:
            self.assistant.clear_history()

        # Re-add welcome message
        self.chat_layout.addStretch()
        self.add_message(
            "Chat cleared. How can I help you?",
            is_user=False
        )

    def update_trading_context(self, context: dict):
        """Update the trading context for the assistant."""
        if self.assistant:
            self.assistant.set_trading_context(context)


class AITerminalDock(QDockWidget):
    """Dockable AI Terminal panel."""

    def __init__(self, parent=None):
        super().__init__("AI Assistant", parent)
        self.setAllowedAreas(
            Qt.DockWidgetArea.LeftDockWidgetArea |
            Qt.DockWidgetArea.RightDockWidgetArea |
            Qt.DockWidgetArea.BottomDockWidgetArea
        )

        # Set minimum size
        self.setMinimumWidth(350)
        self.setMinimumHeight(400)

        # Create terminal widget
        self.terminal = AITerminalWidget()
        self.setWidget(self.terminal)

        # Style the dock
        self.setStyleSheet(f"""
            QDockWidget {{
                background-color: {COLORS['bg_medium']};
                color: {COLORS['text_primary']};
                font-weight: bold;
            }}
            QDockWidget::title {{
                background-color: {COLORS['bg_card']};
                padding: 8px;
                text-align: left;
            }}
            QDockWidget::close-button, QDockWidget::float-button {{
                background-color: transparent;
                border: none;
            }}
        """)

    def update_trading_context(self, context: dict):
        """Update trading context in the terminal."""
        self.terminal.update_trading_context(context)
