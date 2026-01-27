"""
Trading Panel Widget
Buy/Sell buttons and order entry
"""

from PyQt6.QtWidgets import (
    QWidget, QVBoxLayout, QHBoxLayout, QLabel, QPushButton,
    QLineEdit, QComboBox, QSpinBox, QDoubleSpinBox, QFrame,
    QMessageBox, QGroupBox
)
from PyQt6.QtCore import Qt, pyqtSignal
from PyQt6.QtGui import QDoubleValidator, QIntValidator

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))

from ui.styles import COLORS


class TradingPanel(QWidget):
    """Trading panel with buy/sell buttons and order entry"""

    # Signals
    order_submitted = pyqtSignal(dict)  # Order details
    order_executed = pyqtSignal(str, str, float, int, float)  # symbol, side, price, qty, total
    ai_scan_requested = pyqtSignal(str)  # symbol to scan

    def __init__(self, parent=None):
        super().__init__(parent)
        self.broker = None
        self.current_price = 0.0
        self.current_symbol = "AAPL"
        self.ai_risk_params = None  # Stored AI-suggested risk parameters
        self.setup_ui()

    def setup_ui(self):
        layout = QVBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(12)

        # Header
        header = QLabel("Quick Trade")
        header.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        layout.addWidget(header)

        # Main trading frame
        trade_frame = QFrame()
        trade_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_card']};
                border: 1px solid {COLORS['border']};
                border-radius: 8px;
            }}
        """)
        frame_layout = QVBoxLayout(trade_frame)
        frame_layout.setContentsMargins(16, 12, 16, 12)
        frame_layout.setSpacing(12)

        # Symbol row
        symbol_layout = QHBoxLayout()
        symbol_label = QLabel("Symbol:")
        symbol_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 60px;")
        symbol_layout.addWidget(symbol_label)

        self.symbol_input = QLineEdit("AAPL")
        self.symbol_input.setMaxLength(5)
        self.symbol_input.setStyleSheet(f"""
            QLineEdit {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS['text_primary']};
                font-size: 14px;
                font-weight: 600;
            }}
            QLineEdit:focus {{
                border-color: {COLORS['accent_blue']};
            }}
        """)
        self.symbol_input.textChanged.connect(self.on_symbol_changed)
        symbol_layout.addWidget(self.symbol_input)
        frame_layout.addLayout(symbol_layout)

        # Price display
        price_layout = QHBoxLayout()
        price_label = QLabel("Price:")
        price_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 60px;")
        price_layout.addWidget(price_label)

        self.price_display = QLabel("$0.00")
        self.price_display.setStyleSheet(f"""
            font-size: 18px;
            font-weight: 700;
            color: {COLORS['text_primary']};
        """)
        price_layout.addWidget(self.price_display)
        price_layout.addStretch()
        frame_layout.addLayout(price_layout)

        # Quantity row
        qty_layout = QHBoxLayout()
        qty_label = QLabel("Shares:")
        qty_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 60px;")
        qty_layout.addWidget(qty_label)

        self.qty_spin = QSpinBox()
        self.qty_spin.setRange(1, 10000)
        self.qty_spin.setValue(10)
        self.qty_spin.setStyleSheet(f"""
            QSpinBox {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS['text_primary']};
                font-size: 14px;
            }}
            QSpinBox:focus {{
                border-color: {COLORS['accent_blue']};
            }}
        """)
        self.qty_spin.valueChanged.connect(self.update_total)
        qty_layout.addWidget(self.qty_spin)

        # Quick quantity buttons
        for qty in [1, 10, 50, 100]:
            btn = QPushButton(str(qty))
            btn.setFixedWidth(40)
            btn.setStyleSheet(f"""
                QPushButton {{
                    background-color: {COLORS['bg_light']};
                    color: {COLORS['text_secondary']};
                    border: 1px solid {COLORS['border']};
                    border-radius: 4px;
                    padding: 4px;
                    font-size: 11px;
                }}
                QPushButton:hover {{
                    border-color: {COLORS['accent_blue']};
                }}
            """)
            btn.clicked.connect(lambda checked, q=qty: self.qty_spin.setValue(q))
            qty_layout.addWidget(btn)

        frame_layout.addLayout(qty_layout)

        # Order type row
        type_layout = QHBoxLayout()
        type_label = QLabel("Type:")
        type_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 60px;")
        type_layout.addWidget(type_label)

        self.order_type = QComboBox()
        self.order_type.addItems(["Market", "Limit"])
        self.order_type.setStyleSheet(f"""
            QComboBox {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS['text_primary']};
                min-width: 100px;
            }}
            QComboBox:focus {{
                border-color: {COLORS['accent_blue']};
            }}
            QComboBox::drop-down {{
                border: none;
            }}
            QComboBox QAbstractItemView {{
                background-color: {COLORS['bg_card']};
                color: {COLORS['text_primary']};
                selection-background-color: {COLORS['bg_light']};
            }}
        """)
        self.order_type.currentTextChanged.connect(self.on_order_type_changed)
        type_layout.addWidget(self.order_type)

        # Limit price input (hidden by default)
        self.limit_price_input = QDoubleSpinBox()
        self.limit_price_input.setRange(0.01, 99999.99)
        self.limit_price_input.setDecimals(2)
        self.limit_price_input.setPrefix("$")
        self.limit_price_input.setStyleSheet(f"""
            QDoubleSpinBox {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['border']};
                border-radius: 4px;
                padding: 8px;
                color: {COLORS['text_primary']};
            }}
        """)
        self.limit_price_input.setVisible(False)
        type_layout.addWidget(self.limit_price_input)

        type_layout.addStretch()
        frame_layout.addLayout(type_layout)

        # Estimated total
        total_layout = QHBoxLayout()
        total_label = QLabel("Est. Total:")
        total_label.setStyleSheet(f"color: {COLORS['text_secondary']}; min-width: 60px;")
        total_layout.addWidget(total_label)

        self.total_display = QLabel("$0.00")
        self.total_display.setStyleSheet(f"""
            font-size: 16px;
            font-weight: 600;
            color: {COLORS['text_primary']};
        """)
        total_layout.addWidget(self.total_display)
        total_layout.addStretch()
        frame_layout.addLayout(total_layout)

        # Separator
        sep = QFrame()
        sep.setFrameShape(QFrame.Shape.HLine)
        sep.setStyleSheet(f"background-color: {COLORS['border']};")
        frame_layout.addWidget(sep)

        # Buy/Sell buttons
        buttons_layout = QHBoxLayout()
        buttons_layout.setSpacing(12)

        self.buy_button = QPushButton("BUY")
        self.buy_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['profit']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 14px 24px;
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #00c853;
            }}
            QPushButton:pressed {{
                background-color: #009624;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.buy_button.clicked.connect(self.on_buy_clicked)
        buttons_layout.addWidget(self.buy_button)

        self.sell_button = QPushButton("SELL")
        self.sell_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['loss']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 14px 24px;
                font-size: 16px;
                font-weight: 700;
            }}
            QPushButton:hover {{
                background-color: #ff1744;
            }}
            QPushButton:pressed {{
                background-color: #c4001d;
            }}
            QPushButton:disabled {{
                background-color: {COLORS['bg_light']};
                color: {COLORS['text_muted']};
            }}
        """)
        self.sell_button.clicked.connect(self.on_sell_clicked)
        buttons_layout.addWidget(self.sell_button)

        frame_layout.addLayout(buttons_layout)

        # AI Scan button
        self.ai_scan_button = QPushButton("🤖 AI Scan")
        self.ai_scan_button.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['accent_blue']};
                color: white;
                border: none;
                border-radius: 6px;
                padding: 10px 24px;
                font-size: 14px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #2196f3;
            }}
            QPushButton:pressed {{
                background-color: #1565c0;
            }}
        """)
        self.ai_scan_button.clicked.connect(self.on_ai_scan_clicked)
        frame_layout.addWidget(self.ai_scan_button)

        # AI Risk Parameters display (hidden by default)
        self.ai_params_frame = QFrame()
        self.ai_params_frame.setStyleSheet(f"""
            QFrame {{
                background-color: {COLORS['bg_light']};
                border: 1px solid {COLORS['accent_blue']};
                border-radius: 4px;
                padding: 8px;
            }}
        """)
        self.ai_params_frame.setVisible(False)
        ai_params_layout = QVBoxLayout(self.ai_params_frame)
        ai_params_layout.setContentsMargins(8, 8, 8, 8)
        ai_params_layout.setSpacing(4)

        ai_header = QLabel("AI Risk Parameters")
        ai_header.setStyleSheet(f"color: {COLORS['accent_blue']}; font-weight: 600; font-size: 12px;")
        ai_params_layout.addWidget(ai_header)

        self.ai_model_label = QLabel("Model: --")
        self.ai_model_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        ai_params_layout.addWidget(self.ai_model_label)

        self.ai_stop_label = QLabel("Stop: --")
        self.ai_stop_label.setStyleSheet(f"color: {COLORS['loss']}; font-size: 11px;")
        ai_params_layout.addWidget(self.ai_stop_label)

        self.ai_target_label = QLabel("Target: --")
        self.ai_target_label.setStyleSheet(f"color: {COLORS['profit']}; font-size: 11px;")
        ai_params_layout.addWidget(self.ai_target_label)

        self.ai_trailing_label = QLabel("Trailing: --")
        self.ai_trailing_label.setStyleSheet(f"color: {COLORS['text_secondary']}; font-size: 11px;")
        ai_params_layout.addWidget(self.ai_trailing_label)

        frame_layout.addWidget(self.ai_params_frame)

        # Status message
        self.status_label = QLabel("")
        self.status_label.setStyleSheet(f"color: {COLORS['text_muted']}; font-size: 12px;")
        self.status_label.setWordWrap(True)
        frame_layout.addWidget(self.status_label)

        layout.addWidget(trade_frame)
        layout.addStretch()

    def set_broker(self, broker):
        """Set the broker for order execution"""
        self.broker = broker
        if broker:
            self.buy_button.setEnabled(True)
            self.sell_button.setEnabled(True)
        else:
            self.buy_button.setEnabled(False)
            self.sell_button.setEnabled(False)
            self.status_label.setText("No broker connected")

    def set_symbol(self, symbol: str):
        """Set the current symbol"""
        self.symbol_input.setText(symbol.upper())
        self.current_symbol = symbol.upper()

    def set_price(self, price: float):
        """Update the current price display"""
        self.current_price = price
        self.price_display.setText(f"${price:,.2f}")
        self.limit_price_input.setValue(price)
        self.update_total()

    def update_total(self):
        """Update the estimated total"""
        qty = self.qty_spin.value()
        if self.order_type.currentText() == "Limit":
            price = self.limit_price_input.value()
        else:
            price = self.current_price
        total = qty * price
        self.total_display.setText(f"${total:,.2f}")

    def on_symbol_changed(self, text: str):
        """Handle symbol input change"""
        self.current_symbol = text.upper()
        self.symbol_input.setText(self.current_symbol)

    def on_order_type_changed(self, order_type: str):
        """Handle order type change"""
        self.limit_price_input.setVisible(order_type == "Limit")
        self.update_total()

    def on_buy_clicked(self):
        """Handle buy button click"""
        self.execute_order("buy")

    def on_sell_clicked(self):
        """Handle sell button click"""
        self.execute_order("sell")

    def execute_order(self, side: str):
        """Execute a buy or sell order"""
        symbol = self.current_symbol.strip().upper()
        qty = self.qty_spin.value()
        order_type = self.order_type.currentText().lower()

        if not symbol:
            self.show_error("Please enter a symbol")
            return

        if qty <= 0:
            self.show_error("Please enter a valid quantity")
            return

        # Confirm order
        price = self.limit_price_input.value() if order_type == "limit" else self.current_price
        total = qty * price

        confirm = QMessageBox.question(
            self,
            f"Confirm {side.upper()} Order",
            f"Are you sure you want to {side.upper()}?\n\n"
            f"Symbol: {symbol}\n"
            f"Quantity: {qty} shares\n"
            f"Type: {order_type.title()}\n"
            f"Est. Price: ${price:,.2f}\n"
            f"Est. Total: ${total:,.2f}",
            QMessageBox.StandardButton.Yes | QMessageBox.StandardButton.No,
            QMessageBox.StandardButton.No
        )

        if confirm != QMessageBox.StandardButton.Yes:
            return

        if not self.broker:
            self.show_error("No broker connected")
            return

        try:
            self.status_label.setText(f"Submitting {side} order...")
            self.status_label.setStyleSheet(f"color: {COLORS['accent_yellow']}; font-size: 12px;")

            # Import order types
            from data.broker_adapter import OrderSide, OrderType

            # Execute order
            if order_type == "market":
                order = self.broker.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    type=OrderType.MARKET
                )
            else:
                order = self.broker.place_order(
                    symbol=symbol,
                    qty=qty,
                    side=OrderSide.BUY if side == "buy" else OrderSide.SELL,
                    type=OrderType.LIMIT,
                    limit_price=self.limit_price_input.value()
                )

            # Success
            color = COLORS['profit'] if side == "buy" else COLORS['loss']
            self.status_label.setText(f"Order submitted: {side.upper()} {qty} {symbol}")
            self.status_label.setStyleSheet(f"color: {color}; font-size: 12px;")

            # Emit signal
            self.order_executed.emit(symbol, side, price, qty, total)

            # Emit order details
            self.order_submitted.emit({
                'symbol': symbol,
                'side': side,
                'qty': qty,
                'type': order_type,
                'price': price,
                'order_id': order.id if order else None
            })

        except Exception as e:
            self.show_error(f"Order failed: {str(e)}")

    def show_error(self, message: str):
        """Show error message"""
        self.status_label.setText(message)
        self.status_label.setStyleSheet(f"color: {COLORS['loss']}; font-size: 12px;")
        QMessageBox.warning(self, "Order Error", message)

    def on_ai_scan_clicked(self):
        """Handle AI scan button click"""
        self.ai_scan_button.setEnabled(False)
        self.ai_scan_button.setText("Scanning...")
        self.ai_scan_requested.emit(self.current_symbol)
        # Re-enable after a short delay
        from PyQt6.QtCore import QTimer
        QTimer.singleShot(2000, self.reset_ai_button)

    def reset_ai_button(self):
        """Reset AI scan button after scan"""
        self.ai_scan_button.setEnabled(True)
        self.ai_scan_button.setText("🤖 AI Scan")

    def update_ai_params(self, params: dict):
        """Update the AI risk parameters display"""
        if not params:
            self.ai_params_frame.setVisible(False)
            self.ai_risk_params = None
            return

        self.ai_risk_params = params
        self.ai_params_frame.setVisible(True)

        model = params.get('model', 'Unknown')
        stop_pct = params.get('stop_loss_pct', 0.02) * 100
        target_pct = params.get('take_profit_pct', 0.06) * 100
        trailing_act = params.get('trailing_activation_pct', 0.015) * 100
        trailing_stop = params.get('trailing_stop_pct', 0.01) * 100

        self.ai_model_label.setText(f"Model: {model}")
        self.ai_stop_label.setText(f"Stop: -{stop_pct:.1f}%")
        self.ai_target_label.setText(f"Target: +{target_pct:.1f}%")
        self.ai_trailing_label.setText(f"Trailing: +{trailing_act:.1f}% → {trailing_stop:.1f}%")

        # Calculate and show actual prices if we have current price
        if self.current_price > 0:
            stop_price = self.current_price * (1 - params.get('stop_loss_pct', 0.02))
            target_price = self.current_price * (1 + params.get('take_profit_pct', 0.06))
            self.ai_stop_label.setText(f"Stop: ${stop_price:.2f} (-{stop_pct:.1f}%)")
            self.ai_target_label.setText(f"Target: ${target_price:.2f} (+{target_pct:.1f}%)")

    def clear_ai_params(self):
        """Clear AI risk parameters"""
        self.ai_params_frame.setVisible(False)
        self.ai_risk_params = None


class QuickTradeButtons(QWidget):
    """Compact buy/sell buttons for embedding in other widgets"""

    buy_clicked = pyqtSignal(str)  # symbol
    sell_clicked = pyqtSignal(str)  # symbol

    def __init__(self, parent=None):
        super().__init__(parent)
        self.symbol = ""
        self.setup_ui()

    def setup_ui(self):
        layout = QHBoxLayout(self)
        layout.setContentsMargins(0, 0, 0, 0)
        layout.setSpacing(8)

        self.buy_btn = QPushButton("BUY")
        self.buy_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['profit']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #00c853;
            }}
        """)
        self.buy_btn.clicked.connect(lambda: self.buy_clicked.emit(self.symbol))
        layout.addWidget(self.buy_btn)

        self.sell_btn = QPushButton("SELL")
        self.sell_btn.setStyleSheet(f"""
            QPushButton {{
                background-color: {COLORS['loss']};
                color: white;
                border: none;
                border-radius: 4px;
                padding: 6px 16px;
                font-size: 12px;
                font-weight: 600;
            }}
            QPushButton:hover {{
                background-color: #ff1744;
            }}
        """)
        self.sell_btn.clicked.connect(lambda: self.sell_clicked.emit(self.symbol))
        layout.addWidget(self.sell_btn)

    def set_symbol(self, symbol: str):
        """Set the symbol for quick trade"""
        self.symbol = symbol
