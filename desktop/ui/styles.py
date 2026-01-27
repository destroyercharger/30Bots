"""
30Bots Desktop Trading Application - Dark Theme Styles
Bloomberg/TradeStation inspired dark theme
"""

# Color Palette
COLORS = {
    # Background colors
    'bg_dark': '#0d1117',
    'bg_medium': '#161b22',
    'bg_light': '#21262d',
    'bg_card': '#1c2128',

    # Border colors
    'border': '#30363d',
    'border_light': '#484f58',

    # Text colors
    'text_primary': '#e6edf3',
    'text_secondary': '#8b949e',
    'text_muted': '#6e7681',

    # Accent colors
    'accent_blue': '#58a6ff',
    'accent_green': '#3fb950',
    'accent_red': '#f85149',
    'accent_yellow': '#d29922',
    'accent_orange': '#db6d28',
    'accent_purple': '#a371f7',

    # Trading colors
    'profit': '#26a69a',
    'loss': '#ef5350',
    'neutral': '#8b949e',

    # Status colors
    'status_active': '#3fb950',
    'status_warning': '#d29922',
    'status_error': '#f85149',
}

# Main application stylesheet
DARK_THEME = f"""
/* ============================================
   GLOBAL STYLES
   ============================================ */
QWidget {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_primary']};
    font-family: 'Segoe UI', 'SF Pro Display', -apple-system, sans-serif;
    font-size: 13px;
}}

/* ============================================
   MAIN WINDOW
   ============================================ */
QMainWindow {{
    background-color: {COLORS['bg_dark']};
}}

QMainWindow::separator {{
    background-color: {COLORS['border']};
    width: 1px;
    height: 1px;
}}

/* ============================================
   TAB WIDGET
   ============================================ */
QTabWidget::pane {{
    border: 1px solid {COLORS['border']};
    background-color: {COLORS['bg_medium']};
    border-radius: 4px;
}}

QTabBar::tab {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_secondary']};
    padding: 10px 20px;
    margin-right: 2px;
    border-top-left-radius: 4px;
    border-top-right-radius: 4px;
    border: 1px solid {COLORS['border']};
    border-bottom: none;
}}

QTabBar::tab:selected {{
    background-color: {COLORS['bg_medium']};
    color: {COLORS['text_primary']};
    border-bottom: 2px solid {COLORS['accent_blue']};
}}

QTabBar::tab:hover:!selected {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
}}

/* ============================================
   TABLES
   ============================================ */
QTableWidget, QTableView {{
    background-color: {COLORS['bg_card']};
    alternate-background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    gridline-color: {COLORS['border']};
    selection-background-color: {COLORS['accent_blue']};
    selection-color: {COLORS['text_primary']};
}}

QTableWidget::item, QTableView::item {{
    padding: 8px;
    border-bottom: 1px solid {COLORS['border']};
}}

QHeaderView::section {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_secondary']};
    padding: 10px;
    border: none;
    border-bottom: 2px solid {COLORS['border']};
    font-weight: 600;
}}

/* ============================================
   BUTTONS
   ============================================ */
QPushButton {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
}}

QPushButton:hover {{
    background-color: {COLORS['bg_card']};
    border-color: {COLORS['border_light']};
}}

QPushButton:pressed {{
    background-color: {COLORS['bg_medium']};
}}

QPushButton:disabled {{
    background-color: {COLORS['bg_dark']};
    color: {COLORS['text_muted']};
}}

/* Primary button */
QPushButton[class="primary"] {{
    background-color: {COLORS['accent_blue']};
    border-color: {COLORS['accent_blue']};
}}

QPushButton[class="primary"]:hover {{
    background-color: #4c9aed;
}}

/* Success button (Buy) */
QPushButton[class="success"] {{
    background-color: {COLORS['accent_green']};
    border-color: {COLORS['accent_green']};
}}

/* Danger button (Sell) */
QPushButton[class="danger"] {{
    background-color: {COLORS['accent_red']};
    border-color: {COLORS['accent_red']};
}}

/* ============================================
   INPUT FIELDS
   ============================================ */
QLineEdit, QSpinBox, QDoubleSpinBox, QComboBox {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
    padding: 8px 12px;
}}

QLineEdit:focus, QSpinBox:focus, QDoubleSpinBox:focus, QComboBox:focus {{
    border-color: {COLORS['accent_blue']};
}}

QComboBox::drop-down {{
    border: none;
    padding-right: 10px;
}}

QComboBox::down-arrow {{
    image: none;
    border-left: 5px solid transparent;
    border-right: 5px solid transparent;
    border-top: 5px solid {COLORS['text_secondary']};
}}

QComboBox QAbstractItemView {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    selection-background-color: {COLORS['accent_blue']};
}}

/* ============================================
   SCROLL BARS
   ============================================ */
QScrollBar:vertical {{
    background-color: {COLORS['bg_dark']};
    width: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:vertical {{
    background-color: {COLORS['bg_light']};
    min-height: 30px;
    border-radius: 6px;
    margin: 2px;
}}

QScrollBar::handle:vertical:hover {{
    background-color: {COLORS['border_light']};
}}

QScrollBar::add-line:vertical, QScrollBar::sub-line:vertical {{
    height: 0px;
}}

QScrollBar:horizontal {{
    background-color: {COLORS['bg_dark']};
    height: 12px;
    border-radius: 6px;
}}

QScrollBar::handle:horizontal {{
    background-color: {COLORS['bg_light']};
    min-width: 30px;
    border-radius: 6px;
    margin: 2px;
}}

/* ============================================
   GROUP BOX / CARDS
   ============================================ */
QGroupBox {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
    margin-top: 16px;
    padding: 16px;
    font-weight: 600;
}}

QGroupBox::title {{
    subcontrol-origin: margin;
    subcontrol-position: top left;
    padding: 0 8px;
    color: {COLORS['text_secondary']};
}}

/* ============================================
   LABELS
   ============================================ */
QLabel {{
    color: {COLORS['text_primary']};
}}

QLabel[class="header"] {{
    font-size: 18px;
    font-weight: 600;
    color: {COLORS['text_primary']};
}}

QLabel[class="subheader"] {{
    font-size: 14px;
    color: {COLORS['text_secondary']};
}}

QLabel[class="value-positive"] {{
    color: {COLORS['profit']};
    font-weight: 600;
}}

QLabel[class="value-negative"] {{
    color: {COLORS['loss']};
    font-weight: 600;
}}

/* ============================================
   STATUS BAR
   ============================================ */
QStatusBar {{
    background-color: {COLORS['bg_light']};
    border-top: 1px solid {COLORS['border']};
    color: {COLORS['text_secondary']};
    padding: 4px;
}}

QStatusBar::item {{
    border: none;
}}

/* ============================================
   TOOL BAR
   ============================================ */
QToolBar {{
    background-color: {COLORS['bg_medium']};
    border: none;
    spacing: 8px;
    padding: 8px;
}}

QToolButton {{
    background-color: transparent;
    border: none;
    border-radius: 4px;
    padding: 8px;
}}

QToolButton:hover {{
    background-color: {COLORS['bg_light']};
}}

/* ============================================
   MENU
   ============================================ */
QMenuBar {{
    background-color: {COLORS['bg_medium']};
    border-bottom: 1px solid {COLORS['border']};
}}

QMenuBar::item {{
    padding: 8px 12px;
}}

QMenuBar::item:selected {{
    background-color: {COLORS['bg_light']};
}}

QMenu {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 4px;
}}

QMenu::item {{
    padding: 8px 24px;
}}

QMenu::item:selected {{
    background-color: {COLORS['accent_blue']};
}}

/* ============================================
   PROGRESS BAR
   ============================================ */
QProgressBar {{
    background-color: {COLORS['bg_light']};
    border: none;
    border-radius: 4px;
    height: 8px;
    text-align: center;
}}

QProgressBar::chunk {{
    background-color: {COLORS['accent_blue']};
    border-radius: 4px;
}}

/* ============================================
   SPLITTER
   ============================================ */
QSplitter::handle {{
    background-color: {COLORS['border']};
}}

QSplitter::handle:horizontal {{
    width: 2px;
}}

QSplitter::handle:vertical {{
    height: 2px;
}}

/* ============================================
   TOOLTIPS
   ============================================ */
QToolTip {{
    background-color: {COLORS['bg_card']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 10px 14px;
    font-size: 12px;
}}

/* ============================================
   DOCK WIDGETS
   ============================================ */
QDockWidget {{
    background-color: {COLORS['bg_medium']};
    border: 1px solid {COLORS['border']};
}}

QDockWidget::title {{
    background-color: {COLORS['bg_card']};
    padding: 10px;
    text-align: left;
    font-weight: 600;
}}

QDockWidget::close-button, QDockWidget::float-button {{
    background: transparent;
    border: none;
    padding: 4px;
}}

QDockWidget::close-button:hover, QDockWidget::float-button:hover {{
    background-color: {COLORS['bg_light']};
    border-radius: 4px;
}}

/* ============================================
   CHECKBOX & RADIO
   ============================================ */
QCheckBox {{
    spacing: 8px;
    color: {COLORS['text_primary']};
}}

QCheckBox::indicator {{
    width: 18px;
    height: 18px;
    border-radius: 4px;
    border: 2px solid {COLORS['border']};
    background-color: {COLORS['bg_card']};
}}

QCheckBox::indicator:checked {{
    background-color: {COLORS['accent_blue']};
    border-color: {COLORS['accent_blue']};
}}

QCheckBox::indicator:hover {{
    border-color: {COLORS['accent_blue']};
}}

/* ============================================
   ENHANCED BUTTONS
   ============================================ */
QPushButton {{
    background-color: {COLORS['bg_light']};
    color: {COLORS['text_primary']};
    border: 1px solid {COLORS['border']};
    border-radius: 6px;
    padding: 8px 16px;
    font-weight: 500;
    min-height: 16px;
}}

QPushButton:hover {{
    background-color: {COLORS['bg_card']};
    border-color: {COLORS['accent_blue']};
}}

QPushButton:pressed {{
    background-color: {COLORS['bg_medium']};
    border-color: {COLORS['accent_blue']};
}}

/* ============================================
   FRAMES & CARDS (Enhanced)
   ============================================ */
QFrame[frameShape="4"] {{
    background-color: {COLORS['bg_card']};
    border: 1px solid {COLORS['border']};
    border-radius: 8px;
}}
"""


def get_profit_color(value: float) -> str:
    """Get color based on profit/loss value"""
    if value > 0:
        return COLORS['profit']
    elif value < 0:
        return COLORS['loss']
    return COLORS['neutral']


def get_risk_color(score: float) -> str:
    """Get color based on risk score (0-100, higher = safer)"""
    if score >= 80:
        return COLORS['profit']
    elif score >= 60:
        return COLORS['accent_yellow']
    elif score >= 40:
        return COLORS['accent_orange']
    return COLORS['loss']
