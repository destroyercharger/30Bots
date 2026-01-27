"""
Trade Logger - Track all trades, wins, and losses
"""

import sqlite3
from datetime import datetime, timedelta
from pathlib import Path
from typing import Optional, Dict, List, Tuple
import json


class TradeLogger:
    """SQLite-based trade logging for tracking performance."""

    def __init__(self, db_path: str = None):
        if db_path is None:
            db_path = Path(__file__).parent.parent / "trades.db"
        self.db_path = str(db_path)
        self.init_db()

    def init_db(self):
        """Initialize the database tables."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Trades table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS trades (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                symbol TEXT NOT NULL,
                side TEXT NOT NULL,
                entry_price REAL NOT NULL,
                exit_price REAL,
                shares INTEGER NOT NULL,
                entry_time TEXT NOT NULL,
                exit_time TEXT,
                status TEXT DEFAULT 'open',
                pnl REAL DEFAULT 0,
                pnl_pct REAL DEFAULT 0,
                ai_model TEXT,
                ai_confidence REAL,
                stop_loss REAL,
                take_profit REAL,
                trailing_stop REAL,
                exit_reason TEXT,
                trade_type TEXT DEFAULT 'day',
                notes TEXT
            )
        ''')

        # Daily stats table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS daily_stats (
                date TEXT PRIMARY KEY,
                total_trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                win_rate REAL DEFAULT 0,
                best_trade REAL DEFAULT 0,
                worst_trade REAL DEFAULT 0,
                avg_win REAL DEFAULT 0,
                avg_loss REAL DEFAULT 0
            )
        ''')

        # Model performance table
        cursor.execute('''
            CREATE TABLE IF NOT EXISTS model_performance (
                id INTEGER PRIMARY KEY AUTOINCREMENT,
                model_name TEXT NOT NULL,
                date TEXT NOT NULL,
                trades INTEGER DEFAULT 0,
                wins INTEGER DEFAULT 0,
                losses INTEGER DEFAULT 0,
                total_pnl REAL DEFAULT 0,
                UNIQUE(model_name, date)
            )
        ''')

        conn.commit()
        conn.close()

    def log_entry(self, symbol: str, side: str, price: float, shares: int,
                  ai_model: str = None, ai_confidence: float = None,
                  stop_loss: float = None, take_profit: float = None,
                  trade_type: str = 'day') -> int:
        """Log a new trade entry. Returns trade ID."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            INSERT INTO trades (
                symbol, side, entry_price, shares, entry_time,
                ai_model, ai_confidence, stop_loss, take_profit, trade_type, status
            ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, 'open')
        ''', (
            symbol, side, price, shares, datetime.now().isoformat(),
            ai_model, ai_confidence, stop_loss, take_profit, trade_type
        ))

        trade_id = cursor.lastrowid
        conn.commit()
        conn.close()

        return trade_id

    def log_exit(self, trade_id: int, exit_price: float, exit_reason: str = 'manual') -> Dict:
        """Log a trade exit and calculate P&L."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        # Get the trade
        cursor.execute('SELECT * FROM trades WHERE id = ?', (trade_id,))
        row = cursor.fetchone()

        if not row:
            conn.close()
            return None

        columns = [desc[0] for desc in cursor.description]
        trade = dict(zip(columns, row))

        # Calculate P&L
        entry_price = trade['entry_price']
        shares = trade['shares']
        side = trade['side']

        if side == 'buy':
            pnl = (exit_price - entry_price) * shares
            pnl_pct = ((exit_price - entry_price) / entry_price) * 100
        else:  # sell/short
            pnl = (entry_price - exit_price) * shares
            pnl_pct = ((entry_price - exit_price) / entry_price) * 100

        status = 'win' if pnl > 0 else 'loss'

        # Update trade
        cursor.execute('''
            UPDATE trades SET
                exit_price = ?,
                exit_time = ?,
                status = ?,
                pnl = ?,
                pnl_pct = ?,
                exit_reason = ?
            WHERE id = ?
        ''', (exit_price, datetime.now().isoformat(), status, pnl, pnl_pct, exit_reason, trade_id))

        # Update daily stats
        self._update_daily_stats(cursor, pnl)

        # Update model performance
        if trade['ai_model']:
            self._update_model_performance(cursor, trade['ai_model'], pnl)

        conn.commit()
        conn.close()

        return {
            'trade_id': trade_id,
            'symbol': trade['symbol'],
            'side': side,
            'entry_price': entry_price,
            'exit_price': exit_price,
            'shares': shares,
            'pnl': pnl,
            'pnl_pct': pnl_pct,
            'status': status,
            'exit_reason': exit_reason,
            'ai_model': trade['ai_model']
        }

    def log_exit_by_symbol(self, symbol: str, exit_price: float, exit_reason: str = 'manual') -> Dict:
        """Log exit for the most recent open trade of a symbol."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT id FROM trades
            WHERE symbol = ? AND status = 'open'
            ORDER BY entry_time DESC LIMIT 1
        ''', (symbol,))

        row = cursor.fetchone()
        conn.close()

        if row:
            return self.log_exit(row[0], exit_price, exit_reason)
        return None

    def _update_daily_stats(self, cursor, pnl: float):
        """Update daily statistics."""
        today = datetime.now().strftime('%Y-%m-%d')

        cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (today,))
        row = cursor.fetchone()

        if row:
            total_trades = row[1] + 1
            wins = row[2] + (1 if pnl > 0 else 0)
            losses = row[3] + (1 if pnl < 0 else 0)
            total_pnl = row[4] + pnl
            win_rate = (wins / total_trades * 100) if total_trades > 0 else 0
            best_trade = max(row[5], pnl)
            worst_trade = min(row[6], pnl)

            cursor.execute('''
                UPDATE daily_stats SET
                    total_trades = ?,
                    wins = ?,
                    losses = ?,
                    total_pnl = ?,
                    win_rate = ?,
                    best_trade = ?,
                    worst_trade = ?
                WHERE date = ?
            ''', (total_trades, wins, losses, total_pnl, win_rate, best_trade, worst_trade, today))
        else:
            cursor.execute('''
                INSERT INTO daily_stats (date, total_trades, wins, losses, total_pnl, win_rate, best_trade, worst_trade)
                VALUES (?, 1, ?, ?, ?, ?, ?, ?)
            ''', (today, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0, pnl,
                  100 if pnl > 0 else 0, pnl if pnl > 0 else 0, pnl if pnl < 0 else 0))

    def _update_model_performance(self, cursor, model_name: str, pnl: float):
        """Update model performance stats."""
        today = datetime.now().strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT * FROM model_performance WHERE model_name = ? AND date = ?
        ''', (model_name, today))
        row = cursor.fetchone()

        if row:
            cursor.execute('''
                UPDATE model_performance SET
                    trades = trades + 1,
                    wins = wins + ?,
                    losses = losses + ?,
                    total_pnl = total_pnl + ?
                WHERE model_name = ? AND date = ?
            ''', (1 if pnl > 0 else 0, 1 if pnl < 0 else 0, pnl, model_name, today))
        else:
            cursor.execute('''
                INSERT INTO model_performance (model_name, date, trades, wins, losses, total_pnl)
                VALUES (?, ?, 1, ?, ?, ?)
            ''', (model_name, today, 1 if pnl > 0 else 0, 1 if pnl < 0 else 0, pnl))

    def get_open_trades(self) -> List[Dict]:
        """Get all open trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('SELECT * FROM trades WHERE status = "open" ORDER BY entry_time DESC')
        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def get_open_trade_for_symbol(self, symbol: str) -> Optional[Dict]:
        """Get open trade for a symbol."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM trades
            WHERE symbol = ? AND status = 'open'
            ORDER BY entry_time DESC LIMIT 1
        ''', (symbol,))

        row = cursor.fetchone()
        if row:
            columns = [desc[0] for desc in cursor.description]
            conn.close()
            return dict(zip(columns, row))

        conn.close()
        return None

    def get_today_stats(self) -> Dict:
        """Get today's trading statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        today = datetime.now().strftime('%Y-%m-%d')
        cursor.execute('SELECT * FROM daily_stats WHERE date = ?', (today,))
        row = cursor.fetchone()

        conn.close()

        if row:
            return {
                'date': row[0],
                'total_trades': row[1],
                'wins': row[2],
                'losses': row[3],
                'total_pnl': row[4],
                'win_rate': row[5],
                'best_trade': row[6],
                'worst_trade': row[7]
            }

        return {
            'date': today,
            'total_trades': 0,
            'wins': 0,
            'losses': 0,
            'total_pnl': 0,
            'win_rate': 0,
            'best_trade': 0,
            'worst_trade': 0
        }

    def get_recent_trades(self, limit: int = 20) -> List[Dict]:
        """Get recent closed trades."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT * FROM trades
            WHERE status != 'open'
            ORDER BY exit_time DESC LIMIT ?
        ''', (limit,))

        rows = cursor.fetchall()
        columns = [desc[0] for desc in cursor.description]

        conn.close()
        return [dict(zip(columns, row)) for row in rows]

    def get_model_performance(self, days: int = 7) -> List[Dict]:
        """Get model performance over recent days."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        start_date = (datetime.now() - timedelta(days=days)).strftime('%Y-%m-%d')

        cursor.execute('''
            SELECT model_name,
                   SUM(trades) as total_trades,
                   SUM(wins) as total_wins,
                   SUM(losses) as total_losses,
                   SUM(total_pnl) as total_pnl
            FROM model_performance
            WHERE date >= ?
            GROUP BY model_name
            ORDER BY total_pnl DESC
        ''', (start_date,))

        rows = cursor.fetchall()
        conn.close()

        results = []
        for row in rows:
            win_rate = (row[2] / row[1] * 100) if row[1] > 0 else 0
            results.append({
                'model': row[0],
                'trades': row[1],
                'wins': row[2],
                'losses': row[3],
                'pnl': row[4],
                'win_rate': win_rate
            })

        return results

    def get_overall_stats(self) -> Dict:
        """Get overall trading statistics."""
        conn = sqlite3.connect(self.db_path)
        cursor = conn.cursor()

        cursor.execute('''
            SELECT
                COUNT(*) as total,
                SUM(CASE WHEN status = 'win' THEN 1 ELSE 0 END) as wins,
                SUM(CASE WHEN status = 'loss' THEN 1 ELSE 0 END) as losses,
                SUM(pnl) as total_pnl,
                AVG(CASE WHEN pnl > 0 THEN pnl END) as avg_win,
                AVG(CASE WHEN pnl < 0 THEN pnl END) as avg_loss,
                MAX(pnl) as best_trade,
                MIN(pnl) as worst_trade
            FROM trades WHERE status != 'open'
        ''')

        row = cursor.fetchone()
        conn.close()

        total = row[0] or 0
        wins = row[1] or 0
        losses = row[2] or 0

        return {
            'total_trades': total,
            'wins': wins,
            'losses': losses,
            'win_rate': (wins / total * 100) if total > 0 else 0,
            'total_pnl': row[3] or 0,
            'avg_win': row[4] or 0,
            'avg_loss': row[5] or 0,
            'best_trade': row[6] or 0,
            'worst_trade': row[7] or 0
        }


# Global logger instance
_trade_logger = None


def get_trade_logger() -> TradeLogger:
    """Get or create global trade logger."""
    global _trade_logger
    if _trade_logger is None:
        _trade_logger = TradeLogger()
    return _trade_logger
