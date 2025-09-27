import pandas as pd
import sqlite3
import numpy as np
from datetime import datetime, timedelta
from pathlib import Path
import json
import logging
from typing import Dict, List, Optional, Tuple, Union
from dataclasses import dataclass, asdict
from collections import defaultdict

# PDF and plotting imports
try:
    from reportlab.lib.pagesizes import letter, A4
    from reportlab.platypus import SimpleDocTemplate, Paragraph, Spacer, Table, TableStyle, PageBreak, Image
    from reportlab.lib.styles import getSampleStyleSheet, ParagraphStyle
    from reportlab.lib.units import inch
    from reportlab.lib import colors
    from reportlab.graphics.shapes import Drawing
    from reportlab.graphics.charts.lineplots import LinePlot
    from reportlab.graphics.charts.barcharts import VerticalBarChart
    from reportlab.graphics.widgetbase import Widget
    REPORTLAB_AVAILABLE = True
except ImportError:
    REPORTLAB_AVAILABLE = False
    print("ReportLab not available. PDF export will be disabled.")

try:
    import matplotlib.pyplot as plt
    import matplotlib.dates as mdates
    from matplotlib.backends.backend_pdf import PdfPages
    import seaborn as sns
    MATPLOTLIB_AVAILABLE = True
    plt.style.use('seaborn-v0_8')
except ImportError:
    MATPLOTLIB_AVAILABLE = False
    print("Matplotlib not available. Chart generation will be limited.")

@dataclass
class TradeInfo:
    """Data class for trade information with validation"""
    signal: str
    price: float
    quantity: float
    timestamp: Optional[datetime] = None
    order_id: Optional[str] = None
    commission: float = 0.0
    slippage: float = 0.0
    
    def __post_init__(self):
        if self.timestamp is None:
            self.timestamp = datetime.now()
        self.validate()
    
    def validate(self):
        """Validate trade data"""
        if self.signal not in ['buy', 'sell']:
            raise ValueError(f"Invalid signal: {self.signal}. Must be 'buy' or 'sell'")
        if self.price <= 0:
            raise ValueError(f"Price must be positive: {self.price}")
        if self.quantity <= 0:
            raise ValueError(f"Quantity must be positive: {self.quantity}")
        if self.commission < 0:
            raise ValueError(f"Commission cannot be negative: {self.commission}")
        if abs(self.slippage) > self.price * 0.1:  # 10% slippage seems excessive
            raise ValueError(f"Slippage seems excessive: {self.slippage}")

class PositionTracker:
    """Tracks positions and calculates realized/unrealized P&L"""
    
    def __init__(self):
        self.positions = defaultdict(lambda: {'quantity': 0.0, 'avg_price': 0.0, 'cost_basis': 0.0})
        self.realized_pnl = []
    
    def process_trade(self, symbol: str, trade: TradeInfo) -> Dict:
        """Process a trade and return P&L information"""
        position = self.positions[symbol]
        
        if trade.signal == 'buy':
            return self._process_buy(symbol, trade, position)
        else:
            return self._process_sell(symbol, trade, position)
    
    def _process_buy(self, symbol: str, trade: TradeInfo, position: Dict) -> Dict:
        """Process buy trade"""
        old_quantity = position['quantity']
        old_cost_basis = position['cost_basis']
        
        # Calculate new position
        trade_cost = trade.quantity * trade.price + trade.commission
        new_quantity = old_quantity + trade.quantity
        new_cost_basis = old_cost_basis + trade_cost
        
        position['quantity'] = new_quantity
        position['cost_basis'] = new_cost_basis
        position['avg_price'] = new_cost_basis / new_quantity if new_quantity > 0 else 0
        
        return {
            'realized_pnl': 0.0,  # No P&L on buy
            'position_after': new_quantity,
            'avg_price_after': position['avg_price']
        }
    
    def _process_sell(self, symbol: str, trade: TradeInfo, position: Dict) -> Dict:
        """Process sell trade and calculate realized P&L"""
        if position['quantity'] <= 0:
            # Short selling or selling without position
            old_quantity = position['quantity']
            old_cost_basis = position['cost_basis']
            
            trade_proceeds = trade.quantity * trade.price - trade.commission
            new_quantity = old_quantity - trade.quantity
            new_cost_basis = old_cost_basis - trade_proceeds
            
            position['quantity'] = new_quantity
            position['cost_basis'] = new_cost_basis
            position['avg_price'] = new_cost_basis / abs(new_quantity) if new_quantity != 0 else 0
            
            return {
                'realized_pnl': 0.0,  # Will be realized when position is closed
                'position_after': new_quantity,
                'avg_price_after': position['avg_price']
            }
        
        # Selling from long position
        sell_quantity = min(trade.quantity, position['quantity'])
        remaining_quantity = position['quantity'] - sell_quantity
        
        # Calculate realized P&L
        avg_cost_per_share = position['avg_price']
        proceeds = sell_quantity * trade.price - trade.commission
        cost = sell_quantity * avg_cost_per_share
        realized_pnl = proceeds - cost
        
        # Update position
        if remaining_quantity > 0:
            position['quantity'] = remaining_quantity
            position['cost_basis'] = remaining_quantity * avg_cost_per_share
        else:
            position['quantity'] = 0
            position['cost_basis'] = 0
            position['avg_price'] = 0
        
        # Record realized P&L
        pnl_record = {
            'timestamp': trade.timestamp,
            'symbol': symbol,
            'realized_pnl': realized_pnl,
            'quantity_sold': sell_quantity,
            'sell_price': trade.price,
            'avg_cost': avg_cost_per_share
        }
        self.realized_pnl.append(pnl_record)
        
        return {
            'realized_pnl': realized_pnl,
            'position_after': position['quantity'],
            'avg_price_after': position['avg_price']
        }
    
    def get_current_positions(self) -> Dict:
        """Get current positions"""
        return dict(self.positions)
    
    def get_realized_pnl_history(self) -> List[Dict]:
        """Get realized P&L history"""
        return self.realized_pnl.copy()

class DatabaseManager:
    """Manages SQLite database for trade persistence"""
    
    def __init__(self, db_path: str = "trades.db"):
        self.db_path = db_path
        self.init_database()
    
    def init_database(self):
        """Initialize database tables"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                CREATE TABLE IF NOT EXISTS trades (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    signal TEXT NOT NULL,
                    price REAL NOT NULL,
                    quantity REAL NOT NULL,
                    side TEXT NOT NULL,
                    mode TEXT,
                    order_id TEXT,
                    commission REAL DEFAULT 0.0,
                    slippage REAL DEFAULT 0.0,
                    realized_pnl REAL DEFAULT 0.0,
                    position_after REAL DEFAULT 0.0,
                    avg_price_after REAL DEFAULT 0.0,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS positions (
                    symbol TEXT PRIMARY KEY,
                    quantity REAL NOT NULL,
                    avg_price REAL NOT NULL,
                    cost_basis REAL NOT NULL,
                    last_updated TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
            
            conn.execute("""
                CREATE TABLE IF NOT EXISTS realized_pnl (
                    id INTEGER PRIMARY KEY AUTOINCREMENT,
                    timestamp TEXT NOT NULL,
                    symbol TEXT NOT NULL,
                    realized_pnl REAL NOT NULL,
                    quantity_sold REAL NOT NULL,
                    sell_price REAL NOT NULL,
                    avg_cost REAL NOT NULL,
                    created_at TEXT DEFAULT CURRENT_TIMESTAMP
                )
            """)
    
    def save_trade(self, trade_record: Dict):
        """Save trade to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO trades (
                    timestamp, symbol, signal, price, quantity, side, mode,
                    order_id, commission, slippage, realized_pnl, position_after, avg_price_after
                ) VALUES (?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?, ?)
            """, (
                trade_record['timestamp'].isoformat(),
                trade_record['symbol'],
                trade_record['signal'],
                trade_record['price'],
                trade_record['quantity'],
                trade_record['side'],
                trade_record.get('mode'),
                trade_record.get('order_id'),
                trade_record.get('commission', 0.0),
                trade_record.get('slippage', 0.0),
                trade_record.get('realized_pnl', 0.0),
                trade_record.get('position_after', 0.0),
                trade_record.get('avg_price_after', 0.0)
            ))
    
    def save_position(self, symbol: str, position: Dict):
        """Save position to database"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT OR REPLACE INTO positions (symbol, quantity, avg_price, cost_basis, last_updated)
                VALUES (?, ?, ?, ?, ?)
            """, (symbol, position['quantity'], position['avg_price'], 
                  position['cost_basis'], datetime.now().isoformat()))
    
    def save_realized_pnl(self, pnl_record: Dict):
        """Save realized P&L record"""
        with sqlite3.connect(self.db_path) as conn:
            conn.execute("""
                INSERT INTO realized_pnl (timestamp, symbol, realized_pnl, quantity_sold, sell_price, avg_cost)
                VALUES (?, ?, ?, ?, ?, ?)
            """, (
                pnl_record['timestamp'].isoformat(),
                pnl_record['symbol'],
                pnl_record['realized_pnl'],
                pnl_record['quantity_sold'],
                pnl_record['sell_price'],
                pnl_record['avg_cost']
            ))
    
    def load_trades(self) -> pd.DataFrame:
        """Load all trades from database"""
        with sqlite3.connect(self.db_path) as conn:
            return pd.read_sql_query("SELECT * FROM trades", conn, parse_dates=['timestamp'])
    
    def load_positions(self) -> Dict:
        """Load current positions from database"""
        with sqlite3.connect(self.db_path) as conn:
            df = pd.read_sql_query("SELECT * FROM positions", conn)
            positions = {}
            for _, row in df.iterrows():
                positions[row['symbol']] = {
                    'quantity': row['quantity'],
                    'avg_price': row['avg_price'],
                    'cost_basis': row['cost_basis']
                }
            return positions

class AdvancedStatistics:
    """Calculate advanced trading statistics"""
    
    @staticmethod
    def calculate_comprehensive_stats(trades_df: pd.DataFrame, realized_pnl: List[Dict]) -> Dict:
        """Calculate comprehensive trading statistics"""
        if trades_df.empty:
            return AdvancedStatistics._empty_stats()
        
        # Convert realized P&L to DataFrame
        pnl_df = pd.DataFrame(realized_pnl) if realized_pnl else pd.DataFrame()
        
        # Basic stats
        total_trades = len(trades_df)
        
        # P&L Analysis
        if not pnl_df.empty:
            total_realized_pnl = pnl_df['realized_pnl'].sum()
            winning_trades = pnl_df[pnl_df['realized_pnl'] > 0]
            losing_trades = pnl_df[pnl_df['realized_pnl'] < 0]
            
            win_rate = len(winning_trades) / len(pnl_df) * 100 if len(pnl_df) > 0 else 0
            avg_win = winning_trades['realized_pnl'].mean() if len(winning_trades) > 0 else 0
            avg_loss = losing_trades['realized_pnl'].mean() if len(losing_trades) > 0 else 0
            profit_factor = abs(winning_trades['realized_pnl'].sum() / losing_trades['realized_pnl'].sum()) if len(losing_trades) > 0 and losing_trades['realized_pnl'].sum() != 0 else float('inf')
        else:
            total_realized_pnl = 0
            win_rate = 0
            avg_win = 0
            avg_loss = 0
            profit_factor = 0
        
        # Risk Metrics
        if not pnl_df.empty and len(pnl_df) > 1:
            returns = pnl_df['realized_pnl'].pct_change().dropna()
            
            # Sharpe Ratio (assuming 252 trading days)
            if returns.std() != 0:
                sharpe_ratio = returns.mean() / returns.std() * np.sqrt(252)
            else:
                sharpe_ratio = 0
            
            # Sortino Ratio (downside deviation)
            downside_returns = returns[returns < 0]
            if len(downside_returns) > 0 and downside_returns.std() != 0:
                sortino_ratio = returns.mean() / downside_returns.std() * np.sqrt(252)
            else:
                sortino_ratio = 0
            
            # Maximum Drawdown
            cumulative_pnl = pnl_df['realized_pnl'].cumsum()
            running_max = cumulative_pnl.cummax()
            drawdown = (cumulative_pnl - running_max)
            max_drawdown = drawdown.min()
            max_drawdown_pct = (max_drawdown / running_max.max() * 100) if running_max.max() != 0 else 0
            
            # Calmar Ratio (annual return / max drawdown)
            annual_return = total_realized_pnl * 252 / len(pnl_df) if len(pnl_df) > 0 else 0
            calmar_ratio = annual_return / abs(max_drawdown) if max_drawdown != 0 else 0
        else:
            sharpe_ratio = 0
            sortino_ratio = 0
            max_drawdown = 0
            max_drawdown_pct = 0
            calmar_ratio = 0
        
        # Consecutive wins/losses
        consecutive_stats = AdvancedStatistics._calculate_consecutive_stats(pnl_df)
        
        # Trading frequency
        if len(trades_df) > 1:
            date_range = (trades_df['timestamp'].max() - trades_df['timestamp'].min()).days
            trades_per_day = total_trades / max(date_range, 1)
        else:
            trades_per_day = 0
        
        return {
            'total_trades': total_trades,
            'total_realized_pnl': total_realized_pnl,
            'win_rate': win_rate,
            'avg_win': avg_win,
            'avg_loss': avg_loss,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'sortino_ratio': sortino_ratio,
            'calmar_ratio': calmar_ratio,
            'max_drawdown': max_drawdown,
            'max_drawdown_pct': max_drawdown_pct,
            'max_consecutive_wins': consecutive_stats['max_wins'],
            'max_consecutive_losses': consecutive_stats['max_losses'],
            'current_streak': consecutive_stats['current_streak'],
            'trades_per_day': trades_per_day,
            'total_commission_paid': trades_df['commission'].sum() if 'commission' in trades_df.columns else 0,
            'avg_holding_period': AdvancedStatistics._calculate_avg_holding_period(trades_df)
        }
    
    @staticmethod
    def _empty_stats() -> Dict:
        """Return empty statistics"""
        return {
            'total_trades': 0,
            'total_realized_pnl': 0.0,
            'win_rate': 0.0,
            'avg_win': 0.0,
            'avg_loss': 0.0,
            'profit_factor': 0.0,
            'sharpe_ratio': 0.0,
            'sortino_ratio': 0.0,
            'calmar_ratio': 0.0,
            'max_drawdown': 0.0,
            'max_drawdown_pct': 0.0,
            'max_consecutive_wins': 0,
            'max_consecutive_losses': 0,
            'current_streak': 0,
            'trades_per_day': 0.0,
            'total_commission_paid': 0.0,
            'avg_holding_period': 0.0
        }
    
    @staticmethod
    def _calculate_consecutive_stats(pnl_df: pd.DataFrame) -> Dict:
        """Calculate consecutive wins/losses statistics"""
        if pnl_df.empty:
            return {'max_wins': 0, 'max_losses': 0, 'current_streak': 0}
        
        wins_losses = (pnl_df['realized_pnl'] > 0).astype(int)
        
        max_wins = 0
        max_losses = 0
        current_wins = 0
        current_losses = 0
        
        for is_win in wins_losses:
            if is_win:
                current_wins += 1
                current_losses = 0
                max_wins = max(max_wins, current_wins)
            else:
                current_losses += 1
                current_wins = 0
                max_losses = max(max_losses, current_losses)
        
        # Current streak
        if len(wins_losses) > 0:
            current_streak = current_wins if wins_losses.iloc[-1] else -current_losses
        else:
            current_streak = 0
        
        return {
            'max_wins': max_wins,
            'max_losses': max_losses,
            'current_streak': current_streak
        }
    
    @staticmethod
    def _calculate_avg_holding_period(trades_df: pd.DataFrame) -> float:
        """Calculate average holding period in days"""
        if len(trades_df) < 2:
            return 0.0
        
        # Group by symbol and calculate holding periods
        holding_periods = []
        
        for symbol in trades_df['symbol'].unique():
            symbol_trades = trades_df[trades_df['symbol'] == symbol].sort_values('timestamp')
            
            # Simple approach: time between consecutive opposite trades
            for i in range(len(symbol_trades) - 1):
                current_trade = symbol_trades.iloc[i]
                next_trade = symbol_trades.iloc[i + 1]
                
                if current_trade['signal'] != next_trade['signal']:
                    time_diff = (next_trade['timestamp'] - current_trade['timestamp']).total_seconds() / 86400  # days
                    holding_periods.append(time_diff)
        
        return np.mean(holding_periods) if holding_periods else 0.0

class EnhancedReporting:
    """Enhanced reporting system with all improvements"""
    
    def __init__(self, db_path: str = "trades.db", auto_backup: bool = True):
        self.db = DatabaseManager(db_path)
        self.position_tracker = PositionTracker()
        self.auto_backup = auto_backup
        self.logger = logging.getLogger(__name__)
        
        # Load existing data
        self._load_existing_data()
        
        # Setup auto-backup
        if auto_backup:
            self._setup_auto_backup()
    
    def _load_existing_data(self):
        """Load existing trades and positions from database"""
        try:
            # Load positions
            positions = self.db.load_positions()
            for symbol, position in positions.items():
                self.position_tracker.positions[symbol] = position
            
            # Load realized P&L history
            with sqlite3.connect(self.db.db_path) as conn:
                pnl_df = pd.read_sql_query("SELECT * FROM realized_pnl", conn, parse_dates=['timestamp'])
                self.position_tracker.realized_pnl = pnl_df.to_dict('records') if not pnl_df.empty else []
            
            self.logger.info("Loaded existing data from database")
        except Exception as e:
            self.logger.error(f"Error loading existing data: {str(e)}")
    
    def _setup_auto_backup(self):
        """Setup automatic CSV backup"""
        try:
            backup_dir = Path("backups")
            backup_dir.mkdir(exist_ok=True)
            
            # Create daily backup
            today = datetime.now().strftime("%Y%m%d")
            backup_file = backup_dir / f"trades_backup_{today}.csv"
            
            if not backup_file.exists():
                self.export_to_csv(str(backup_file))
        except Exception as e:
            self.logger.error(f"Error setting up auto-backup: {str(e)}")
    
    def log_trade(self, trade_info: Union[Dict, TradeInfo], symbol: str, order: Dict = None, mode: str = None):
        """Log a trade with enhanced validation and P&L calculation"""
        try:
            # Convert to TradeInfo if needed
            if isinstance(trade_info, dict):
                trade_info = TradeInfo(**trade_info)
            
            # Process trade through position tracker
            pnl_info = self.position_tracker.process_trade(symbol, trade_info)
            
            # Create trade record
            trade_record = {
                'timestamp': trade_info.timestamp,
                'symbol': symbol,
                'signal': trade_info.signal,
                'price': trade_info.price,
                'quantity': trade_info.quantity,
                'side': trade_info.signal,  # 'buy' or 'sell'
                'mode': mode,
                'order_id': trade_info.order_id,
                'commission': trade_info.commission,
                'slippage': trade_info.slippage,
                'realized_pnl': pnl_info['realized_pnl'],
                'position_after': pnl_info['position_after'],
                'avg_price_after': pnl_info['avg_price_after']
            }
            
            # Save to database
            self.db.save_trade(trade_record)
            self.db.save_position(symbol, self.position_tracker.positions[symbol])
            
            # Save realized P&L if any
            if pnl_info['realized_pnl'] != 0:
                pnl_record = self.position_tracker.realized_pnl[-1]  # Latest P&L record
                self.db.save_realized_pnl(pnl_record)
            
            self.logger.info(f"Logged trade: {symbol} {trade_info.signal} {trade_info.quantity}@{trade_info.price}, P&L: {pnl_info['realized_pnl']:.2f}")
            
            # Auto backup if enabled
            if self.auto_backup and len(self.get_trades()) % 100 == 0:  # Backup every 100 trades
                self._setup_auto_backup()
                
        except Exception as e:
            self.logger.error(f"Error logging trade: {str(e)}")
            raise
    
    def get_trades(self, mode: str = None, symbol: str = None, start_date: str = None, end_date: str = None, limit: int = None) -> pd.DataFrame:
        """Get trades with filtering"""
        try:
            df = self.db.load_trades()
            
            if df.empty:
                return df
            
            # Apply filters
            if mode:
                df = df[df['mode'] == mode]
            if symbol:
                df = df[df['symbol'] == symbol]
            if start_date:
                df = df[df['timestamp'] >= pd.to_datetime(start_date)]
            if end_date:
                df = df[df['timestamp'] <= pd.to_datetime(end_date)]
            if limit:
                df = df.tail(limit)
            
            return df.sort_values('timestamp')
            
        except Exception as e:
            self.logger.error(f"Error getting trades: {str(e)}")
            return pd.DataFrame()
    
    def get_positions(self) -> Dict:
        """Get current positions"""
        return self.position_tracker.get_current_positions()
    
    def get_realized_pnl(self) -> List[Dict]:
        """Get realized P&L history"""
        return self.position_tracker.get_realized_pnl_history()
    
    def get_comprehensive_stats(self, mode: str = None, symbol: str = None, start_date: str = None, end_date: str = None) -> Dict:
        """Get comprehensive trading statistics"""
        try:
            trades_df = self.get_trades(mode, symbol, start_date, end_date)
            realized_pnl = self.get_realized_pnl()
            
            # Filter realized P&L by date if needed
            if start_date or end_date:
                filtered_pnl = []
                for pnl in realized_pnl:
                    pnl_date = pnl['timestamp']
                    if isinstance(pnl_date, str):
                        pnl_date = pd.to_datetime(pnl_date)
                    
                    if start_date and pnl_date < pd.to_datetime(start_date):
                        continue
                    if end_date and pnl_date > pd.to_datetime(end_date):
                        continue
                    
                    filtered_pnl.append(pnl)
                realized_pnl = filtered_pnl
            
            return AdvancedStatistics.calculate_comprehensive_stats(trades_df, realized_pnl)
            
        except Exception as e:
            self.logger.error(f"Error calculating comprehensive stats: {str(e)}")
            return AdvancedStatistics._empty_stats()
    
    def export_to_csv(self, filename: str, mode: str = None, symbol: str = None, start_date: str = None, end_date: str = None):
        """Export trades to CSV with additional data"""
        try:
            df = self.get_trades(mode, symbol, start_date, end_date)
            
            if not df.empty:
                # Add calculated fields
                df['cumulative_pnl'] = df['realized_pnl'].cumsum()
                
            df.to_csv(filename, index=False)
            self.logger.info(f"Exported {len(df)} trades to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to CSV: {str(e)}")
    
    def export_to_json(self, filename: str, mode: str = None, symbol: str = None, start_date: str = None, end_date: str = None):
        """Export comprehensive data to JSON"""
        try:
            trades_df = self.get_trades(mode, symbol, start_date, end_date)
            stats = self.get_comprehensive_stats(mode, symbol, start_date, end_date)
            positions = self.get_positions()
            realized_pnl = self.get_realized_pnl()
            
            export_data = {
                'export_timestamp': datetime.now().isoformat(),
                'filters': {
                    'mode': mode,
                    'symbol': symbol,
                    'start_date': start_date,
                    'end_date': end_date
                },
                'trades': trades_df.to_dict('records') if not trades_df.empty else [],
                'positions': positions,
                'realized_pnl': realized_pnl,
                'statistics': stats
            }
            
            with open(filename, 'w') as f:
                json.dump(export_data, f, indent=2, default=str)
            
            self.logger.info(f"Exported comprehensive data to {filename}")
            
        except Exception as e:
            self.logger.error(f"Error exporting to JSON: {str(e)}")
    
    def generate_charts(self, output_dir: str = "charts", mode: str = None, symbol: str = None):
        """Generate trading charts"""
        if not MATPLOTLIB_AVAILABLE:
            self.logger.warning("Matplotlib not available. Cannot generate charts.")
            return
        
        try:
            Path(output_dir).mkdir(exist_ok=True)
            
            trades_df = self.get_trades(mode, symbol)
            realized_pnl = self.get_realized_pnl()
            
            if trades_df.empty:
                self.logger.warning("No trades to chart")
                return
            
            # Set up the plotting style
            plt.style.use('seaborn-v0_8')
            
            # 1. P&L Over Time
            if realized_pnl:
                pnl_df = pd.DataFrame(realized_pnl)
                pnl_df['timestamp'] = pd.to_datetime(pnl_df['timestamp'])
                pnl_df['cumulative_pnl'] = pnl_df['realized_pnl'].cumsum()
                
                fig, (ax1, ax2) = plt.subplots(2, 1, figsize=(12, 10))
                
                # Cumulative P&L
                ax1.plot(pnl_df['timestamp'], pnl_df['cumulative_pnl'], linewidth=2)
                ax1.set_title('Cumulative P&L Over Time')
                ax1.set_ylabel('Cumulative P&L ($)')
                ax1.grid(True, alpha=0.3)
                ax1.xaxis.set_major_formatter(mdates.DateFormatter('%Y-%m-%d'))
                ax1.tick_params(axis='x', rotation=45)
                
                # Individual trade P&L
                colors = ['green' if pnl > 0 else 'red' for pnl in pnl_df['realized_pnl']]
                ax2.bar(range(len(pnl_df)), pnl_df['realized_pnl'], color=colors, alpha=0.7)
                ax2.set_title('Individual Trade P&L')
                ax2.set_xlabel('Trade Number')
                ax2.set_ylabel('P&L ($)')
                ax2.grid(True, alpha=0.3)
                ax2.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/pnl_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 2. Trade Distribution by Symbol
            symbol_counts = trades_df['symbol'].value_counts()
            if len(symbol_counts) > 1:
                plt.figure(figsize=(10, 6))
                symbol_counts.plot(kind='bar')
                plt.title('Trade Distribution by Symbol')
                plt.xlabel('Symbol')
                plt.ylabel('Number of Trades')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/trade_distribution.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 3. Monthly Performance
            if realized_pnl:
                pnl_df['month'] = pnl_df['timestamp'].dt.to_period('M')
                monthly_pnl = pnl_df.groupby('month')['realized_pnl'].sum()
                
                plt.figure(figsize=(12, 6))
                colors = ['green' if pnl > 0 else 'red' for pnl in monthly_pnl.values]
                monthly_pnl.plot(kind='bar', color=colors, alpha=0.7)
                plt.title('Monthly P&L Performance')
                plt.xlabel('Month')
                plt.ylabel('P&L ($)')
                plt.xticks(rotation=45)
                plt.grid(True, alpha=0.3)
                plt.axhline(y=0, color='black', linestyle='-', alpha=0.5)
                plt.tight_layout()
                plt.savefig(f"{output_dir}/monthly_performance.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            # 4. Drawdown Analysis
            if realized_pnl and len(realized_pnl) > 1:
                cumulative_pnl = pnl_df['cumulative_pnl']
                running_max = cumulative_pnl.cummax()
                drawdown = cumulative_pnl - running_max
                
                plt.figure(figsize=(12, 8))
                
                plt.subplot(2, 1, 1)
                plt.plot(pnl_df['timestamp'], cumulative_pnl, label='Cumulative P&L', linewidth=2)
                plt.plot(pnl_df['timestamp'], running_max, label='Running Maximum', linestyle='--', alpha=0.7)
                plt.title('Cumulative P&L vs Running Maximum')
                plt.ylabel('P&L ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.subplot(2, 1, 2)
                plt.fill_between(pnl_df['timestamp'], drawdown, 0, color='red', alpha=0.3, label='Drawdown')
                plt.plot(pnl_df['timestamp'], drawdown, color='red', linewidth=1)
                plt.title('Drawdown Analysis')
                plt.xlabel('Date')
                plt.ylabel('Drawdown ($)')
                plt.legend()
                plt.grid(True, alpha=0.3)
                
                plt.tight_layout()
                plt.savefig(f"{output_dir}/drawdown_analysis.png", dpi=300, bbox_inches='tight')
                plt.close()
            
            self.logger.info(f"Generated charts in {output_dir}")
            
        except Exception as e:
            self.logger.error(f"Error generating charts: {str(e)}")
    
    def export_to_pdf(self, filename: str, mode: str = None, symbol: str = None, start_date: str = None, end_date: str = None):
        """Export comprehensive PDF report"""
        if not REPORTLAB_AVAILABLE:
            self.logger.warning("ReportLab not available. Cannot generate PDF report.")
            return
        
        try:
            # Generate charts first
            charts_dir = "temp_charts"
            self.generate_charts(charts_dir, mode, symbol)
            
            # Get data
            trades_df = self.get_trades(mode, symbol, start_date, end_date)
            stats = self.get_comprehensive_stats(mode, symbol, start_date, end_date)
            positions = self.get_positions()
            
            # Create PDF
            doc = SimpleDocTemplate(filename, pagesize=A4)
            styles = getSampleStyleSheet()
            elements = []
            
            # Title
            title_style = ParagraphStyle(
                'CustomTitle',
                parent=styles['Heading1'],
                fontSize=24,
                spaceAfter=30,
                alignment=1  # Center alignment
            )
            elements.append(Paragraph("Trading Performance Report", title_style))
            elements.append(Spacer(1, 20))
            
            # Report metadata
            metadata = f"""
            <b>Report Generated:</b> {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}<br/>
            <b>Period:</b> {start_date or 'All time'} to {end_date or 'Present'}<br/>
            <b>Mode Filter:</b> {mode or 'All modes'}<br/>
            <b>Symbol Filter:</b> {symbol or 'All symbols'}
            """
            elements.append(Paragraph(metadata, styles['Normal']))
            elements.append(Spacer(1, 20))
            
            # Executive Summary
            elements.append(Paragraph("Executive Summary", styles['Heading2']))
            
            summary_data = [
                ['Metric', 'Value'],
                ['Total Trades', f"{stats['total_trades']:,}"],
                ['Total Realized P&L', f"${stats['total_realized_pnl']:,.2f}"],
                ['Win Rate', f"{stats['win_rate']:.1f}%"],
                ['Profit Factor', f"{stats['profit_factor']:.2f}"],
                ['Sharpe Ratio', f"{stats['sharpe_ratio']:.2f}"],
                ['Max Drawdown', f"${stats['max_drawdown']:,.2f} ({stats['max_drawdown_pct']:.1f}%)"],
            ]
            
            summary_table = Table(summary_data, colWidths=[2*inch, 2*inch])
            summary_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 14),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(summary_table)
            elements.append(Spacer(1, 20))
            
            # Detailed Statistics
            elements.append(Paragraph("Detailed Statistics", styles['Heading2']))
            
            detailed_data = [
                ['Metric', 'Value'],
                ['Average Win', f"${stats['avg_win']:,.2f}"],
                ['Average Loss', f"${stats['avg_loss']:,.2f}"],
                ['Sortino Ratio', f"{stats['sortino_ratio']:.2f}"],
                ['Calmar Ratio', f"{stats['calmar_ratio']:.2f}"],
                ['Max Consecutive Wins', f"{stats['max_consecutive_wins']}"],
                ['Max Consecutive Losses', f"{stats['max_consecutive_losses']}"],
                ['Current Streak', f"{stats['current_streak']}"],
                ['Trades per Day', f"{stats['trades_per_day']:.2f}"],
                ['Total Commission Paid', f"${stats['total_commission_paid']:,.2f}"],
                ['Average Holding Period', f"{stats['avg_holding_period']:.1f} days"],
            ]
            
            detailed_table = Table(detailed_data, colWidths=[2*inch, 2*inch])
            detailed_table.setStyle(TableStyle([
                ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                ('FONTSIZE', (0, 0), (-1, 0), 12),
                ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                ('GRID', (0, 0), (-1, -1), 1, colors.black)
            ]))
            
            elements.append(detailed_table)
            elements.append(Spacer(1, 20))
            
            # Current Positions
            if positions:
                elements.append(Paragraph("Current Positions", styles['Heading2']))
                
                position_data = [['Symbol', 'Quantity', 'Avg Price', 'Market Value']]
                for symbol, pos in positions.items():
                    if pos['quantity'] != 0:
                        market_value = pos['quantity'] * pos['avg_price']
                        position_data.append([
                            symbol,
                            f"{pos['quantity']:.2f}",
                            f"${pos['avg_price']:.2f}",
                            f"${market_value:,.2f}"
                        ])
                
                if len(position_data) > 1:
                    position_table = Table(position_data, colWidths=[1.5*inch, 1.5*inch, 1.5*inch, 1.5*inch])
                    position_table.setStyle(TableStyle([
                        ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                        ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                        ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                        ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                        ('FONTSIZE', (0, 0), (-1, 0), 12),
                        ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                        ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                        ('GRID', (0, 0), (-1, -1), 1, colors.black)
                    ]))
                    elements.append(position_table)
                else:
                    elements.append(Paragraph("No open positions", styles['Normal']))
                
                elements.append(Spacer(1, 20))
            
            # Add charts if available
            chart_files = [
                ("temp_charts/pnl_analysis.png", "P&L Analysis"),
                ("temp_charts/trade_distribution.png", "Trade Distribution"),
                ("temp_charts/monthly_performance.png", "Monthly Performance"),
                ("temp_charts/drawdown_analysis.png", "Drawdown Analysis")
            ]
            
            for chart_file, chart_title in chart_files:
                if Path(chart_file).exists():
                    elements.append(PageBreak())
                    elements.append(Paragraph(chart_title, styles['Heading2']))
                    elements.append(Spacer(1, 12))
                    
                    # Add image
                    img = Image(chart_file, width=6*inch, height=4*inch)
                    elements.append(img)
                    elements.append(Spacer(1, 12))
            
            # Recent Trades Table
            if not trades_df.empty:
                elements.append(PageBreak())
                elements.append(Paragraph("Recent Trades", styles['Heading2']))
                
                # Show last 20 trades
                recent_trades = trades_df.tail(20)
                trade_data = [['Date', 'Symbol', 'Side', 'Quantity', 'Price', 'P&L']]
                
                for _, trade in recent_trades.iterrows():
                    trade_data.append([
                        trade['timestamp'].strftime('%Y-%m-%d %H:%M'),
                        trade['symbol'],
                        trade['side'].upper(),
                        f"{trade['quantity']:.2f}",
                        f"${trade['price']:.2f}",
                        f"${trade['realized_pnl']:.2f}" if trade['realized_pnl'] != 0 else "-"
                    ])
                
                trade_table = Table(trade_data, colWidths=[1.2*inch, 0.8*inch, 0.6*inch, 0.8*inch, 0.8*inch, 0.8*inch])
                trade_table.setStyle(TableStyle([
                    ('BACKGROUND', (0, 0), (-1, 0), colors.grey),
                    ('TEXTCOLOR', (0, 0), (-1, 0), colors.whitesmoke),
                    ('ALIGN', (0, 0), (-1, -1), 'CENTER'),
                    ('FONTNAME', (0, 0), (-1, 0), 'Helvetica-Bold'),
                    ('FONTSIZE', (0, 0), (-1, -1), 8),
                    ('BOTTOMPADDING', (0, 0), (-1, 0), 12),
                    ('BACKGROUND', (0, 1), (-1, -1), colors.beige),
                    ('GRID', (0, 0), (-1, -1), 1, colors.black)
                ]))
                
                elements.append(trade_table)
            
            # Build PDF
            doc.build(elements)
            
            # Cleanup temp charts
            import shutil
            if Path(charts_dir).exists():
                shutil.rmtree(charts_dir)
            
            self.logger.info(f"Generated comprehensive PDF report: {filename}")
            
        except Exception as e:
            self.logger.error(f"Error generating PDF report: {str(e)}")
    
    def get_performance_metrics(self, benchmark_returns: List[float] = None) -> Dict:
        """Get advanced performance metrics with optional benchmark comparison"""
        try:
            realized_pnl = self.get_realized_pnl()
            
            if not realized_pnl:
                return {}
            
            pnl_df = pd.DataFrame(realized_pnl)
            returns = pnl_df['realized_pnl'].values
            
            metrics = {
                'total_return': np.sum(returns),
                'volatility': np.std(returns) * np.sqrt(252),  # Annualized
                'skewness': float(pd.Series(returns).skew()),
                'kurtosis': float(pd.Series(returns).kurtosis()),
                'var_95': np.percentile(returns, 5),  # Value at Risk (95%)
                'cvar_95': np.mean(returns[returns <= np.percentile(returns, 5)]),  # Conditional VaR
            }
            
            # Benchmark comparison if provided
            if benchmark_returns and len(benchmark_returns) == len(returns):
                benchmark_returns = np.array(benchmark_returns)
                
                # Beta calculation
                covariance = np.cov(returns, benchmark_returns)[0, 1]
                benchmark_variance = np.var(benchmark_returns)
                beta = covariance / benchmark_variance if benchmark_variance != 0 else 0
                
                # Alpha calculation (assuming risk-free rate = 0)
                portfolio_return = np.mean(returns)
                benchmark_return = np.mean(benchmark_returns)
                alpha = portfolio_return - (beta * benchmark_return)
                
                # Information ratio
                excess_returns = returns - benchmark_returns
                information_ratio = np.mean(excess_returns) / np.std(excess_returns) if np.std(excess_returns) != 0 else 0
                
                metrics.update({
                    'beta': beta,
                    'alpha': alpha,
                    'information_ratio': information_ratio,
                    'correlation': np.corrcoef(returns, benchmark_returns)[0, 1]
                })
            
            return metrics
            
        except Exception as e:
            self.logger.error(f"Error calculating performance metrics: {str(e)}")
            return {}
    
    def run_backtest_analysis(self, symbol: str, strategy_params: Dict = None) -> Dict:
        """Run backtest analysis on historical trades for a specific symbol"""
        try:
            trades_df = self.get_trades(symbol=symbol)
            
            if trades_df.empty:
                return {'error': 'No trades found for symbol'}
            
            # Group trades by day for daily analysis
            trades_df['date'] = trades_df['timestamp'].dt.date
            daily_trades = trades_df.groupby('date').agg({
                'realized_pnl': 'sum',
                'quantity': 'sum',
                'commission': 'sum'
            }).reset_index()
            
            # Calculate key backtest metrics
            total_days = len(daily_trades)
            profitable_days = len(daily_trades[daily_trades['realized_pnl'] > 0])
            
            analysis = {
                'symbol': symbol,
                'total_trading_days': total_days,
                'profitable_days': profitable_days,
                'profitable_day_ratio': profitable_days / total_days if total_days > 0 else 0,
                'avg_daily_pnl': daily_trades['realized_pnl'].mean(),
                'best_day': daily_trades['realized_pnl'].max(),
                'worst_day': daily_trades['realized_pnl'].min(),
                'daily_volatility': daily_trades['realized_pnl'].std(),
                'total_volume_traded': trades_df['quantity'].sum(),
                'total_commission_paid': trades_df['commission'].sum(),
                'commission_as_pct_of_pnl': (trades_df['commission'].sum() / abs(trades_df['realized_pnl'].sum())) * 100 if trades_df['realized_pnl'].sum() != 0 else 0
            }
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Error running backtest analysis: {str(e)}")
            return {'error': str(e)}

# Example usage and testing
if __name__ == "__main__":
    # Setup logging
    logging.basicConfig(level=logging.INFO)
    
    # Create enhanced reporting instance
    reporter = EnhancedReporting()
    
    # Example trades
    example_trades = [
        {'signal': 'buy', 'price': 100.0, 'quantity': 10, 'commission': 1.0},
        {'signal': 'sell', 'price': 105.0, 'quantity': 5, 'commission': 1.0},
        {'signal': 'sell', 'price': 102.0, 'quantity': 5, 'commission': 1.0},
        {'signal': 'buy', 'price': 98.0, 'quantity': 15, 'commission': 1.5},
        {'signal': 'sell', 'price': 103.0, 'quantity': 15, 'commission': 1.5},
    ]
    
    # Log example trades
    for i, trade in enumerate(example_trades):
        try:
            reporter.log_trade(trade, 'AAPL', mode='paper_trading')
            print(f"Logged trade {i+1}")
        except Exception as e:
            print(f"Error logging trade {i+1}: {e}")
    
    # Get comprehensive statistics
    stats = reporter.get_comprehensive_stats()
    print("\nComprehensive Statistics:")
    for key, value in stats.items():
        if isinstance(value, float):
            print(f"{key}: {value:.4f}")
        else:
            print(f"{key}: {value}")
    
    # Get current positions
    positions = reporter.get_positions()
    print(f"\nCurrent Positions: {positions}")
    
    # Export data
    reporter.export_to_csv("enhanced_trades.csv")
    reporter.export_to_json("trading_report.json")
    reporter.generate_charts()
    reporter.export_to_pdf("comprehensive_report.pdf")
    
    print("\nEnhanced reporting system demonstration complete!")
    print("Files generated:")
    print("- enhanced_trades.csv")
    print("- trading_report.json") 
    print("- comprehensive_report.pdf")
    print("- charts/ directory with performance charts")