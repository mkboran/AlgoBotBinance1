#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - PORTFOLIO LOGGER FIX
💎 FIXED: Portfolio sınıfına logger özniteliği eklendi

ÇÖZÜMLER:
1. ✅ Portfolio.__init__'e logger eklendi
2. ✅ Tüm log mesajları self.logger kullanıyor
3. ✅ Position sınıfına da logger eklendi
"""

import logging
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Any, Tuple
from dataclasses import dataclass, field, asdict
import pandas as pd
import numpy as np
import json
from uuid import uuid4

# Configure module logger
logger = logging.getLogger("algobot.portfolio")


@dataclass
class Position:
    """
    💎 ULTRA ADVANCED POSITION TRACKING
    🚀 Hedge Fund Level Position Management
    """
    
    position_id: str
    strategy_name: str
    symbol: str
    quantity_btc: float
    entry_price: float
    entry_cost_usdt_total: float
    timestamp: str
    
    # Optional fields with defaults
    stop_loss_price: Optional[float] = None
    take_profit_price: Optional[float] = None
    trailing_stop_distance: Optional[float] = None
    
    # Performance tracking
    highest_price_seen: float = field(init=False)
    lowest_price_seen: float = field(init=False)
    unrealized_pnl: float = 0.0
    unrealized_pnl_pct: float = 0.0
    
    # Risk metrics
    position_risk_usdt: float = field(init=False)
    position_risk_pct: float = field(init=False)
    
    # Advanced fields
    entry_context: Dict[str, Any] = field(default_factory=dict)
    exit_context: Dict[str, Any] = field(default_factory=dict)
    
    # ML predictions
    ml_exit_prediction: Optional[float] = None
    ml_confidence: Optional[float] = None
    
    # Logger
    logger: logging.Logger = field(default=None, init=False, repr=False)
    
    def __post_init__(self):
        """Initialize calculated fields and logger"""
        # ✅ LOGGER INITIALIZATION
        self.logger = logging.getLogger(f"algobot.portfolio.position.{self.position_id}")
        
        # Initialize price tracking
        self.highest_price_seen = self.entry_price
        self.lowest_price_seen = self.entry_price
        
        # Calculate initial risk
        if self.stop_loss_price:
            self.position_risk_usdt = abs(self.entry_price - self.stop_loss_price) * self.quantity_btc
            self.position_risk_pct = abs(self.entry_price - self.stop_loss_price) / self.entry_price
        else:
            self.position_risk_usdt = self.entry_cost_usdt_total * 0.02  # Default 2% risk
            self.position_risk_pct = 0.02
        
        # Extract quality score from context if available
        self.quality_score = self.entry_context.get('quality_score', 0)
        
        self.logger.debug(f"Position initialized: {self.symbol} - {self.quantity_btc} BTC @ {self.entry_price}")
    
    def update_performance_metrics(self, current_price: float) -> Dict[str, float]:
        """Update position performance metrics"""
        # Update price tracking
        self.highest_price_seen = max(self.highest_price_seen, current_price)
        self.lowest_price_seen = min(self.lowest_price_seen, current_price)
        
        # Calculate PnL
        current_value = current_price * self.quantity_btc
        self.unrealized_pnl = current_value - self.entry_cost_usdt_total
        self.unrealized_pnl_pct = (current_value - self.entry_cost_usdt_total) / self.entry_cost_usdt_total
        
        # Return metrics
        return {
            'current_profit': self.unrealized_pnl,
            'current_profit_pct': self.unrealized_pnl_pct * 100,
            'highest_price': self.highest_price_seen,
            'lowest_price': self.lowest_price_seen,
            'current_value': current_value
        }
    
    def should_trail_stop(self, current_price: float, activation_pct: float = 0.02) -> Tuple[bool, Optional[float]]:
        """Check if trailing stop should be activated or updated"""
        if not self.trailing_stop_distance:
            # Check if profit is enough to activate trailing stop
            profit_pct = (current_price - self.entry_price) / self.entry_price
            if profit_pct >= activation_pct:
                self.logger.info(f"Trailing stop activated at {profit_pct:.2%} profit")
                return True, current_price * 0.98  # 2% trailing stop
            return False, None
        
        # Update existing trailing stop
        new_stop = current_price * (1 - self.trailing_stop_distance)
        if new_stop > self.stop_loss_price:
            self.logger.info(f"Trailing stop updated: {self.stop_loss_price} -> {new_stop}")
            return True, new_stop
        
        return False, None
    
    def to_dict(self) -> Dict[str, Any]:
        """Convert position to dictionary"""
        return {
            'position_id': self.position_id,
            'strategy_name': self.strategy_name,
            'symbol': self.symbol,
            'quantity_btc': self.quantity_btc,
            'entry_price': self.entry_price,
            'entry_cost_usdt_total': self.entry_cost_usdt_total,
            'timestamp': self.timestamp,
            'stop_loss_price': self.stop_loss_price,
            'take_profit_price': self.take_profit_price,
            'unrealized_pnl': self.unrealized_pnl,
            'unrealized_pnl_pct': self.unrealized_pnl_pct,
            'highest_price_seen': self.highest_price_seen,
            'lowest_price_seen': self.lowest_price_seen,
            'quality_score': self.quality_score
        }
    
    def __repr__(self):
        return f"Position({self.position_id}, {self.symbol}, {self.quantity_btc} BTC, PnL: {self.unrealized_pnl:.2f} USDT)"


class Portfolio:
    """
    💼 ULTRA ADVANCED PORTFOLIO MANAGEMENT SYSTEM
    🚀 Institutional Grade Portfolio Tracking
    
    Features:
    - Multi-strategy position management
    - Real-time performance tracking
    - Risk analytics and controls
    - Trade history and analytics
    - Advanced position sizing
    """
    
    def __init__(self, initial_capital_usdt: float = 10000.0):
        """Initialize portfolio with starting capital"""
        
        # ✅ LOGGER INITIALIZATION - FIXED
        self.logger = logging.getLogger("algobot.portfolio")
        
        # Capital management
        self.initial_capital_usdt = initial_capital_usdt
        self.available_usdt = initial_capital_usdt
        
        # Position tracking
        self.positions: List[Position] = []
        self.closed_trades: List[Dict[str, Any]] = []
        
        # Performance tracking
        self.cumulative_pnl = 0.0
        self.total_fees_paid = 0.0
        
        # Risk metrics
        self.max_portfolio_risk_pct = 0.06  # 6% max portfolio heat
        self.max_position_risk_pct = 0.02   # 2% max per position
        self.max_correlation_risk = 0.7     # Max correlation between positions
        
        # Performance history
        self.equity_curve: List[Dict[str, Any]] = []
        self.drawdown_series: List[float] = []
        self.peak_equity = initial_capital_usdt
        
        # Strategy performance tracking
        self.strategy_performance: Dict[str, Dict[str, Any]] = {}
        
        # Analytics
        self.trade_analytics = {
            'total_trades': 0,
            'winning_trades': 0,
            'losing_trades': 0,
            'largest_win': 0.0,
            'largest_loss': 0.0,
            'consecutive_wins': 0,
            'consecutive_losses': 0,
            'current_streak': 0
        }
        
        self.logger.info(f"💼 Portfolio initialized with ${initial_capital_usdt:,.2f}")
    
    def get_available_usdt(self) -> float:
        """Get available USDT balance"""
        return self.available_usdt
    
    def get_total_portfolio_value_usdt(self, current_btc_price: float) -> float:
        """Calculate total portfolio value including positions"""
        if current_btc_price <= 0:
            raise ValueError(f"Invalid BTC price: {current_btc_price}")
        
        # Calculate position values
        position_value = sum(pos.quantity_btc * current_btc_price for pos in self.positions)
        
        # Total value = available cash + position values
        total_value = self.available_usdt + position_value
        
        # Update equity curve
        self._update_equity_curve(total_value, current_btc_price)
        
        return total_value
    
    def get_open_positions(self, 
                          strategy_name: Optional[str] = None,
                          symbol: Optional[str] = None) -> List[Position]:
        """Get open positions with optional filtering"""
        positions = self.positions
        
        if strategy_name:
            positions = [p for p in positions if p.strategy_name == strategy_name]
        
        if symbol:
            positions = [p for p in positions if p.symbol == symbol]
        
        return positions
    
    async def execute_buy(self,
                         strategy_name: str,
                         symbol: str,
                         current_price: float,
                         timestamp: str,
                         reason: str,
                         amount_usdt_override: Optional[float] = None,
                         stop_loss_price: Optional[float] = None,
                         take_profit_price: Optional[float] = None,
                         metadata: Optional[Dict[str, Any]] = None) -> Optional[Position]:
        """Execute buy order with comprehensive tracking"""
        
        try:
            # Determine position size
            if amount_usdt_override:
                position_size_usdt = amount_usdt_override
            else:
                position_size_usdt = self._calculate_position_size(strategy_name, current_price)
            
            # Check available balance
            if position_size_usdt > self.available_usdt:
                self.logger.warning(f"Insufficient balance: Need ${position_size_usdt:.2f}, Have ${self.available_usdt:.2f}")
                return None
            
            # Check risk limits
            if not self._check_risk_limits(position_size_usdt, current_price):
                self.logger.warning("Risk limits exceeded, skipping trade")
                return None
            
            # Calculate fees
            fee = position_size_usdt * 0.001  # 0.1% fee
            total_cost = position_size_usdt + fee
            
            if total_cost > self.available_usdt:
                self.logger.warning("Insufficient balance after fees")
                return None
            
            # Calculate BTC quantity
            quantity_btc = position_size_usdt / current_price
            
            # Create position
            position = Position(
                position_id=f"{strategy_name}_{symbol}_{uuid4().hex[:8]}",
                strategy_name=strategy_name,
                symbol=symbol,
                quantity_btc=quantity_btc,
                entry_price=current_price,
                entry_cost_usdt_total=position_size_usdt,
                timestamp=timestamp,
                stop_loss_price=stop_loss_price,
                take_profit_price=take_profit_price,
                entry_context={
                    'reason': reason,
                    'metadata': metadata or {},
                    'fee_paid': fee
                }
            )
            
            # Update portfolio
            self.available_usdt -= total_cost
            self.total_fees_paid += fee
            self.positions.append(position)
            
            # Update strategy tracking
            self._update_strategy_tracking(strategy_name, 'buy', position_size_usdt)
            
            self.logger.info(f"✅ BUY executed: {quantity_btc:.6f} {symbol} @ ${current_price:,.2f}")
            self.logger.info(f"   Position: {position.position_id}")
            self.logger.info(f"   Cost: ${position_size_usdt:.2f} + ${fee:.2f} fee")
            self.logger.info(f"   Reason: {reason}")
            
            return position
            
        except Exception as e:
            self.logger.error(f"❌ Buy execution error: {e}")
            return None
    
    async def execute_sell(self,
                          position_to_close: Position,
                          current_price: float,
                          timestamp: str,
                          reason: str,
                          partial_pct: float = 1.0) -> bool:
        """Execute sell order with PnL tracking"""
        
        try:
            if position_to_close not in self.positions:
                self.logger.error(f"Position {position_to_close.position_id} not found")
                return False
            
            # Calculate sell value
            sell_quantity = position_to_close.quantity_btc * partial_pct
            gross_value = sell_quantity * current_price
            fee = gross_value * 0.001  # 0.1% fee
            net_value = gross_value - fee
            
            # Calculate PnL
            cost_basis = position_to_close.entry_cost_usdt_total * partial_pct
            profit = net_value - cost_basis
            profit_pct = (profit / cost_basis) * 100
            
            # Update portfolio
            self.available_usdt += net_value
            self.cumulative_pnl += profit
            self.total_fees_paid += fee
            
            # Create trade record
            trade_record = {
                'position_id': position_to_close.position_id,
                'strategy_name': position_to_close.strategy_name,
                'symbol': position_to_close.symbol,
                'entry_price': position_to_close.entry_price,
                'exit_price': current_price,
                'quantity': sell_quantity,
                'entry_time': position_to_close.timestamp,
                'exit_time': timestamp,
                'profit_usdt': profit,
                'profit_pct': profit_pct,
                'fees_paid': position_to_close.entry_context.get('fee_paid', 0) + fee,
                'reason': reason,
                'quality_score': position_to_close.quality_score
            }
            
            # Handle partial or full close
            if partial_pct >= 1.0:
                # Full close
                self.positions.remove(position_to_close)
            else:
                # Partial close
                position_to_close.quantity_btc *= (1 - partial_pct)
                position_to_close.entry_cost_usdt_total *= (1 - partial_pct)
            
            # Update records
            self.closed_trades.append(trade_record)
            self._update_trade_analytics(profit > 0, profit)
            self._update_strategy_tracking(position_to_close.strategy_name, 'sell', profit)
            
            self.logger.info(f"✅ SELL executed: {sell_quantity:.6f} {position_to_close.symbol} @ ${current_price:,.2f}")
            self.logger.info(f"   Profit: ${profit:.2f} ({profit_pct:.2f}%)")
            self.logger.info(f"   Reason: {reason}")
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Sell execution error: {e}")
            return False
    
    def get_portfolio_metrics(self, current_btc_price: float) -> Dict[str, Any]:
        """Get comprehensive portfolio metrics"""
        
        total_value = self.get_total_portfolio_value_usdt(current_btc_price)
        
        # Calculate returns
        total_return = (total_value - self.initial_capital_usdt) / self.initial_capital_usdt
        
        # Calculate position metrics
        position_value = sum(pos.quantity_btc * current_btc_price for pos in self.positions)
        cash_pct = self.available_usdt / total_value if total_value > 0 else 1.0
        
        # Risk metrics
        current_drawdown = self._calculate_current_drawdown(total_value)
        portfolio_heat = self._calculate_portfolio_heat(current_btc_price)
        
        # Performance metrics
        sharpe_ratio = self._calculate_sharpe_ratio()
        win_rate = self._calculate_win_rate()
        profit_factor = self._calculate_profit_factor()
        
        return {
            # Value metrics
            'total_value': total_value,
            'available_cash': self.available_usdt,
            'position_value': position_value,
            'cash_percentage': cash_pct * 100,
            
            # Return metrics
            'total_return': total_return,
            'cumulative_pnl': self.cumulative_pnl,
            
            # Risk metrics
            'current_drawdown': current_drawdown,
            'max_drawdown': max(self.drawdown_series) if self.drawdown_series else 0,
            'portfolio_heat': portfolio_heat,
            
            # Performance metrics
            'sharpe_ratio': sharpe_ratio,
            'win_rate': win_rate,
            'profit_factor': profit_factor,
            
            # Position metrics
            'open_positions': len(self.positions),
            'total_trades': len(self.closed_trades),
            'fees_paid': self.total_fees_paid,
            
            # Analytics
            'trade_analytics': self.trade_analytics,
            'strategy_performance': self.strategy_performance
        }
    
    def _calculate_position_size(self, strategy_name: str, current_price: float) -> float:
        """Calculate position size based on risk management rules"""
        
        # Base position size (2% of portfolio)
        base_size = self.available_usdt * 0.02
        
        # Adjust based on strategy performance
        if strategy_name in self.strategy_performance:
            perf = self.strategy_performance[strategy_name]
            if perf['total_trades'] > 10:
                # Scale based on win rate
                win_rate = perf['win_rate']
                if win_rate > 0.6:
                    base_size *= 1.2
                elif win_rate < 0.4:
                    base_size *= 0.8
        
        # Ensure minimum and maximum limits
        min_size = 100.0  # $100 minimum
        max_size = self.available_usdt * 0.1  # 10% maximum
        
        return max(min_size, min(base_size, max_size))
    
    def _check_risk_limits(self, position_size: float, current_price: float) -> bool:
        """Check if new position respects risk limits"""
        
        # Check portfolio heat
        current_heat = self._calculate_portfolio_heat(current_price)
        new_heat = current_heat + (position_size / self.get_total_portfolio_value_usdt(current_price))
        
        if new_heat > self.max_portfolio_risk_pct:
            self.logger.warning(f"Portfolio heat would exceed limit: {new_heat:.2%} > {self.max_portfolio_risk_pct:.2%}")
            return False
        
        # Check correlation risk (simplified)
        if len(self.positions) >= 3:
            # Limit highly correlated positions
            btc_positions = len([p for p in self.positions if 'BTC' in p.symbol])
            if btc_positions >= 3:
                self.logger.warning("Too many correlated positions")
                return False
        
        return True
    
    def _update_equity_curve(self, total_value: float, btc_price: float):
        """Update equity curve and drawdown tracking"""
        
        timestamp = datetime.now(timezone.utc)
        
        # Add to equity curve
        self.equity_curve.append({
            'timestamp': timestamp.isoformat(),
            'total_value': total_value,
            'btc_price': btc_price,
            'positions': len(self.positions)
        })
        
        # Update peak and drawdown
        if total_value > self.peak_equity:
            self.peak_equity = total_value
        
        current_drawdown = (self.peak_equity - total_value) / self.peak_equity
        self.drawdown_series.append(current_drawdown)
        
        # Keep only recent history (last 1000 points)
        if len(self.equity_curve) > 1000:
            self.equity_curve = self.equity_curve[-1000:]
            self.drawdown_series = self.drawdown_series[-1000:]
    
    def _update_strategy_tracking(self, strategy_name: str, action: str, value: float):
        """Update strategy-specific performance tracking"""
        
        if strategy_name not in self.strategy_performance:
            self.strategy_performance[strategy_name] = {
                'total_trades': 0,
                'winning_trades': 0,
                'losing_trades': 0,
                'total_profit': 0.0,
                'total_loss': 0.0,
                'win_rate': 0.0,
                'profit_factor': 0.0,
                'largest_win': 0.0,
                'largest_loss': 0.0
            }
        
        perf = self.strategy_performance[strategy_name]
        
        if action == 'sell':
            perf['total_trades'] += 1
            
            if value > 0:
                perf['winning_trades'] += 1
                perf['total_profit'] += value
                perf['largest_win'] = max(perf['largest_win'], value)
            else:
                perf['losing_trades'] += 1
                perf['total_loss'] += abs(value)
                perf['largest_loss'] = min(perf['largest_loss'], value)
            
            # Update metrics
            if perf['total_trades'] > 0:
                perf['win_rate'] = perf['winning_trades'] / perf['total_trades']
            
            if perf['total_loss'] > 0:
                perf['profit_factor'] = perf['total_profit'] / perf['total_loss']
    
    def _update_trade_analytics(self, is_winner: bool, profit: float):
        """Update trade analytics"""
        
        self.trade_analytics['total_trades'] += 1
        
        if is_winner:
            self.trade_analytics['winning_trades'] += 1
            self.trade_analytics['largest_win'] = max(self.trade_analytics['largest_win'], profit)
            
            if self.trade_analytics['current_streak'] >= 0:
                self.trade_analytics['current_streak'] += 1
                self.trade_analytics['consecutive_wins'] = max(
                    self.trade_analytics['consecutive_wins'],
                    self.trade_analytics['current_streak']
                )
            else:
                self.trade_analytics['current_streak'] = 1
        else:
            self.trade_analytics['losing_trades'] += 1
            self.trade_analytics['largest_loss'] = min(self.trade_analytics['largest_loss'], profit)
            
            if self.trade_analytics['current_streak'] <= 0:
                self.trade_analytics['current_streak'] -= 1
                self.trade_analytics['consecutive_losses'] = max(
                    self.trade_analytics['consecutive_losses'],
                    abs(self.trade_analytics['current_streak'])
                )
            else:
                self.trade_analytics['current_streak'] = -1
    
    def _calculate_current_drawdown(self, current_value: float) -> float:
        """Calculate current drawdown from peak"""
        if current_value >= self.peak_equity:
            return 0.0
        return (self.peak_equity - current_value) / self.peak_equity
    
    def _calculate_portfolio_heat(self, current_price: float) -> float:
        """Calculate total portfolio risk exposure"""
        total_risk = 0.0
        
        for position in self.positions:
            # Risk as percentage of portfolio
            position_value = position.quantity_btc * current_price
            position_risk = position.position_risk_pct * position_value
            portfolio_value = self.get_total_portfolio_value_usdt(current_price)
            
            total_risk += position_risk / portfolio_value
        
        return total_risk
    
    def _calculate_sharpe_ratio(self, risk_free_rate: float = 0.02) -> float:
        """Calculate Sharpe ratio from returns"""
        if len(self.equity_curve) < 2:
            return 0.0
        
        # Calculate returns
        returns = []
        for i in range(1, len(self.equity_curve)):
            prev_value = self.equity_curve[i-1]['total_value']
            curr_value = self.equity_curve[i]['total_value']
            returns.append((curr_value - prev_value) / prev_value)
        
        if not returns:
            return 0.0
        
        # Calculate metrics
        returns_array = np.array(returns)
        avg_return = np.mean(returns_array) * 252  # Annualized
        std_return = np.std(returns_array) * np.sqrt(252)
        
        if std_return == 0:
            return 0.0
        
        return (avg_return - risk_free_rate) / std_return
    
    def _calculate_win_rate(self) -> float:
        """Calculate overall win rate"""
        if not self.closed_trades:
            return 0.0
        
        winners = sum(1 for trade in self.closed_trades if trade['profit_usdt'] > 0)
        return winners / len(self.closed_trades)
    
    def _calculate_profit_factor(self) -> float:
        """Calculate profit factor"""
        if not self.closed_trades:
            return 0.0
        
        gross_profit = sum(trade['profit_usdt'] for trade in self.closed_trades if trade['profit_usdt'] > 0)
        gross_loss = abs(sum(trade['profit_usdt'] for trade in self.closed_trades if trade['profit_usdt'] < 0))
        
        if gross_loss == 0:
            return float('inf') if gross_profit > 0 else 0.0
        
        return gross_profit / gross_loss
    
    def save_portfolio_state(self, filepath: str):
        """Save portfolio state to file"""
        state = {
            'timestamp': datetime.now(timezone.utc).isoformat(),
            'initial_capital': self.initial_capital_usdt,
            'available_usdt': self.available_usdt,
            'cumulative_pnl': self.cumulative_pnl,
            'total_fees_paid': self.total_fees_paid,
            'positions': [pos.to_dict() for pos in self.positions],
            'closed_trades': self.closed_trades,
            'trade_analytics': self.trade_analytics,
            'strategy_performance': self.strategy_performance
        }
        
        with open(filepath, 'w') as f:
            json.dump(state, f, indent=2)
        
        self.logger.info(f"Portfolio state saved to {filepath}")
    
    def __repr__(self):
        return (f"Portfolio(capital=${self.available_usdt:,.2f}, "
                f"positions={len(self.positions)}, "
                f"PnL=${self.cumulative_pnl:,.2f})")