#!/usr/bin/env python3
"""
🧠 PROJE PHOENIX - BASE STRATEGY FOUNDATION
💎 Tüm Stratejilerin Ortak Beyni - Hedge Fund Seviyesi

Bu temel sınıf tüm stratejilere şunları sağlar:
1. ✅ Ortak logger ve hata yönetimi
2. ✅ Portfolio interface standardizasyonu  
3. ✅ Performance tracking altyapısı
4. ✅ Risk management foundation
5. ✅ ML integration interface
6. ✅ Sentiment analysis base
7. ✅ Parameter evolution system
8. ✅ Lifecycle management

KALITSAL HIYERARŞI:
BaseStrategy → EnhancedMomentumStrategy
BaseStrategy → BollingerRSIStrategy
BaseStrategy → RSIMLStrategy
BaseStrategy → MACDMLStrategy  
BaseStrategy → VolumeProfileStrategy

📍 DOSYA: base_strategy.py
📁 KONUM: strategies/
🔄 DURUM: kalıcı - tüm stratejilerin temeli
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from collections import deque, defaultdict
import warnings
warnings.filterwarnings('ignore')

from utils.portfolio import Portfolio, Position
from utils.config import settings
from utils.logger import logger


class StrategyState(Enum):
    """Strategy lifecycle states"""
    INITIALIZING = "initializing"
    ACTIVE = "active" 
    PAUSED = "paused"
    ERROR = "error"
    STOPPED = "stopped"


class SignalType(Enum):
    """Trading signal types"""
    BUY = "buy"
    SELL = "sell"
    HOLD = "hold"
    CLOSE = "close"


@dataclass
class TradingSignal:
    """Trading signal data structure"""
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    reasons: List[str] = field(default_factory=list)
    metadata: Dict[str, Any] = field(default_factory=dict)


@dataclass
class StrategyMetrics:
    """Strategy performance metrics"""
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    total_profit_usdt: float = 0.0
    total_return_pct: float = 0.0
    win_rate_pct: float = 0.0
    avg_profit_per_trade: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))


class BaseStrategy(ABC):
    """🧠 Phoenix Base Strategy - Foundation for All Trading Strategies"""
    
    def __init__(
        self,
        portfolio: Portfolio,
        symbol: str = "BTC/USDT",
        strategy_name: str = "BaseStrategy",
        **kwargs
    ):
        """
        Initialize base strategy with common functionality
        
        Args:
            portfolio: Portfolio instance for trade execution
            symbol: Trading pair symbol
            strategy_name: Unique strategy identifier
            **kwargs: Additional strategy-specific parameters
        """
        # Core identification
        self.strategy_name = strategy_name
        self.symbol = symbol
        self.portfolio = portfolio
        
        # Strategy state management
        self.state = StrategyState.INITIALIZING
        self.created_at = datetime.now(timezone.utc)
        self.last_update = self.created_at
        
        # Logging setup - każda strategia ma własny logger
        self.logger = logging.getLogger(f"Strategy.{self.strategy_name}")
        
        # Performance tracking
        self.metrics = StrategyMetrics()
        self.trade_history = deque(maxlen=1000)  # Keep last 1000 trades
        self.signal_history = deque(maxlen=500)  # Keep last 500 signals
        self.performance_history = deque(maxlen=100)  # Keep last 100 performance snapshots
        
        # Risk management defaults
        self.max_positions = kwargs.get('max_positions', 3)
        self.max_loss_pct = kwargs.get('max_loss_pct', 10.0)
        self.min_profit_target_usdt = kwargs.get('min_profit_target_usdt', 5.0)
        
        # Position sizing defaults
        self.base_position_size_pct = kwargs.get('base_position_size_pct', 25.0)
        self.min_position_usdt = kwargs.get('min_position_usdt', 150.0)
        self.max_position_usdt = kwargs.get('max_position_usdt', 350.0)
        
        # Technical analysis storage
        self.indicators = {}
        self.market_data = None
        self.current_price = 0.0
        
        # ML integration interface
        self.ml_enabled = kwargs.get('ml_enabled', False)
        self.ml_predictor = None
        self.ml_confidence_threshold = kwargs.get('ml_confidence_threshold', 0.6)
        
        # Sentiment integration interface
        self.sentiment_enabled = kwargs.get('sentiment_enabled', False)
        self.sentiment_provider = None
        
        # Parameter evolution interface
        self.evolution_enabled = kwargs.get('evolution_enabled', False)
        self.parameter_history = deque(maxlen=50)
        
        # Strategy-specific parameters storage
        self.parameters = kwargs
        
        self.logger.info(f"✅ {self.strategy_name} base initialization completed")
        self.state = StrategyState.ACTIVE
    
    # ABSTRACT METHODS - Must be implemented by child strategies
    
    @abstractmethod
    async def analyze_market(self, data: pd.DataFrame) -> TradingSignal:
        """
        Analyze market data and generate trading signal
        
        Args:
            data: Market data (OHLCV)
            
        Returns:
            TradingSignal: Generated trading signal
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """
        Calculate position size for given signal
        
        Args:
            signal: Trading signal
            
        Returns:
            float: Position size in USDT
        """
        pass
    
    # COMMON FUNCTIONALITY - Shared by all strategies
    
    async def process_data(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        Main data processing pipeline
        
        Args:
            data: Market data to process
            
        Returns:
            Optional[TradingSignal]: Generated signal if any
        """
        try:
            self.market_data = data
            if len(data) > 0:
                self.current_price = float(data['close'].iloc[-1])
            
            # Update timestamp
            self.last_update = datetime.now(timezone.utc)
            
            # Run strategy-specific analysis
            signal = await self.analyze_market(data)
            
            if signal:
                # Record signal
                self.signal_history.append(signal)
                
                # Execute trade if signal is strong enough
                await self.execute_signal(signal)
                
                return signal
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ Data processing error: {e}")
            self.state = StrategyState.ERROR
            return None
    
    async def execute_signal(self, signal: TradingSignal) -> bool:
        """
        Execute trading signal
        
        Args:
            signal: Signal to execute
            
        Returns:
            bool: True if execution successful
        """
        try:
            if signal.signal_type == SignalType.BUY:
                return await self._execute_buy_signal(signal)
            elif signal.signal_type == SignalType.SELL:
                return await self._execute_sell_signal(signal)
            elif signal.signal_type == SignalType.CLOSE:
                return await self._execute_close_signal(signal)
            
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Signal execution error: {e}")
            return False
    
    async def _execute_buy_signal(self, signal: TradingSignal) -> bool:
        """Execute buy signal"""
        try:
            # Check if we can open new position
            if len(self.portfolio.positions) >= self.max_positions:
                self.logger.warning(f"⚠️ Max positions reached ({self.max_positions})")
                return False
            
            # Calculate position size
            position_size = self.calculate_position_size(signal)
            
            if position_size < self.min_position_usdt:
                self.logger.warning(f"⚠️ Position size too small: ${position_size:.2f}")
                return False
            
            # Execute buy order
            success = await self.portfolio.buy(
                symbol=self.symbol,
                amount_usdt=position_size,
                price=signal.price
            )
            
            if success:
                self.logger.info(f"✅ BUY executed: ${position_size:.2f} at ${signal.price:.2f}")
                self._record_trade("BUY", position_size, signal.price, signal.confidence)
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Buy execution error: {e}")
            return False
    
    async def _execute_sell_signal(self, signal: TradingSignal) -> bool:
        """Execute sell signal"""
        try:
            # Find positions to close
            positions_to_close = [
                pos for pos in self.portfolio.positions.values()
                if pos.symbol == self.symbol
            ]
            
            if not positions_to_close:
                self.logger.warning("⚠️ No positions to close")
                return False
            
            closed_count = 0
            for position in positions_to_close:
                success = await self.portfolio.sell(
                    symbol=self.symbol,
                    position_id=position.id,
                    price=signal.price
                )
                
                if success:
                    closed_count += 1
                    profit = (signal.price - position.entry_price) * position.quantity
                    self.logger.info(f"✅ SELL executed: Position {position.id}, Profit: ${profit:.2f}")
                    self._record_trade("SELL", position.quantity * signal.price, signal.price, signal.confidence)
            
            return closed_count > 0
            
        except Exception as e:
            self.logger.error(f"❌ Sell execution error: {e}")
            return False
    
    async def _execute_close_signal(self, signal: TradingSignal) -> bool:
        """Execute close signal (emergency close)"""
        return await self._execute_sell_signal(signal)
    
    def _record_trade(self, trade_type: str, amount: float, price: float, confidence: float):
        """Record trade for analytics"""
        trade_record = {
            "timestamp": datetime.now(timezone.utc),
            "type": trade_type,
            "amount": amount,
            "price": price,
            "confidence": confidence,
            "portfolio_value": self.portfolio.get_total_value()
        }
        self.trade_history.append(trade_record)
    
    def update_metrics(self) -> None:
        """Update strategy performance metrics"""
        try:
            # Calculate basic metrics from trade history
            if len(self.trade_history) < 2:
                return
            
            buy_trades = [t for t in self.trade_history if t["type"] == "BUY"]
            sell_trades = [t for t in self.trade_history if t["type"] == "SELL"]
            
            self.metrics.total_trades = len(buy_trades)
            
            if len(sell_trades) > 0:
                # Calculate profits for completed trades
                profits = []
                for sell_trade in sell_trades:
                    # Find corresponding buy trade (simplified)
                    buy_trades_before = [
                        t for t in buy_trades 
                        if t["timestamp"] < sell_trade["timestamp"]
                    ]
                    if buy_trades_before:
                        last_buy = buy_trades_before[-1]
                        profit = sell_trade["amount"] - last_buy["amount"]
                        profits.append(profit)
                
                if profits:
                    self.metrics.winning_trades = len([p for p in profits if p > 0])
                    self.metrics.losing_trades = len([p for p in profits if p <= 0])
                    self.metrics.total_profit_usdt = sum(profits)
                    self.metrics.avg_profit_per_trade = sum(profits) / len(profits)
                    
                    if self.metrics.total_trades > 0:
                        self.metrics.win_rate_pct = (self.metrics.winning_trades / self.metrics.total_trades) * 100
            
            self.metrics.last_updated = datetime.now(timezone.utc)
            
        except Exception as e:
            self.logger.error(f"❌ Metrics update error: {e}")
    
    def get_strategy_analytics(self) -> Dict[str, Any]:
        """Get comprehensive strategy analytics"""
        try:
            self.update_metrics()
            
            analytics = {
                "strategy_info": {
                    "name": self.strategy_name,
                    "symbol": self.symbol,
                    "state": self.state.value,
                    "created_at": self.created_at.isoformat(),
                    "last_update": self.last_update.isoformat(),
                    "uptime_hours": (datetime.now(timezone.utc) - self.created_at).total_seconds() / 3600
                },
                "performance_metrics": {
                    "total_trades": self.metrics.total_trades,
                    "winning_trades": self.metrics.winning_trades,
                    "losing_trades": self.metrics.losing_trades,
                    "win_rate_pct": self.metrics.win_rate_pct,
                    "total_profit_usdt": self.metrics.total_profit_usdt,
                    "avg_profit_per_trade": self.metrics.avg_profit_per_trade,
                    "max_drawdown_pct": self.metrics.max_drawdown_pct,
                    "sharpe_ratio": self.metrics.sharpe_ratio
                },
                "current_status": {
                    "active_positions": len(self.portfolio.positions),
                    "portfolio_value": self.portfolio.get_total_value(),
                    "current_price": self.current_price,
                    "last_signal_time": self.signal_history[-1].timestamp.isoformat() if self.signal_history else None,
                    "signals_generated": len(self.signal_history)
                },
                "configuration": {
                    "max_positions": self.max_positions,
                    "max_loss_pct": self.max_loss_pct,
                    "base_position_size_pct": self.base_position_size_pct,
                    "ml_enabled": self.ml_enabled,
                    "sentiment_enabled": self.sentiment_enabled,
                    "evolution_enabled": self.evolution_enabled
                }
            }
            
            return analytics
            
        except Exception as e:
            self.logger.error(f"❌ Analytics generation error: {e}")
            return {"error": str(e)}
    
    def pause_strategy(self) -> None:
        """Pause strategy execution"""
        self.state = StrategyState.PAUSED
        self.logger.info(f"⏸️ Strategy paused")
    
    def resume_strategy(self) -> None:
        """Resume strategy execution"""
        self.state = StrategyState.ACTIVE
        self.logger.info(f"▶️ Strategy resumed")
    
    def stop_strategy(self) -> None:
        """Stop strategy execution"""
        self.state = StrategyState.STOPPED
        self.logger.info(f"⏹️ Strategy stopped")
    
    def is_active(self) -> bool:
        """Check if strategy is active"""
        return self.state == StrategyState.ACTIVE
    
    def __str__(self) -> str:
        """String representation"""
        return f"{self.strategy_name}({self.symbol}) - {self.state.value}"
    
    def __repr__(self) -> str:
        """Debug representation"""
        return f"<{self.__class__.__name__}: {self.strategy_name}, {self.state.value}, {len(self.portfolio.positions)} positions>"


# UTILITY FUNCTIONS FOR STRATEGY DEVELOPMENT

def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
    """
    Calculate common technical indicators
    
    Args:
        data: OHLCV data
        
    Returns:
        Dict of indicator series
    """
    indicators = {}
    
    try:
        # Moving averages
        indicators['ema_12'] = data['close'].ewm(span=12).mean()
        indicators['ema_21'] = data['close'].ewm(span=21).mean()
        indicators['ema_50'] = data['close'].ewm(span=50).mean()
        
        # RSI
        delta = data['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / loss
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # Bollinger Bands
        indicators['bb_middle'] = data['close'].rolling(window=20).mean()
        bb_std = data['close'].rolling(window=20).std()
        indicators['bb_upper'] = indicators['bb_middle'] + (bb_std * 2)
        indicators['bb_lower'] = indicators['bb_middle'] - (bb_std * 2)
        
        # Volume indicators
        indicators['volume_sma'] = data['volume'].rolling(window=20).mean()
        indicators['volume_ratio'] = data['volume'] / indicators['volume_sma']
        
    except Exception as e:
        logger.error(f"❌ Technical indicators calculation error: {e}")
    
    return indicators


def create_signal(
    signal_type: SignalType, 
    confidence: float, 
    price: float, 
    reasons: List[str] = None
) -> TradingSignal:
    """
    Helper function to create trading signals
    
    Args:
        signal_type: Type of signal
        confidence: Signal confidence (0.0 to 1.0)
        price: Signal price
        reasons: List of reasons for the signal
        
    Returns:
        TradingSignal instance
    """
    return TradingSignal(
        signal_type=signal_type,
        confidence=max(0.0, min(1.0, confidence)),  # Clamp between 0 and 1
        price=price,
        timestamp=datetime.now(timezone.utc),
        reasons=reasons or [],
        metadata={}
    )