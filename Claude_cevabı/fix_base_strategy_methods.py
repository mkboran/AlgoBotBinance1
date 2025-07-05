#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - BASE STRATEGY METHODS FIX
💎 FIXED: BaseStrategy'ye eksik metodlar eklendi

ÇÖZÜMLER:
1. ✅ should_sell metodu eklendi
2. ✅ _get_position_age_minutes metodu eklendi
3. ✅ _calculate_performance_multiplier metodu eklendi
4. ✅ Diğer yardımcı metodlar eklendi
"""

from abc import ABC, abstractmethod
from dataclasses import dataclass, field
from datetime import datetime, timezone, timedelta
from enum import Enum
from typing import Dict, List, Optional, Any, Tuple, Union
import pandas as pd
import numpy as np
import logging
from collections import deque
import asyncio

# Configure logger
logger = logging.getLogger("algobot.strategies")


class SignalType(Enum):
    """Trading signal types"""
    BUY = "BUY"
    SELL = "SELL"
    HOLD = "HOLD"
    CLOSE = "CLOSE"  # Specific close signal


class VolatilityRegime(Enum):
    """Market volatility regimes"""
    LOW = "LOW"
    NORMAL = "NORMAL"
    HIGH = "HIGH"
    EXTREME = "EXTREME"


class GlobalMarketRegime(Enum):
    """Global market regime classification"""
    RISK_ON = "RISK_ON"
    RISK_OFF = "RISK_OFF"
    NEUTRAL = "NEUTRAL"
    UNCERTAIN = "UNCERTAIN"


@dataclass
class TradingSignal:
    """Enhanced trading signal with comprehensive metadata"""
    signal_type: SignalType
    confidence: float  # 0.0 to 1.0
    price: float
    timestamp: datetime
    reasons: List[str]
    
    # Optional metadata
    metadata: Dict[str, Any] = field(default_factory=dict)
    quality_score: Optional[int] = None
    ml_prediction: Optional[float] = None
    ml_confidence: Optional[float] = None
    volatility_regime: Optional[VolatilityRegime] = None
    global_regime: Optional[GlobalMarketRegime] = None
    position_size_override: Optional[float] = None
    stop_loss: Optional[float] = None
    take_profit: Optional[float] = None
    
    def __post_init__(self):
        # Add metadata to dict if provided separately
        if self.quality_score:
            self.metadata['quality_score'] = self.quality_score
        if self.ml_prediction:
            self.metadata['ml_prediction'] = self.ml_prediction
        if self.ml_confidence:
            self.metadata['ml_confidence'] = self.ml_confidence


class BaseStrategy(ABC):
    """
    🧠 BASE STRATEGY v2.0 - HEDGE FUND LEVEL FRAMEWORK
    💎 Complete Trading Strategy Foundation
    
    All strategies inherit from this base class
    """
    
    class StrategyState(Enum):
        """Strategy operational states"""
        INITIALIZING = "initializing"
        ACTIVE = "active"
        PAUSED = "paused"
        ERROR = "error"
        STOPPED = "stopped"
    
    @dataclass
    class GlobalMarketAnalysis:
        """Global market risk analysis results"""
        market_regime: GlobalMarketRegime
        regime_confidence: float
        risk_score: float  # 0-1, higher = more risk
        btc_spy_correlation: float
        btc_dxy_correlation: float
        btc_vix_correlation: float
        position_size_adjustment: float  # Multiplier for position sizing
        warnings: List[str] = field(default_factory=list)
    
    def __init__(self, 
                 portfolio,  # Portfolio instance
                 symbol: str = "BTC/USDT",
                 strategy_name: str = "BaseStrategy",
                 **kwargs):
        """Initialize base strategy"""
        
        # Core attributes
        self.portfolio = portfolio
        self.symbol = symbol
        self.strategy_name = strategy_name
        self.state = self.StrategyState.INITIALIZING
        
        # Logging
        self.logger = logging.getLogger(f"algobot.strategies.{strategy_name}")
        
        # Performance tracking
        self.trades_executed = 0
        self.winning_trades = 0
        self.losing_trades = 0
        self.total_profit_usdt = 0.0
        self.total_fees_paid = 0.0
        
        # Signal history
        self.signal_history = deque(maxlen=100)
        self.trade_history = []
        
        # Advanced features flags
        self.ml_enabled = kwargs.get('ml_enabled', True)
        self.dynamic_exit_enabled = kwargs.get('dynamic_exit_enabled', True)
        self.kelly_enabled = kwargs.get('kelly_enabled', True)
        self.global_intelligence_enabled = kwargs.get('global_intelligence_enabled', True)
        
        # Risk parameters
        self.max_position_size_pct = kwargs.get('max_position_size_pct', 0.5)
        self.max_portfolio_risk_pct = kwargs.get('max_portfolio_risk_pct', 0.06)
        self.position_risk_pct = kwargs.get('position_risk_pct', 0.02)
        
        # ML components (to be initialized by child classes)
        self.ml_predictor = None
        self.ml_confidence_threshold = 0.65
        
        # Initialize strategy
        self._initialize_strategy()
        
        self.logger.info(f"🚀 {strategy_name} initialized for {symbol}")
        self.state = self.StrategyState.ACTIVE
    
    def _initialize_strategy(self):
        """Initialize strategy-specific components"""
        # Override in child classes for specific initialization
        pass
    
    @abstractmethod
    async def analyze_market(self, data: pd.DataFrame) -> TradingSignal:
        """
        Analyze market data and generate trading signal
        Must be implemented by each strategy
        """
        pass
    
    @abstractmethod
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """
        Calculate position size based on signal and risk management
        Must be implemented by each strategy
        """
        pass
    
    async def should_sell(self, 
                         position,  # Position instance
                         current_data: pd.DataFrame) -> Tuple[bool, str]:
        """
        ✅ FIXED: Dynamic exit decision logic
        
        Returns:
            Tuple[bool, str]: (should_sell, reason)
        """
        
        current_price = current_data['close'].iloc[-1]
        
        # Update position metrics
        position.update_performance_metrics(current_price)
        
        # Get position age
        position_age_minutes = self._get_position_age_minutes(position)
        
        # 1. Stop Loss Check
        if position.stop_loss_price and current_price <= position.stop_loss_price:
            return True, f"Stop loss hit at ${current_price:.2f}"
        
        # 2. Take Profit Check
        if position.take_profit_price and current_price >= position.take_profit_price:
            return True, f"Take profit hit at ${current_price:.2f}"
        
        # 3. Trailing Stop Check
        if position.trailing_stop_distance:
            should_trail, new_stop = position.should_trail_stop(current_price)
            if should_trail and new_stop:
                position.stop_loss_price = new_stop
        
        # 4. Time-based Exit (position too old)
        max_hold_minutes = getattr(self, 'max_hold_minutes', 1440)  # 24 hours default
        if position_age_minutes > max_hold_minutes:
            return True, f"Position age exceeded {max_hold_minutes} minutes"
        
        # 5. Profit Target Check
        profit_pct = position.unrealized_pnl_pct * 100
        
        # Dynamic profit targets based on position quality
        quality_score = position.quality_score
        if quality_score >= 18:
            profit_target = 3.0  # 3% for high quality
        elif quality_score >= 15:
            profit_target = 2.0  # 2% for good quality
        else:
            profit_target = 1.5  # 1.5% for normal quality
        
        if profit_pct >= profit_target:
            return True, f"Profit target {profit_target}% reached"
        
        # 6. Loss Limit Check
        max_loss_pct = getattr(self, 'max_loss_pct', 2.0)
        if profit_pct <= -max_loss_pct:
            return True, f"Max loss {max_loss_pct}% reached"
        
        # 7. Strategy-specific Exit Signals
        if hasattr(self, '_check_exit_signals'):
            should_exit, exit_reason = self._check_exit_signals(position, current_data)
            if should_exit:
                return True, exit_reason
        
        # 8. ML-based Exit (if enabled)
        if self.ml_enabled and self.ml_predictor:
            ml_exit_signal = await self._get_ml_exit_signal(position, current_data)
            if ml_exit_signal:
                return True, "ML model exit signal"
        
        return False, "Hold position"
    
    def _get_position_age_minutes(self, position) -> int:
        """
        ✅ FIXED: Calculate position age in minutes
        """
        try:
            # Parse position timestamp
            if isinstance(position.timestamp, str):
                position_time = datetime.fromisoformat(position.timestamp.replace('Z', '+00:00'))
            else:
                position_time = position.timestamp
            
            # Ensure timezone aware
            if position_time.tzinfo is None:
                position_time = position_time.replace(tzinfo=timezone.utc)
            
            # Calculate age
            current_time = datetime.now(timezone.utc)
            age_delta = current_time - position_time
            
            return int(age_delta.total_seconds() / 60)
            
        except Exception as e:
            self.logger.error(f"Error calculating position age: {e}")
            return 0
    
    def _calculate_performance_multiplier(self) -> float:
        """
        ✅ FIXED: Calculate performance-based position size multiplier
        """
        if self.trades_executed < 10:
            # Not enough trades for reliable calculation
            return 1.0
        
        # Calculate win rate
        win_rate = self.winning_trades / self.trades_executed if self.trades_executed > 0 else 0.5
        
        # Calculate profit factor
        avg_win = self.total_profit_usdt / self.winning_trades if self.winning_trades > 0 else 0
        avg_loss = abs(self.total_profit_usdt - (self.winning_trades * avg_win)) / self.losing_trades if self.losing_trades > 0 else 0
        
        profit_factor = avg_win / avg_loss if avg_loss > 0 else 1.0
        
        # Performance multiplier based on win rate and profit factor
        if win_rate >= 0.6 and profit_factor >= 1.5:
            return 1.2  # Increase size by 20%
        elif win_rate >= 0.5 and profit_factor >= 1.2:
            return 1.1  # Increase size by 10%
        elif win_rate < 0.4 or profit_factor < 0.8:
            return 0.8  # Decrease size by 20%
        else:
            return 1.0  # Normal size
    
    def _analyze_global_market_risk(self, global_data: Dict[str, pd.DataFrame]) -> GlobalMarketAnalysis:
        """
        Analyze global market conditions for risk assessment
        
        Args:
            global_data: Dict with keys like 'BTC', 'SPY', 'DXY', 'VIX'
        """
        try:
            # Default analysis
            analysis = self.GlobalMarketAnalysis(
                market_regime=GlobalMarketRegime.NEUTRAL,
                regime_confidence=0.5,
                risk_score=0.5,
                btc_spy_correlation=0.0,
                btc_dxy_correlation=0.0,
                btc_vix_correlation=0.0,
                position_size_adjustment=1.0
            )
            
            # Check if we have required data
            if 'BTC' not in global_data or len(global_data['BTC']) < 20:
                return analysis
            
            btc_returns = global_data['BTC']['close'].pct_change().dropna()
            
            # SPY correlation (risk-on/risk-off indicator)
            if 'SPY' in global_data and len(global_data['SPY']) >= 20:
                spy_returns = global_data['SPY']['close'].pct_change().dropna()
                analysis.btc_spy_correlation = btc_returns.tail(20).corr(spy_returns.tail(20))
            
            # DXY correlation (dollar strength)
            if 'DXY' in global_data and len(global_data['DXY']) >= 20:
                dxy_returns = global_data['DXY']['close'].pct_change().dropna()
                analysis.btc_dxy_correlation = btc_returns.tail(20).corr(dxy_returns.tail(20))
            
            # VIX level (market fear)
            if 'VIX' in global_data and len(global_data['VIX']) > 0:
                vix_level = global_data['VIX']['close'].iloc[-1]
                if vix_level > 30:
                    analysis.warnings.append("High VIX - Market fear elevated")
                    analysis.risk_score = min(analysis.risk_score + 0.3, 1.0)
            
            # Determine market regime
            if analysis.btc_spy_correlation > 0.5:
                analysis.market_regime = GlobalMarketRegime.RISK_ON
                analysis.regime_confidence = min(analysis.btc_spy_correlation, 0.9)
            elif analysis.btc_spy_correlation < -0.3:
                analysis.market_regime = GlobalMarketRegime.RISK_OFF
                analysis.regime_confidence = min(abs(analysis.btc_spy_correlation), 0.9)
                analysis.risk_score = min(analysis.risk_score + 0.2, 1.0)
            
            # Position size adjustment based on risk
            if analysis.risk_score > 0.7:
                analysis.position_size_adjustment = 0.5  # Halve position size
            elif analysis.risk_score > 0.5:
                analysis.position_size_adjustment = 0.75  # Reduce by 25%
            
            return analysis
            
        except Exception as e:
            self.logger.error(f"Global market analysis error: {e}")
            return self.GlobalMarketAnalysis(
                market_regime=GlobalMarketRegime.UNCERTAIN,
                regime_confidence=0.0,
                risk_score=0.5,
                btc_spy_correlation=0.0,
                btc_dxy_correlation=0.0,
                btc_vix_correlation=0.0,
                position_size_adjustment=0.8  # Be conservative on error
            )
    
    @staticmethod
    def calculate_technical_indicators(data: pd.DataFrame) -> Dict[str, pd.Series]:
        """Calculate common technical indicators"""
        
        indicators = {}
        
        # Price data
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # Moving averages
        indicators['ema_short'] = close.ewm(span=12, adjust=False).mean()
        indicators['ema_medium'] = close.ewm(span=26, adjust=False).mean()
        indicators['ema_long'] = close.ewm(span=50, adjust=False).mean()
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=14).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=14).mean()
        rs = gain / (loss + 1e-10)
        indicators['rsi'] = 100 - (100 / (1 + rs))
        
        # ATR
        tr1 = high - low
        tr2 = abs(high - close.shift(1))
        tr3 = abs(low - close.shift(1))
        tr = pd.concat([tr1, tr2, tr3], axis=1).max(axis=1)
        indicators['atr'] = tr.rolling(window=14).mean()
        
        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        indicators['bb_upper'] = sma_20 + (2 * std_20)
        indicators['bb_lower'] = sma_20 - (2 * std_20)
        indicators['bb_middle'] = sma_20
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        indicators['macd'] = ema_12 - ema_26
        indicators['macd_signal'] = indicators['macd'].ewm(span=9, adjust=False).mean()
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Volume indicators
        indicators['volume_sma'] = volume.rolling(window=20).mean()
        indicators['volume_ratio'] = volume / indicators['volume_sma']
        
        return indicators
    
    @staticmethod
    def create_signal(signal_type: SignalType,
                     confidence: float,
                     price: float,
                     reasons: List[str],
                     metadata: Optional[Dict[str, Any]] = None) -> TradingSignal:
        """Create a trading signal with current timestamp"""
        
        return TradingSignal(
            signal_type=signal_type,
            confidence=confidence,
            price=price,
            timestamp=datetime.now(timezone.utc),
            reasons=reasons,
            metadata=metadata or {}
        )
    
    def get_strategy_analytics(self) -> Dict[str, Any]:
        """Get comprehensive strategy performance analytics"""
        
        total_trades = self.trades_executed
        
        # Basic metrics
        win_rate = self.winning_trades / total_trades if total_trades > 0 else 0
        avg_profit_per_trade = self.total_profit_usdt / total_trades if total_trades > 0 else 0
        
        # Calculate profit factor
        if hasattr(self, 'gross_profit') and hasattr(self, 'gross_loss'):
            profit_factor = self.gross_profit / abs(self.gross_loss) if self.gross_loss != 0 else 0
        else:
            profit_factor = 0
        
        # Calculate Sharpe ratio (simplified)
        if len(self.trade_history) > 1:
            returns = [t.get('profit_pct', 0) for t in self.trade_history]
            sharpe_ratio = np.mean(returns) / (np.std(returns) + 1e-10) * np.sqrt(252)
        else:
            sharpe_ratio = 0
        
        # Max drawdown calculation
        if hasattr(self, 'equity_curve') and len(self.equity_curve) > 1:
            equity_array = np.array(self.equity_curve)
            peak = np.maximum.accumulate(equity_array)
            drawdown = (peak - equity_array) / peak
            max_drawdown = np.max(drawdown)
        else:
            max_drawdown = 0
        
        return {
            'total_trades': total_trades,
            'winning_trades': self.winning_trades,
            'losing_trades': self.losing_trades,
            'total_profit_usdt': self.total_profit_usdt,
            'total_return_pct': (self.total_profit_usdt / self.portfolio.initial_capital_usdt * 100) if self.portfolio else 0,
            'win_rate_pct': win_rate * 100,
            'avg_profit_per_trade': avg_profit_per_trade,
            'profit_factor': profit_factor,
            'sharpe_ratio': sharpe_ratio,
            'max_drawdown_pct': max_drawdown * 100,
            'total_fees_paid': self.total_fees_paid,
            'strategy_state': self.state.value
        }
    
    def is_active(self) -> bool:
        """Check if strategy is active"""
        return self.state == self.StrategyState.ACTIVE
    
    def pause_strategy(self):
        """Pause strategy execution"""
        self.state = self.StrategyState.PAUSED
        self.logger.info(f"⏸️ {self.strategy_name} paused")
    
    def resume_strategy(self):
        """Resume strategy execution"""
        self.state = self.StrategyState.ACTIVE
        self.logger.info(f"▶️ {self.strategy_name} resumed")
    
    def stop_strategy(self):
        """Stop strategy execution"""
        self.state = self.StrategyState.STOPPED
        self.logger.info(f"⏹️ {self.strategy_name} stopped")
    
    async def _get_ml_exit_signal(self, position, current_data: pd.DataFrame) -> bool:
        """Get ML-based exit signal (override in child classes)"""
        return False
    
    def _check_exit_signals(self, position, current_data: pd.DataFrame) -> Tuple[bool, str]:
        """Check strategy-specific exit signals (override in child classes)"""
        return False, ""
    
    def __repr__(self):
        return f"{self.strategy_name}(symbol={self.symbol}, state={self.state.value})"