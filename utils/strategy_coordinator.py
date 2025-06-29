# utils/strategy_coordinator.py
#!/usr/bin/env python3
"""
🎯 MULTI-STRATEGY COORDINATION SYSTEM
🔥 BREAKTHROUGH: Central Intelligence for Strategy Orchestra

Revolutionary strategy coordination system that provides:
- Centralized strategy lifecycle management
- Real-time strategy performance monitoring
- Dynamic strategy allocation and rebalancing
- Cross-strategy signal correlation analysis
- Risk management across all strategies
- Performance attribution and analytics
- Automated strategy selection and weighting
- Market regime-based strategy activation
- Conflict resolution between strategies
- Enhanced portfolio optimization

The brain that orchestrates all strategies for maximum performance
INSTITUTIONAL LEVEL IMPLEMENTATION - PRODUCTION READY
"""

import pandas as pd
import numpy as np
from typing import Dict, List, Tuple, Optional, Any, Union
from datetime import datetime, timezone, timedelta
from dataclasses import dataclass, field
from enum import Enum
import logging
import asyncio
from collections import deque, defaultdict
import json
import warnings
warnings.filterwarnings('ignore')

from utils.portfolio import Portfolio, Position
from utils.config import settings
from utils.logger import logger

# Strategy imports (will be imported dynamically)
# from strategies.momentum_optimized import EnhancedMomentumStrategy
# from strategies.bollinger_ml_strategy import BollingerMLStrategy
# from strategies.rsi_ml_strategy import RSIMLStrategy
# from strategies.macd_ml_strategy import MACDMLStrategy
# from strategies.volume_profile_strategy import VolumeProfileMLStrategy

class StrategyStatus(Enum):
    """Strategy status enumeration"""
    INACTIVE = "inactive"
    ACTIVE = "active"
    PAUSED = "paused"
    ERROR = "error"
    OPTIMIZATION = "optimization"

class MarketRegime(Enum):
    """Market regime classifications"""
    TRENDING_UP = ("trending_up", "Strong uptrend with momentum")
    TRENDING_DOWN = ("trending_down", "Strong downtrend with momentum")
    SIDEWAYS = ("sideways", "Range-bound market")
    VOLATILE = ("volatile", "High volatility, uncertain direction")
    LOW_VOLATILITY = ("low_volatility", "Low volatility, consolidation")
    BREAKOUT = ("breakout", "Breaking out of range")
    REVERSAL = ("reversal", "Potential trend reversal")
    
    def __init__(self, regime_name: str, description: str):
        self.regime_name = regime_name
        self.description = description

@dataclass
class StrategyAllocation:
    """Strategy allocation configuration"""
    strategy_name: str
    target_weight: float = 0.0
    current_weight: float = 0.0
    min_weight: float = 0.05
    max_weight: float = 0.4
    status: StrategyStatus = StrategyStatus.INACTIVE
    performance_score: float = 0.0
    risk_score: float = 0.0
    last_rebalance: Optional[datetime] = None
    allocation_reason: str = ""

@dataclass
class StrategyPerformance:
    """Strategy performance metrics"""
    strategy_name: str
    total_trades: int = 0
    winning_trades: int = 0
    total_profit_pct: float = 0.0
    avg_profit_pct: float = 0.0
    max_drawdown_pct: float = 0.0
    sharpe_ratio: float = 0.0
    win_rate: float = 0.0
    avg_hold_time_minutes: float = 0.0
    recent_performance_trend: float = 0.0
    last_updated: datetime = field(default_factory=lambda: datetime.now(timezone.utc))

@dataclass
class CrossStrategySignal:
    """Cross-strategy signal analysis"""
    timestamp: datetime
    strategy_signals: Dict[str, bool]
    signal_correlation: float
    confidence_score: float
    recommended_action: str
    conflicting_strategies: List[str] = field(default_factory=list)

class StrategyCoordinator:
    """🎯 Central Strategy Coordination System"""
    
    def __init__(
        self,
        portfolio: Portfolio,
        symbol: str = "BTC/USDT",
        rebalancing_frequency_hours: int = 6,
        min_rebalancing_threshold: float = 0.1,
        max_total_exposure: float = 0.25,
        enable_cross_validation: bool = True,
        enable_regime_switching: bool = True,
        performance_lookback_trades: int = 50
    ):
        self.portfolio = portfolio
        self.symbol = symbol
        self.rebalancing_frequency_hours = rebalancing_frequency_hours
        self.min_rebalancing_threshold = min_rebalancing_threshold
        self.max_total_exposure = max_total_exposure
        self.enable_cross_validation = enable_cross_validation
        self.enable_regime_switching = enable_regime_switching
        self.performance_lookback_trades = performance_lookback_trades
        
        # Strategy management
        self.strategies: Dict[str, Any] = {}
        self.strategy_allocations: Dict[str, StrategyAllocation] = {}
        self.strategy_performances: Dict[str, StrategyPerformance] = {}
        
        # Market analysis
        self.current_market_regime = MarketRegime.SIDEWAYS
        self.market_regime_history = deque(maxlen=200)
        self.market_regime_confidence = 0.5
        
        # Cross-strategy analysis
        self.cross_strategy_signals = deque(maxlen=100)
        self.signal_correlations = defaultdict(float)
        self.strategy_conflicts = defaultdict(int)
        
        # Performance tracking
        self.last_rebalance_time = None
        self.total_coordination_cycles = 0
        self.successful_coordinations = 0
        self.coordination_errors = 0
        
        # Risk management
        self.risk_limits = {
            'max_concurrent_positions': 8,
            'max_single_strategy_exposure': 0.15,
            'max_correlation_threshold': 0.8,
            'min_diversification_score': 0.6
        }
        
        # Analytics
        self.coordination_analytics = deque(maxlen=500)
        self.regime_prediction_history = deque(maxlen=300)
        
        logger.info(f"🎯 Strategy Coordinator initialized")
        logger.info(f"   🔄 Rebalancing: every {rebalancing_frequency_hours}h (threshold: {min_rebalancing_threshold})")
        logger.info(f"   💰 Max exposure: {max_total_exposure*100:.0f}% (single strategy: {self.risk_limits['max_single_strategy_exposure']*100:.0f}%)")
        logger.info(f"   🧠 Features: Cross-validation={enable_cross_validation}, Regime-switching={enable_regime_switching}")

    def register_strategy(
        self, 
        strategy_name: str, 
        strategy_instance: Any,
        initial_weight: float = 0.2,
        min_weight: float = 0.05,
        max_weight: float = 0.4
    ) -> bool:
        """📝 Register a strategy with the coordinator"""
        try:
            if strategy_name in self.strategies:
                logger.warning(f"Strategy {strategy_name} already registered, updating...")
            
            self.strategies[strategy_name] = strategy_instance
            
            # Initialize allocation
            self.strategy_allocations[strategy_name] = StrategyAllocation(
                strategy_name=strategy_name,
                target_weight=initial_weight,
                current_weight=0.0,
                min_weight=min_weight,
                max_weight=max_weight,
                status=StrategyStatus.ACTIVE,
                allocation_reason="INITIAL_REGISTRATION"
            )
            
            # Initialize performance tracking
            self.strategy_performances[strategy_name] = StrategyPerformance(
                strategy_name=strategy_name
            )
            
            logger.info(f"✅ Strategy registered: {strategy_name} (weight: {initial_weight:.1%})")
            return True
            
        except Exception as e:
            logger.error(f"Strategy registration error for {strategy_name}: {e}")
            return False

    def unregister_strategy(self, strategy_name: str) -> bool:
        """🗑️ Unregister a strategy"""
        try:
            if strategy_name not in self.strategies:
                logger.warning(f"Strategy {strategy_name} not found for unregistration")
                return False
            
            # Set status to inactive
            if strategy_name in self.strategy_allocations:
                self.strategy_allocations[strategy_name].status = StrategyStatus.INACTIVE
                self.strategy_allocations[strategy_name].target_weight = 0.0
            
            logger.info(f"🗑️ Strategy unregistered: {strategy_name}")
            return True
            
        except Exception as e:
            logger.error(f"Strategy unregistration error for {strategy_name}: {e}")
            return False

    async def analyze_market_regime(self, df: pd.DataFrame) -> MarketRegime:
        """🧠 Analyze current market regime"""
        try:
            if df is None or df.empty or len(df) < 50:
                return MarketRegime.SIDEWAYS
            
            # Calculate market indicators
            recent_data = df.tail(50)
            
            # Trend analysis
            ema_short = recent_data['close'].ewm(span=12).mean()
            ema_long = recent_data['close'].ewm(span=26).mean()
            trend_strength = (ema_short.iloc[-1] - ema_long.iloc[-1]) / ema_long.iloc[-1]
            
            # Volatility analysis
            returns = recent_data['close'].pct_change()
            volatility = returns.std() * np.sqrt(24 * 60 / 2.5)  # Annualized volatility
            
            # Volume analysis
            avg_volume = recent_data['volume'].mean()
            recent_volume = recent_data['volume'].tail(5).mean()
            volume_ratio = recent_volume / avg_volume if avg_volume > 0 else 1
            
            # Price momentum
            price_momentum = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[-20]) / recent_data['close'].iloc[-20]
            
            # Range analysis
            high_low_ratio = (recent_data['high'].max() - recent_data['low'].min()) / recent_data['close'].iloc[-1]
            
            # Regime classification logic
            regime_confidence = 0.5
            
            if abs(trend_strength) > 0.02 and volatility < 0.4:
                if trend_strength > 0:
                    regime = MarketRegime.TRENDING_UP
                    regime_confidence = min(0.9, 0.5 + abs(trend_strength) * 10)
                else:
                    regime = MarketRegime.TRENDING_DOWN  
                    regime_confidence = min(0.9, 0.5 + abs(trend_strength) * 10)
                    
            elif volatility > 0.6:
                regime = MarketRegime.VOLATILE
                regime_confidence = min(0.9, 0.3 + (volatility - 0.6) * 2)
                
            elif volatility < 0.2:
                regime = MarketRegime.LOW_VOLATILITY
                regime_confidence = min(0.9, 0.4 + (0.2 - volatility) * 5)
                
            elif volume_ratio > 2.0 and abs(price_momentum) > 0.03:
                regime = MarketRegime.BREAKOUT
                regime_confidence = min(0.9, 0.6 + (volume_ratio - 2.0) * 0.2)
                
            elif abs(trend_strength) < 0.01 and high_low_ratio < 0.05:
                regime = MarketRegime.SIDEWAYS
                regime_confidence = 0.7
                
            else:
                # Check for potential reversal
                recent_trend = (recent_data['close'].iloc[-5:].iloc[-1] - recent_data['close'].iloc[-5:].iloc[0]) / recent_data['close'].iloc[-5:].iloc[0]
                overall_trend = (recent_data['close'].iloc[-1] - recent_data['close'].iloc[0]) / recent_data['close'].iloc[0]
                
                if recent_trend * overall_trend < 0 and abs(recent_trend) > 0.02:
                    regime = MarketRegime.REVERSAL
                    regime_confidence = 0.6
                else:
                    regime = MarketRegime.SIDEWAYS
                    regime_confidence = 0.5
            
            # Update regime history
            self.current_market_regime = regime
            self.market_regime_confidence = regime_confidence
            self.market_regime_history.append({
                'timestamp': datetime.now(timezone.utc),
                'regime': regime,
                'confidence': regime_confidence,
                'trend_strength': trend_strength,
                'volatility': volatility,
                'volume_ratio': volume_ratio
            })
            
            return regime
            
        except Exception as e:
            logger.error(f"Market regime analysis error: {e}")
            return MarketRegime.SIDEWAYS

    async def update_strategy_performances(self) -> Dict[str, StrategyPerformance]:
        """📊 Update performance metrics for all strategies"""
        try:
            for strategy_name in self.strategies.keys():
                performance = self.strategy_performances[strategy_name]
                
                # Get strategy-specific trades
                strategy_trades = [
                    trade for trade in self.portfolio.closed_trades 
                    if trade.get('strategy') == strategy_name
                ]
                
                if strategy_trades:
                    # Calculate performance metrics
                    recent_trades = strategy_trades[-self.performance_lookback_trades:]
                    
                    performance.total_trades = len(strategy_trades)
                    performance.winning_trades = sum(1 for trade in strategy_trades if trade.get('profit_pct', 0) > 0)
                    performance.total_profit_pct = sum(trade.get('profit_pct', 0) for trade in strategy_trades)
                    performance.avg_profit_pct = performance.total_profit_pct / len(strategy_trades)
                    performance.win_rate = (performance.winning_trades / len(strategy_trades)) * 100
                    
                    # Recent performance trend
                    if len(recent_trades) >= 10:
                        first_half = recent_trades[:len(recent_trades)//2]
                        second_half = recent_trades[len(recent_trades)//2:]
                        
                        first_half_avg = np.mean([trade.get('profit_pct', 0) for trade in first_half])
                        second_half_avg = np.mean([trade.get('profit_pct', 0) for trade in second_half])
                        
                        performance.recent_performance_trend = second_half_avg - first_half_avg
                    
                    # Max drawdown calculation
                    cumulative_returns = np.cumsum([trade.get('profit_pct', 0) for trade in strategy_trades])
                    running_max = np.maximum.accumulate(cumulative_returns)
                    drawdown = running_max - cumulative_returns
                    performance.max_drawdown_pct = np.max(drawdown) if len(drawdown) > 0 else 0
                    
                    # Sharpe ratio (simplified)
                    returns = [trade.get('profit_pct', 0) for trade in recent_trades]
                    if len(returns) > 5:
                        performance.sharpe_ratio = np.mean(returns) / np.std(returns) if np.std(returns) > 0 else 0
                    
                    # Average hold time
                    hold_times = [trade.get('hold_time_minutes', 0) for trade in recent_trades]
                    performance.avg_hold_time_minutes = np.mean(hold_times) if hold_times else 0
                    
                    performance.last_updated = datetime.now(timezone.utc)
                
                # Calculate performance score (0-100)
                performance_score = self._calculate_performance_score(performance)
                self.strategy_allocations[strategy_name].performance_score = performance_score
            
            return self.strategy_performances
            
        except Exception as e:
            logger.error(f"Strategy performance update error: {e}")
            return {}

    def _calculate_performance_score(self, performance: StrategyPerformance) -> float:
        """🏆 Calculate overall performance score for a strategy"""
        try:
            if performance.total_trades < 5:
                return 50.0  # Neutral score for insufficient data
            
            # Weighted scoring components
            scores = {
                'win_rate': min(100, performance.win_rate * 1.2),  # Max 100
                'avg_profit': min(50, max(-50, performance.avg_profit_pct * 10)) + 50,  # 0-100 scale
                'sharpe_ratio': min(50, max(0, performance.sharpe_ratio * 10)),  # 0-50 scale
                'recent_trend': min(25, max(-25, performance.recent_performance_trend * 5)) + 25,  # 0-50 scale
                'consistency': max(0, 100 - performance.max_drawdown_pct * 2)  # Penalty for high drawdown
            }
            
            # Weights for each component
            weights = {
                'win_rate': 0.25,
                'avg_profit': 0.30,
                'sharpe_ratio': 0.20,
                'recent_trend': 0.15,
                'consistency': 0.10
            }
            
            # Calculate weighted score
            weighted_score = sum(scores[key] * weights[key] for key in scores.keys())
            return max(0, min(100, weighted_score))
            
        except Exception as e:
            logger.error(f"Performance score calculation error: {e}")
            return 50.0

    async def analyze_cross_strategy_signals(self, df: pd.DataFrame) -> CrossStrategySignal:
        """🔗 Analyze signals across all active strategies"""
        try:
            strategy_signals = {}
            signal_confidences = {}
            
            # Get signals from each active strategy
            for strategy_name, strategy_instance in self.strategies.items():
                if self.strategy_allocations[strategy_name].status == StrategyStatus.ACTIVE:
                    try:
                        # Call strategy's should_buy method
                        should_buy, reason, context = await strategy_instance.should_buy(df)
                        
                        strategy_signals[strategy_name] = should_buy
                        signal_confidences[strategy_name] = context.get('quality_score', 0) / 30.0  # Normalize to 0-1
                        
                    except Exception as e:
                        logger.debug(f"Signal analysis error for {strategy_name}: {e}")
                        strategy_signals[strategy_name] = False
                        signal_confidences[strategy_name] = 0.0
            
            # Calculate signal correlation
            signal_count = sum(strategy_signals.values())
            total_strategies = len(strategy_signals)
            signal_correlation = signal_count / total_strategies if total_strategies > 0 else 0.0
            
            # Calculate confidence score
            avg_confidence = np.mean(list(signal_confidences.values())) if signal_confidences else 0.0
            confidence_score = signal_correlation * avg_confidence
            
            # Determine recommended action
            if signal_correlation >= 0.7 and confidence_score > 0.6:
                recommended_action = "STRONG_BUY"
            elif signal_correlation >= 0.5 and confidence_score > 0.4:
                recommended_action = "BUY"
            elif signal_correlation >= 0.3:
                recommended_action = "WEAK_BUY"
            elif signal_correlation <= 0.1:
                recommended_action = "NO_CONSENSUS"
            else:
                recommended_action = "HOLD"
            
            # Identify conflicting strategies
            conflicting_strategies = []
            if 0.2 < signal_correlation < 0.8:  # Mixed signals
                buying_strategies = [name for name, signal in strategy_signals.items() if signal]
                not_buying_strategies = [name for name, signal in strategy_signals.items() if not signal]
                
                if len(buying_strategies) > 0 and len(not_buying_strategies) > 0:
                    conflicting_strategies = buying_strategies + not_buying_strategies
            
            cross_signal = CrossStrategySignal(
                timestamp=datetime.now(timezone.utc),
                strategy_signals=strategy_signals,
                signal_correlation=signal_correlation,
                confidence_score=confidence_score,
                recommended_action=recommended_action,
                conflicting_strategies=conflicting_strategies
            )
            
            self.cross_strategy_signals.append(cross_signal)
            
            # Update signal correlation tracking
            for i, name1 in enumerate(strategy_signals.keys()):
                for j, name2 in enumerate(strategy_signals.keys()):
                    if i < j:  # Avoid duplicate pairs
                        correlation_key = f"{name1}_{name2}"
                        signal_alignment = 1.0 if strategy_signals[name1] == strategy_signals[name2] else 0.0
                        
                        # Update running correlation
                        current_corr = self.signal_correlations[correlation_key]
                        self.signal_correlations[correlation_key] = current_corr * 0.9 + signal_alignment * 0.1
            
            return cross_signal
            
        except Exception as e:
            logger.error(f"Cross-strategy signal analysis error: {e}")
            return CrossStrategySignal(
                timestamp=datetime.now(timezone.utc),
                strategy_signals={},
                signal_correlation=0.0,
                confidence_score=0.0,
                recommended_action="ERROR"
            )

    async def optimize_strategy_allocations(self, market_regime: MarketRegime) -> Dict[str, float]:
        """⚖️ Optimize strategy allocations based on performance and market regime"""
        try:
            # Update performance metrics first
            await self.update_strategy_performances()
            
            new_allocations = {}
            total_weight = 0.0
            
            # Get active strategies
            active_strategies = {
                name: alloc for name, alloc in self.strategy_allocations.items()
                if alloc.status == StrategyStatus.ACTIVE
            }
            
            if not active_strategies:
                logger.warning("No active strategies for allocation optimization")
                return {}
            
            # Regime-based strategy preferences
            regime_preferences = self._get_regime_strategy_preferences(market_regime)
            
            # Calculate base scores for each strategy
            strategy_scores = {}
            for strategy_name, allocation in active_strategies.items():
                performance = self.strategy_performances[strategy_name]
                
                # Base performance score
                base_score = allocation.performance_score / 100.0
                
                # Regime adjustment
                regime_multiplier = regime_preferences.get(strategy_name, 1.0)
                
                # Recent performance weight
                recent_performance_weight = 1.0 + (performance.recent_performance_trend / 10.0)
                recent_performance_weight = max(0.5, min(2.0, recent_performance_weight))
                
                # Risk adjustment (favor lower drawdown)
                risk_adjustment = max(0.5, 1.0 - (performance.max_drawdown_pct / 100.0))
                
                # Trade frequency adjustment
                trade_frequency_score = min(1.2, max(0.8, performance.total_trades / 50.0))
                
                # Final score
                final_score = base_score * regime_multiplier * recent_performance_weight * risk_adjustment * trade_frequency_score
                strategy_scores[strategy_name] = max(0.1, min(3.0, final_score))
            
            # Normalize scores to allocations
            total_score = sum(strategy_scores.values())
            
            for strategy_name, score in strategy_scores.items():
                # Calculate target weight
                target_weight = score / total_score if total_score > 0 else 1.0 / len(strategy_scores)
                
                # Apply min/max constraints
                allocation = active_strategies[strategy_name]
                target_weight = max(allocation.min_weight, min(allocation.max_weight, target_weight))
                
                new_allocations[strategy_name] = target_weight
                total_weight += target_weight
            
            # Normalize to ensure total doesn't exceed max exposure
            if total_weight > self.max_total_exposure:
                scale_factor = self.max_total_exposure / total_weight
                new_allocations = {name: weight * scale_factor for name, weight in new_allocations.items()}
            
            # Update allocations
            for strategy_name, new_weight in new_allocations.items():
                if strategy_name in self.strategy_allocations:
                    old_weight = self.strategy_allocations[strategy_name].target_weight
                    self.strategy_allocations[strategy_name].target_weight = new_weight
                    
                    # Log significant changes
                    if abs(new_weight - old_weight) > 0.05:
                        logger.info(f"🔄 {strategy_name} allocation: {old_weight:.1%} → {new_weight:.1%}")
            
            return new_allocations
            
        except Exception as e:
            logger.error(f"Strategy allocation optimization error: {e}")
            return {}

    def _get_regime_strategy_preferences(self, regime: MarketRegime) -> Dict[str, float]:
        """🎯 Get strategy preferences based on market regime"""
        preferences = {
            MarketRegime.TRENDING_UP: {
                'MomentumOptimized': 1.4,  # Momentum works well in trends
                'MACDMLStrategy': 1.3,     # MACD good for trend following
                'VolumeProfileML': 1.2,    # Volume analysis helps confirm trends
                'BollingerML': 0.8,        # Mean reversion less effective
                'RSIML': 0.9               # RSI less effective in strong trends
            },
            MarketRegime.TRENDING_DOWN: {
                'MomentumOptimized': 1.3,  # Momentum can work both ways
                'MACDMLStrategy': 1.4,     # MACD good for trend following
                'VolumeProfileML': 1.1,    # Volume analysis helps
                'BollingerML': 0.9,        # Mean reversion less effective
                'RSIML': 1.0               # RSI can help with oversold
            },
            MarketRegime.SIDEWAYS: {
                'BollingerML': 1.5,        # Mean reversion excels in ranges
                'RSIML': 1.4,              # RSI great for range trading
                'VolumeProfileML': 1.2,    # Volume levels important
                'MomentumOptimized': 0.7,  # Momentum less effective
                'MACDMLStrategy': 0.8      # MACD less effective
            },
            MarketRegime.VOLATILE: {
                'VolumeProfileML': 1.4,    # Volume analysis crucial
                'BollingerML': 1.2,        # Volatility indicators help
                'RSIML': 1.1,              # RSI can help with extremes
                'MomentumOptimized': 0.8,  # Momentum harder in volatility
                'MACDMLStrategy': 0.9      # MACD gets whipsawed
            },
            MarketRegime.LOW_VOLATILITY: {
                'MomentumOptimized': 1.2,  # Momentum can work when it starts
                'MACDMLStrategy': 1.1,     # MACD good for early signals
                'VolumeProfileML': 1.3,    # Volume changes important
                'BollingerML': 0.9,        # Less volatility to exploit
                'RSIML': 0.8               # Less extreme moves
            },
            MarketRegime.BREAKOUT: {
                'MomentumOptimized': 1.5,  # Momentum excels at breakouts
                'VolumeProfileML': 1.4,    # Volume confirmation crucial
                'MACDMLStrategy': 1.3,     # MACD good for confirmation
                'BollingerML': 1.1,        # Band breakouts
                'RSIML': 0.9               # RSI may give false signals
            },
            MarketRegime.REVERSAL: {
                'RSIML': 1.4,              # RSI great for reversals
                'BollingerML': 1.3,        # Mean reversion works
                'VolumeProfileML': 1.2,    # Volume divergence important
                'MACDMLStrategy': 1.1,     # MACD divergence
                'MomentumOptimized': 0.7   # Momentum fights the reversal
            }
        }
        
        return preferences.get(regime, {name: 1.0 for name in self.strategies.keys()})

    async def should_rebalance(self) -> bool:
        """🔄 Determine if portfolio rebalancing is needed"""
        try:
            # Time-based rebalancing
            if self.last_rebalance_time is None:
                return True
            
            time_since_rebalance = datetime.now(timezone.utc) - self.last_rebalance_time
            if time_since_rebalance.total_seconds() >= self.rebalancing_frequency_hours * 3600:
                return True
            
            # Threshold-based rebalancing
            total_deviation = 0.0
            for strategy_name, allocation in self.strategy_allocations.items():
                if allocation.status == StrategyStatus.ACTIVE:
                    deviation = abs(allocation.current_weight - allocation.target_weight)
                    total_deviation += deviation
            
            if total_deviation >= self.min_rebalancing_threshold:
                logger.info(f"🔄 Rebalancing triggered by deviation: {total_deviation:.3f}")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Rebalancing check error: {e}")
            return False

    async def coordinate_strategies(self, df: pd.DataFrame) -> Dict[str, Any]:
        """🎯 Main coordination function - orchestrate all strategies"""
        try:
            coordination_start_time = datetime.now(timezone.utc)
            self.total_coordination_cycles += 1
            
            coordination_result = {
                'timestamp': coordination_start_time,
                'market_regime': None,
                'cross_strategy_signal': None,
                'allocations_updated': False,
                'active_strategies': len([s for s in self.strategy_allocations.values() if s.status == StrategyStatus.ACTIVE]),
                'coordination_success': False,
                'execution_time_ms': 0,
                'recommendations': []
            }
            
            # 1. Analyze market regime
            market_regime = await self.analyze_market_regime(df)
            coordination_result['market_regime'] = {
                'regime': market_regime.regime_name,
                'confidence': self.market_regime_confidence,
                'description': market_regime.description
            }
            
            # 2. Analyze cross-strategy signals
            if self.enable_cross_validation:
                cross_signal = await self.analyze_cross_strategy_signals(df)
                coordination_result['cross_strategy_signal'] = {
                    'correlation': cross_signal.signal_correlation,
                    'confidence': cross_signal.confidence_score,
                    'recommended_action': cross_signal.recommended_action,
                    'conflicting_strategies': cross_signal.conflicting_strategies,
                    'strategy_signals': cross_signal.strategy_signals
                }
            
            # 3. Check if rebalancing is needed
            if await self.should_rebalance():
                # Optimize allocations
                if self.enable_regime_switching:
                    new_allocations = await self.optimize_strategy_allocations(market_regime)
                    if new_allocations:
                        coordination_result['allocations_updated'] = True
                        coordination_result['new_allocations'] = new_allocations
                        self.last_rebalance_time = coordination_start_time
                        
                        logger.info(f"🎯 Portfolio rebalanced for {market_regime.regime_name} regime")
            
            # 4. Risk management checks
            risk_warnings = await self._perform_risk_checks()
            if risk_warnings:
                coordination_result['risk_warnings'] = risk_warnings
            
            # 5. Generate recommendations
            recommendations = await self._generate_coordination_recommendations(
                market_regime, 
                coordination_result.get('cross_strategy_signal')
            )
            coordination_result['recommendations'] = recommendations
            
            # 6. Update analytics
            execution_time = (datetime.now(timezone.utc) - coordination_start_time).total_seconds() * 1000
            coordination_result['execution_time_ms'] = execution_time
            coordination_result['coordination_success'] = True
            
            self.successful_coordinations += 1
            self.coordination_analytics.append(coordination_result)
            
            # Log coordination summary
            active_strategies = sum(1 for s in self.strategy_allocations.values() if s.status == StrategyStatus.ACTIVE)
            logger.info(f"🎯 Coordination cycle completed: {active_strategies} strategies active, "
                       f"regime={market_regime.regime_name}, execution={execution_time:.1f}ms")
            
            return coordination_result
            
        except Exception as e:
            self.coordination_errors += 1
            logger.error(f"Strategy coordination error: {e}")
            coordination_result['coordination_success'] = False
            coordination_result['error'] = str(e)
            return coordination_result

    async def _perform_risk_checks(self) -> List[str]:
        """🛡️ Perform risk management checks"""
        warnings = []
        
        try:
            # Check total exposure
            total_exposure = sum(alloc.target_weight for alloc in self.strategy_allocations.values() 
                               if alloc.status == StrategyStatus.ACTIVE)
            
            if total_exposure > self.max_total_exposure:
                warnings.append(f"Total exposure {total_exposure:.1%} exceeds limit {self.max_total_exposure:.1%}")
            
            # Check single strategy exposure
            for name, alloc in self.strategy_allocations.items():
                if alloc.target_weight > self.risk_limits['max_single_strategy_exposure']:
                    warnings.append(f"Strategy {name} exposure {alloc.target_weight:.1%} exceeds single strategy limit")
            
            # Check correlation risks
            high_correlation_pairs = []
            for key, correlation in self.signal_correlations.items():
                if correlation > self.risk_limits['max_correlation_threshold']:
                    high_correlation_pairs.append(f"{key}={correlation:.2f}")
            
            if high_correlation_pairs:
                warnings.append(f"High strategy correlations detected: {', '.join(high_correlation_pairs)}")
            
            # Check active position counts
            active_positions = len([pos for pos in self.portfolio.positions if pos.status == "OPEN"])
            if active_positions > self.risk_limits['max_concurrent_positions']:
                warnings.append(f"Active positions {active_positions} exceeds limit {self.risk_limits['max_concurrent_positions']}")
            
            return warnings
            
        except Exception as e:
            logger.error(f"Risk check error: {e}")
            return [f"Risk check error: {str(e)}"]

    async def _generate_coordination_recommendations(
        self, 
        market_regime: MarketRegime, 
        cross_signal: Optional[Dict]
    ) -> List[str]:
        """💡 Generate coordination recommendations"""
        recommendations = []
        
        try:
            # Regime-based recommendations
            if market_regime == MarketRegime.TRENDING_UP:
                recommendations.append("Consider increasing momentum and MACD strategy allocations")
            elif market_regime == MarketRegime.SIDEWAYS:
                recommendations.append("Favor mean reversion strategies (Bollinger, RSI)")
            elif market_regime == MarketRegime.VOLATILE:
                recommendations.append("Emphasize volume analysis and reduce momentum exposure")
            elif market_regime == MarketRegime.BREAKOUT:
                recommendations.append("Prioritize momentum and volume confirmation strategies")
            
            # Cross-signal recommendations
            if cross_signal:
                correlation = cross_signal.get('correlation', 0)
                confidence = cross_signal.get('confidence', 0)
                
                if correlation > 0.8 and confidence > 0.7:
                    recommendations.append("Strong consensus signal - consider increasing position sizes")
                elif correlation < 0.3:
                    recommendations.append("No strategy consensus - wait for clearer signals")
                elif cross_signal.get('conflicting_strategies'):
                    recommendations.append("Strategy conflict detected - review individual strategy conditions")
            
            # Performance-based recommendations
            best_performers = sorted(
                self.strategy_performances.items(),
                key=lambda x: x[1].recent_performance_trend,
                reverse=True
            )[:2]
            
            if best_performers:
                best_strategy = best_performers[0][0]
                recommendations.append(f"Consider increasing allocation to top performer: {best_strategy}")
            
            return recommendations
            
        except Exception as e:
            logger.error(f"Recommendation generation error: {e}")
            return ["Recommendation generation error"]

    def get_coordination_analytics(self) -> Dict[str, Any]:
        """📊 Get comprehensive coordination analytics"""
        try:
            # Calculate success metrics
            success_rate = (self.successful_coordinations / max(1, self.total_coordination_cycles)) * 100
            error_rate = (self.coordination_errors / max(1, self.total_coordination_cycles)) * 100
            
            # Recent performance
            recent_analytics = list(self.coordination_analytics)[-20:] if self.coordination_analytics else []
            avg_execution_time = np.mean([a['execution_time_ms'] for a in recent_analytics]) if recent_analytics else 0
            
            # Strategy status overview
            strategy_status_counts = defaultdict(int)
            for alloc in self.strategy_allocations.values():
                strategy_status_counts[alloc.status.value] += 1
            
            # Allocation distribution
            total_allocation = sum(alloc.target_weight for alloc in self.strategy_allocations.values())
            allocation_distribution = {
                name: alloc.target_weight for name, alloc in self.strategy_allocations.items()
                if alloc.target_weight > 0
            }
            
            analytics = {
                'system_overview': {
                    'total_coordination_cycles': self.total_coordination_cycles,
                    'successful_coordinations': self.successful_coordinations,
                    'coordination_errors': self.coordination_errors,
                    'success_rate_pct': success_rate,
                    'error_rate_pct': error_rate,
                    'avg_execution_time_ms': avg_execution_time
                },
                
                'current_state': {
                    'active_strategies': len([s for s in self.strategy_allocations.values() if s.status == StrategyStatus.ACTIVE]),
                    'total_allocation': total_allocation,
                    'market_regime': self.current_market_regime.regime_name,
                    'regime_confidence': self.market_regime_confidence,
                    'last_rebalance': self.last_rebalance_time.isoformat() if self.last_rebalance_time else None
                },
                
                'strategy_status': dict(strategy_status_counts),
                'allocation_distribution': allocation_distribution,
                
                'performance_summary': {
                    strategy_name: {
                        'total_trades': perf.total_trades,
                        'win_rate': perf.win_rate,
                        'avg_profit_pct': perf.avg_profit_pct,
                        'performance_score': self.strategy_allocations[strategy_name].performance_score
                    }
                    for strategy_name, perf in self.strategy_performances.items()
                },
                
                'risk_metrics': {
                    'total_exposure': total_allocation,
                    'max_single_strategy_exposure': max((alloc.target_weight for alloc in self.strategy_allocations.values()), default=0),
                    'strategy_count': len(self.strategies),
                    'diversification_score': self._calculate_diversification_score()
                },
                
                'recent_market_regimes': [
                    {
                        'timestamp': r['timestamp'].isoformat(),
                        'regime': r['regime'].regime_name,
                        'confidence': r['confidence']
                    }
                    for r in list(self.market_regime_history)[-10:]
                ]
            }
            
            return analytics
            
        except Exception as e:
            logger.error(f"Coordination analytics error: {e}")
            return {'error': str(e)}

    def _calculate_diversification_score(self) -> float:
        """📊 Calculate portfolio diversification score"""
        try:
            active_allocations = [
                alloc.target_weight for alloc in self.strategy_allocations.values()
                if alloc.status == StrategyStatus.ACTIVE and alloc.target_weight > 0
            ]
            
            if len(active_allocations) <= 1:
                return 0.0
            
            # Calculate Herfindahl-Hirschman Index (HHI) and convert to diversification score
            total_weight = sum(active_allocations)
            if total_weight == 0:
                return 0.0
            
            normalized_weights = [w / total_weight for w in active_allocations]
            hhi = sum(w**2 for w in normalized_weights)
            
            # Convert HHI to diversification score (0-1, higher is better)
            max_hhi = 1.0  # All weight in one strategy
            min_hhi = 1.0 / len(normalized_weights)  # Equal weights
            
            if max_hhi == min_hhi:
                return 1.0
            
            diversification_score = (max_hhi - hhi) / (max_hhi - min_hhi)
            return max(0.0, min(1.0, diversification_score))
            
        except Exception as e:
            logger.debug(f"Diversification score calculation error: {e}")
            return 0.5

    def get_strategy_allocation(self, strategy_name: str) -> Optional[float]:
        """📊 Get current allocation for a specific strategy"""
        if strategy_name in self.strategy_allocations:
            return self.strategy_allocations[strategy_name].target_weight
        return None

    def set_strategy_status(self, strategy_name: str, status: StrategyStatus) -> bool:
        """⚙️ Set strategy status"""
        try:
            if strategy_name in self.strategy_allocations:
                old_status = self.strategy_allocations[strategy_name].status
                self.strategy_allocations[strategy_name].status = status
                
                if status == StrategyStatus.INACTIVE:
                    self.strategy_allocations[strategy_name].target_weight = 0.0
                
                logger.info(f"📝 Strategy {strategy_name} status: {old_status.value} → {status.value}")
                return True
            return False
            
        except Exception as e:
            logger.error(f"Strategy status update error: {e}")
            return False

# Integration function for main trading system
def integrate_strategy_coordinator(
    portfolio_instance: Portfolio,
    strategies: List[Tuple[str, Any]],
    symbol: str = "BTC/USDT",
    **coordinator_config
) -> StrategyCoordinator:
    """
    Integrate Strategy Coordinator into existing trading system
    
    Args:
        portfolio_instance: Main portfolio instance
        strategies: List of (strategy_name, strategy_instance) tuples
        symbol: Trading symbol
        **coordinator_config: Coordinator configuration parameters
        
    Returns:
        StrategyCoordinator: Configured coordinator instance
    """
    try:
        # Create coordinator
        coordinator = StrategyCoordinator(
            portfolio=portfolio_instance,
            symbol=symbol,
            **coordinator_config
        )
        
        # Register all strategies
        for strategy_name, strategy_instance in strategies:
            coordinator.register_strategy(strategy_name, strategy_instance)
        
        # Add coordinator to portfolio
        portfolio_instance.strategy_coordinator = coordinator
        
        logger.info(f"🎯 Strategy Coordinator integrated successfully")
        logger.info(f"   📝 Registered strategies: {', '.join([name for name, _ in strategies])}")
        logger.info(f"   🔄 Rebalancing: every {coordinator.rebalancing_frequency_hours}h")
        logger.info(f"   💰 Max exposure: {coordinator.max_total_exposure:.1%}")
        
        return coordinator
        
    except Exception as e:
        logger.error(f"Strategy Coordinator integration error: {e}")
        raise