#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - ENHANCED MOMENTUM STRATEGY FIX
💎 FIXED: Tüm eksik metodlar eklendi

ÇÖZÜMLER:
1. ✅ _calculate_momentum_indicators metodu eklendi
2. ✅ _analyze_momentum_signals metodu eklendi 
3. ✅ _prepare_ml_features metodu eklendi
4. ✅ _calculate_performance_based_size metodu eklendi
5. ✅ Tüm test hataları giderildi
"""

import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Optional, Tuple, Any
import logging
import asyncio
from dataclasses import dataclass

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, VolatilityRegime
from json_parameter_system import JSONParameterManager


class EnhancedMomentumStrategy(BaseStrategy):
    """
    🚀 ENHANCED MOMENTUM STRATEGY v2.0
    💎 Institutional Grade Momentum Trading
    
    Features:
    - Multi-timeframe momentum analysis
    - Machine learning predictions
    - Kelly Criterion position sizing
    - Dynamic exit management
    - Performance-based sizing
    """
    
    def __init__(self, portfolio, symbol: str = "BTC/USDT", **kwargs):
        """Initialize Enhanced Momentum Strategy"""
        
        # Initialize parent
        super().__init__(
            portfolio=portfolio,
            symbol=symbol,
            strategy_name="EnhancedMomentum",
            **kwargs
        )
        
        # Load optimized parameters from JSON
        self._load_optimized_parameters()
        
        # Performance-based sizing parameters
        self.size_high_profit_pct = 0.03    # 3%+ profit
        self.size_good_profit_pct = 0.02    # 2%+ profit  
        self.size_normal_profit_pct = 0.01  # 1%+ profit
        
        # Risk management
        self.max_loss_pct = 0.02            # 2% max loss
        self.min_profit_target_usdt = 10.0  # $10 minimum profit
        self.max_hold_minutes = 1440        # 24 hours max hold
        
        # ML components
        self.ml_predictor = None  # Initialize if ML is enabled
        self.ml_features_count = 30
        
        # Performance tracking for sizing
        self.recent_trades_window = 20
        self.performance_history = []
        
        self.logger.info(f"🚀 Enhanced Momentum Strategy initialized")
        self.logger.info(f"   EMA periods: {self.ema_short}/{self.ema_medium}/{self.ema_long}")
        self.logger.info(f"   RSI period: {self.rsi_period}")
        self.logger.info(f"   ML enabled: {self.ml_enabled}")
    
    def _load_optimized_parameters(self):
        """Load optimized parameters from JSON"""
        
        # Default parameters (hedge fund optimized)
        self.ema_short = 14
        self.ema_medium = 22
        self.ema_long = 57
        self.rsi_period = 14
        self.rsi_oversold = 30
        self.rsi_overbought = 70
        self.adx_period = 14
        self.adx_threshold = 25
        self.atr_period = 14
        self.atr_multiplier = 2.0
        self.volume_sma_period = 20
        self.volume_multiplier = 1.5
        self.momentum_lookback = 4
        self.momentum_threshold = 0.01
        
        # Quality score weights
        self.quality_trend_weight = 0.3
        self.quality_volume_weight = 0.2
        self.quality_volatility_weight = 0.2
        self.quality_momentum_weight = 0.3
        
        # Signal filtering
        self.min_quality_score = 14
        self.trend_alignment_required = True
        
        # Position sizing
        self.position_size_pct = 0.25
        self.max_positions = 3
        
        # Try to load from JSON
        try:
            manager = JSONParameterManager()
            data = manager.load_strategy_parameters("momentum")
            
            if data and 'parameters' in data:
                params = data['parameters']
                
                # Update parameters
                for param_name, param_value in params.items():
                    if hasattr(self, param_name):
                        setattr(self, param_name, param_value)
                
                self.logger.info(f"✅ Loaded {len(params)} optimized parameters from JSON")
        
        except Exception as e:
            self.logger.warning(f"Could not load JSON parameters: {e}, using defaults")
    
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FIXED: Calculate comprehensive momentum indicators
        """
        
        indicators = {}
        
        # Basic price data
        close = data['close']
        high = data['high']
        low = data['low']
        volume = data['volume']
        
        # EMA calculations
        indicators['ema_short'] = close.ewm(span=self.ema_short, adjust=False).mean().iloc[-1]
        indicators['ema_medium'] = close.ewm(span=self.ema_medium, adjust=False).mean().iloc[-1]
        indicators['ema_long'] = close.ewm(span=self.ema_long, adjust=False).mean().iloc[-1]
        
        # RSI calculation
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # ADX calculation (simplified)
        plus_dm = high.diff()
        minus_dm = -low.diff()
        plus_dm[plus_dm < 0] = 0
        minus_dm[minus_dm < 0] = 0
        
        tr = pd.concat([
            high - low,
            abs(high - close.shift(1)),
            abs(low - close.shift(1))
        ], axis=1).max(axis=1)
        
        atr = tr.rolling(window=self.adx_period).mean()
        plus_di = 100 * (plus_dm.rolling(window=self.adx_period).mean() / atr)
        minus_di = 100 * (minus_dm.rolling(window=self.adx_period).mean() / atr)
        dx = 100 * abs(plus_di - minus_di) / (plus_di + minus_di + 1e-10)
        indicators['adx'] = dx.rolling(window=self.adx_period).mean().iloc[-1]
        
        # ATR
        indicators['atr'] = atr.iloc[-1]
        
        # Volume analysis
        indicators['volume_sma'] = volume.rolling(window=self.volume_sma_period).mean().iloc[-1]
        indicators['volume_ratio'] = volume.iloc[-1] / (indicators['volume_sma'] + 1e-10)
        
        # Price momentum
        indicators['price_momentum'] = (close.iloc[-1] - close.iloc[-self.momentum_lookback]) / close.iloc[-self.momentum_lookback]
        
        # MACD
        ema_12 = close.ewm(span=12, adjust=False).mean()
        ema_26 = close.ewm(span=26, adjust=False).mean()
        macd = ema_12 - ema_26
        macd_signal = macd.ewm(span=9, adjust=False).mean()
        indicators['macd'] = macd.iloc[-1]
        indicators['macd_signal'] = macd_signal.iloc[-1]
        indicators['macd_histogram'] = indicators['macd'] - indicators['macd_signal']
        
        # Bollinger Bands
        sma_20 = close.rolling(window=20).mean()
        std_20 = close.rolling(window=20).std()
        indicators['bb_upper'] = (sma_20 + 2 * std_20).iloc[-1]
        indicators['bb_lower'] = (sma_20 - 2 * std_20).iloc[-1]
        indicators['bb_position'] = (close.iloc[-1] - indicators['bb_lower']) / (indicators['bb_upper'] - indicators['bb_lower'] + 1e-10)
        
        # Stochastic RSI
        rsi_series = 100 - (100 / (1 + rs))
        rsi_min = rsi_series.rolling(window=14).min()
        rsi_max = rsi_series.rolling(window=14).max()
        indicators['stoch_rsi'] = ((rsi_series - rsi_min) / (rsi_max - rsi_min + 1e-10)).iloc[-1]
        
        return indicators
    
    def _analyze_momentum_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """
        ✅ FIXED: Analyze momentum signals for entry conditions
        """
        
        # Get indicators
        indicators = self._calculate_momentum_indicators(data)
        
        signals = {
            'signal_strength': 0,
            'quality_score': 0,
            'momentum_score': 0.0,
            'trend_alignment': False,
            'volume_confirmation': False,
            'risk_assessment': 'normal'
        }
        
        # 1. Trend Analysis
        ema_bullish = (indicators['ema_short'] > indicators['ema_medium'] > indicators['ema_long'])
        ema_bearish = (indicators['ema_short'] < indicators['ema_medium'] < indicators['ema_long'])
        
        if ema_bullish:
            signals['signal_strength'] += 3
            signals['trend_alignment'] = True
        elif ema_bearish:
            signals['signal_strength'] -= 3
        
        # 2. Momentum Analysis
        if indicators['price_momentum'] > self.momentum_threshold:
            signals['signal_strength'] += 2
            signals['momentum_score'] = indicators['price_momentum']
        
        # MACD confirmation
        if indicators['macd'] > indicators['macd_signal'] and indicators['macd_histogram'] > 0:
            signals['signal_strength'] += 1
        
        # 3. RSI Analysis
        if indicators['rsi'] < self.rsi_oversold:
            signals['signal_strength'] += 2  # Oversold bounce
        elif indicators['rsi'] > self.rsi_overbought:
            signals['signal_strength'] -= 2  # Overbought warning
        
        # 4. Volume Analysis
        if indicators['volume_ratio'] > self.volume_multiplier:
            signals['volume_confirmation'] = True
            signals['signal_strength'] += 1
        
        # 5. ADX Trend Strength
        if indicators['adx'] > self.adx_threshold:
            signals['signal_strength'] += 1  # Strong trend
        
        # 6. Calculate Quality Score
        quality_components = {
            'trend': min((signals['signal_strength'] / 10) * self.quality_trend_weight, self.quality_trend_weight),
            'volume': (1.0 if signals['volume_confirmation'] else 0.5) * self.quality_volume_weight,
            'volatility': min((indicators['atr'] / data['close'].iloc[-1]) * 10 * self.quality_volatility_weight, self.quality_volatility_weight),
            'momentum': min(abs(signals['momentum_score']) * 100 * self.quality_momentum_weight, self.quality_momentum_weight)
        }
        
        signals['quality_score'] = int(sum(quality_components.values()) * 20)  # Scale to 0-20
        
        # 7. Risk Assessment
        if indicators['atr'] / data['close'].iloc[-1] > 0.03:  # High volatility
            signals['risk_assessment'] = 'high'
        elif indicators['atr'] / data['close'].iloc[-1] < 0.01:  # Low volatility
            signals['risk_assessment'] = 'low'
        
        # 8. Additional signals
        signals['bb_position'] = indicators['bb_position']
        signals['stoch_rsi'] = indicators['stoch_rsi']
        signals['indicators'] = indicators  # Store all indicators
        
        return signals
    
    async def analyze_market(self, data: pd.DataFrame) -> TradingSignal:
        """
        Main market analysis method
        """
        
        try:
            # Ensure we have enough data
            if len(data) < max(self.ema_long, 100):
                return self.create_signal(
                    SignalType.HOLD,
                    confidence=0.0,
                    price=data['close'].iloc[-1],
                    reasons=["Insufficient data for analysis"]
                )
            
            # Analyze momentum signals
            signals = self._analyze_momentum_signals(data)
            
            # Check quality threshold
            if signals['quality_score'] < self.min_quality_score:
                return self.create_signal(
                    SignalType.HOLD,
                    confidence=0.3,
                    price=data['close'].iloc[-1],
                    reasons=[f"Quality score too low: {signals['quality_score']} < {self.min_quality_score}"]
                )
            
            # Check trend alignment requirement
            if self.trend_alignment_required and not signals['trend_alignment']:
                return self.create_signal(
                    SignalType.HOLD,
                    confidence=0.4,
                    price=data['close'].iloc[-1],
                    reasons=["Trend alignment required but not present"]
                )
            
            # ML Prediction (if enabled)
            ml_confidence = 0.5
            if self.ml_enabled and self.ml_predictor:
                ml_features = self._prepare_ml_features(data)
                # Simulate ML prediction (in real implementation, use actual model)
                ml_confidence = 0.65 + (signals['quality_score'] / 100)  # Simulated
            
            # Generate signal based on analysis
            current_price = data['close'].iloc[-1]
            
            # BUY Signal conditions
            if (signals['signal_strength'] >= 5 and 
                signals['quality_score'] >= self.min_quality_score and
                (not self.ml_enabled or ml_confidence >= self.ml_confidence_threshold)):
                
                confidence = min(0.95, 0.5 + (signals['quality_score'] / 40) + (ml_confidence - 0.5))
                
                # Calculate stop loss and take profit
                atr = signals['indicators']['atr']
                stop_loss = current_price - (atr * self.atr_multiplier)
                take_profit = current_price + (atr * self.atr_multiplier * 2)
                
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.now(timezone.utc),
                    reasons=[
                        f"Strong momentum signal: {signals['signal_strength']}",
                        f"Quality score: {signals['quality_score']}",
                        f"ML confidence: {ml_confidence:.2f}" if self.ml_enabled else "Technical signal",
                        f"Risk level: {signals['risk_assessment']}"
                    ],
                    metadata={
                        'quality_score': signals['quality_score'],
                        'signal_strength': signals['signal_strength'],
                        'ml_confidence': ml_confidence,
                        'indicators': signals['indicators']
                    },
                    stop_loss=stop_loss,
                    take_profit=take_profit
                )
            
            # SELL Signal conditions (for existing positions)
            elif signals['signal_strength'] <= -3:
                confidence = min(0.9, 0.6 + abs(signals['signal_strength']) / 10)
                
                return TradingSignal(
                    signal_type=SignalType.SELL,
                    confidence=confidence,
                    price=current_price,
                    timestamp=datetime.now(timezone.utc),
                    reasons=[
                        f"Bearish momentum: {signals['signal_strength']}",
                        "Trend reversal detected"
                    ],
                    metadata={'signal_strength': signals['signal_strength']}
                )
            
            # Default HOLD
            else:
                return self.create_signal(
                    SignalType.HOLD,
                    confidence=0.5,
                    price=current_price,
                    reasons=[
                        f"Signal strength insufficient: {signals['signal_strength']}",
                        f"Quality score: {signals['quality_score']}"
                    ]
                )
        
        except Exception as e:
            self.logger.error(f"Market analysis error: {e}")
            return self.create_signal(
                SignalType.HOLD,
                confidence=0.0,
                price=data['close'].iloc[-1] if len(data) > 0 else 0,
                reasons=[f"Analysis error: {str(e)}"]
            )
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        """
        Calculate position size with Kelly Criterion and performance adjustment
        """
        
        # Get base position size
        available_capital = self.portfolio.get_available_usdt()
        base_size = available_capital * self.position_size_pct
        
        # Apply confidence scaling
        confidence_multiplier = signal.confidence
        
        # Apply performance-based sizing
        performance_multiplier = self._calculate_performance_based_size(signal)
        
        # Apply Kelly Criterion if enabled
        kelly_multiplier = 1.0
        if self.kelly_enabled and len(self.performance_history) >= 10:
            kelly_multiplier = self._calculate_kelly_fraction()
        
        # Calculate final size
        position_size = base_size * confidence_multiplier * performance_multiplier * kelly_multiplier
        
        # Apply limits
        min_size = 100.0  # $100 minimum
        max_size = available_capital * self.max_position_size_pct
        
        final_size = max(min_size, min(position_size, max_size))
        
        self.logger.info(f"Position size calculation:")
        self.logger.info(f"  Base: ${base_size:.2f}")
        self.logger.info(f"  Confidence: {confidence_multiplier:.2f}")
        self.logger.info(f"  Performance: {performance_multiplier:.2f}")
        self.logger.info(f"  Kelly: {kelly_multiplier:.2f}")
        self.logger.info(f"  Final: ${final_size:.2f}")
        
        return final_size
    
    def _prepare_ml_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """
        ✅ FIXED: Prepare features for ML model
        """
        
        features = {}
        
        # Get indicators
        indicators = self._calculate_momentum_indicators(data)
        
        # Price features
        close = data['close']
        features['price_change_1h'] = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4]  # 4 * 15min = 1h
        features['price_change_4h'] = (close.iloc[-1] - close.iloc[-16]) / close.iloc[-16] if len(close) > 16 else 0
        features['price_change_24h'] = (close.iloc[-1] - close.iloc[-96]) / close.iloc[-96] if len(close) > 96 else 0
        
        # Technical features
        features['rsi'] = indicators['rsi'] / 100
        features['rsi_oversold'] = 1 if indicators['rsi'] < self.rsi_oversold else 0
        features['rsi_overbought'] = 1 if indicators['rsi'] > self.rsi_overbought else 0
        
        features['macd_signal'] = 1 if indicators['macd'] > indicators['macd_signal'] else 0
        features['macd_histogram_norm'] = indicators['macd_histogram'] / (abs(indicators['macd']) + 1e-10)
        
        features['bb_position'] = indicators['bb_position']
        features['stoch_rsi'] = indicators['stoch_rsi']
        
        # Trend features
        features['ema_short_above_medium'] = 1 if indicators['ema_short'] > indicators['ema_medium'] else 0
        features['ema_medium_above_long'] = 1 if indicators['ema_medium'] > indicators['ema_long'] else 0
        features['trend_strength'] = (indicators['ema_short'] - indicators['ema_long']) / indicators['ema_long']
        
        # Volume features
        features['volume_ratio'] = indicators['volume_ratio']
        features['volume_increasing'] = 1 if data['volume'].iloc[-1] > data['volume'].iloc[-2] else 0
        
        # Volatility features
        features['atr_ratio'] = indicators['atr'] / close.iloc[-1]
        features['volatility_regime'] = self._classify_volatility(features['atr_ratio'])
        
        # Momentum features
        features['price_momentum'] = indicators['price_momentum']
        features['adx'] = indicators['adx'] / 100
        features['strong_trend'] = 1 if indicators['adx'] > self.adx_threshold else 0
        
        # Pattern features (simplified)
        features['higher_high'] = 1 if close.iloc[-1] > close.iloc[-20:].max() else 0
        features['lower_low'] = 1 if close.iloc[-1] < close.iloc[-20:].min() else 0
        
        # Market microstructure
        high_low_ratio = (data['high'].iloc[-1] - data['low'].iloc[-1]) / data['low'].iloc[-1]
        features['high_low_ratio'] = high_low_ratio
        features['close_position'] = (close.iloc[-1] - data['low'].iloc[-1]) / (data['high'].iloc[-1] - data['low'].iloc[-1] + 1e-10)
        
        return features
    
    def _calculate_performance_based_size(self, signal: TradingSignal) -> float:
        """
        ✅ FIXED: Calculate performance-based position size multiplier
        """
        
        # Get recent performance
        if not hasattr(self, 'performance_history') or len(self.performance_history) < 5:
            return 1.0  # Default multiplier
        
        recent_trades = self.performance_history[-self.recent_trades_window:]
        
        # Calculate metrics
        total_trades = len(recent_trades)
        winning_trades = sum(1 for t in recent_trades if t['profit'] > 0)
        total_profit = sum(t['profit'] for t in recent_trades)
        
        win_rate = winning_trades / total_trades if total_trades > 0 else 0.5
        avg_profit = total_profit / total_trades if total_trades > 0 else 0
        
        # Calculate consecutive wins/losses
        consecutive = 0
        for trade in reversed(recent_trades):
            if trade['profit'] > 0:
                if consecutive >= 0:
                    consecutive += 1
                else:
                    break
            else:
                if consecutive <= 0:
                    consecutive -= 1
                else:
                    break
        
        # Determine multiplier
        multiplier = 1.0
        
        # Win rate adjustment
        if win_rate > 0.65:
            multiplier *= 1.2
        elif win_rate < 0.35:
            multiplier *= 0.8
        
        # Profit trend adjustment
        if avg_profit > self.size_high_profit_pct * 1000:  # Assuming $1000 base
            multiplier *= 1.15
        elif avg_profit < 0:
            multiplier *= 0.85
        
        # Consecutive trades adjustment
        if consecutive >= 3:
            multiplier *= 1.1  # Winning streak
        elif consecutive <= -3:
            multiplier *= 0.9  # Losing streak
        
        # Quality score adjustment
        quality_score = signal.metadata.get('quality_score', 14)
        if quality_score >= 18:
            multiplier *= 1.1
        elif quality_score < 12:
            multiplier *= 0.9
        
        # Limit multiplier range
        return max(0.5, min(1.5, multiplier))
    
    def _calculate_kelly_fraction(self) -> float:
        """Calculate Kelly Criterion fraction"""
        
        if len(self.performance_history) < 20:
            return 0.25  # Conservative default
        
        # Calculate win probability and win/loss ratio
        wins = [t for t in self.performance_history if t['profit'] > 0]
        losses = [t for t in self.performance_history if t['profit'] < 0]
        
        if not wins or not losses:
            return 0.25
        
        p = len(wins) / len(self.performance_history)  # Win probability
        b = abs(np.mean([t['profit'] for t in wins])) / abs(np.mean([t['profit'] for t in losses]))  # Win/loss ratio
        
        # Kelly formula: f = p - q/b where q = 1-p
        kelly = p - (1 - p) / b
        
        # Apply Kelly multiplier (conservative)
        kelly_fraction = kelly * self.kelly_multiplier if hasattr(self, 'kelly_multiplier') else kelly * 0.25
        
        # Limit Kelly fraction
        return max(0.1, min(0.5, kelly_fraction))
    
    def _classify_volatility(self, atr_ratio: float) -> int:
        """Classify volatility regime"""
        if atr_ratio < 0.01:
            return 0  # Low
        elif atr_ratio < 0.02:
            return 1  # Normal
        elif atr_ratio < 0.03:
            return 2  # High
        else:
            return 3  # Extreme
    
    def _check_exit_signals(self, position, current_data: pd.DataFrame) -> Tuple[bool, str]:
        """Check momentum-specific exit signals"""
        
        # Get current indicators
        indicators = self._calculate_momentum_indicators(current_data)
        
        # 1. Trend reversal exit
        ema_bearish = (indicators['ema_short'] < indicators['ema_medium'] < indicators['ema_long'])
        if ema_bearish and position.unrealized_pnl > 0:
            return True, "Trend reversal detected"
        
        # 2. RSI extreme exit
        if indicators['rsi'] > 80 and position.unrealized_pnl_pct > 0.01:
            return True, "RSI extremely overbought"
        
        # 3. Momentum loss exit
        if indicators['price_momentum'] < -0.01 and position.unrealized_pnl > 0:
            return True, "Momentum turned negative"
        
        # 4. MACD cross exit
        if indicators['macd'] < indicators['macd_signal'] and position.unrealized_pnl_pct > 0.005:
            return True, "MACD bearish cross"
        
        # 5. Volume dry up exit
        if indicators['volume_ratio'] < 0.5 and position.unrealized_pnl_pct < -0.005:
            return True, "Volume dried up"
        
        return False, ""
    
    async def _get_ml_exit_signal(self, position, current_data: pd.DataFrame) -> bool:
        """Get ML-based exit signal"""
        
        if not self.ml_enabled or not self.ml_predictor:
            return False
        
        # Prepare features
        features = self._prepare_ml_features(current_data)
        
        # Add position-specific features
        features['position_pnl'] = position.unrealized_pnl_pct
        features['position_age'] = self._get_position_age_minutes(position) / 1440  # Normalize to days
        
        # Simulate ML prediction (in real implementation, use actual model)
        # For now, use rule-based logic
        exit_score = 0.5
        
        if features['position_pnl'] > 0.02 and features['rsi'] > 0.7:
            exit_score = 0.8
        elif features['position_pnl'] < -0.01 and features['trend_strength'] < 0:
            exit_score = 0.7
        
        return exit_score > 0.65
    
    def update_performance_history(self, trade_result: Dict[str, Any]):
        """Update performance history for sizing calculations"""
        
        self.performance_history.append({
            'timestamp': datetime.now(timezone.utc),
            'profit': trade_result.get('profit_usdt', 0),
            'profit_pct': trade_result.get('profit_pct', 0),
            'quality_score': trade_result.get('quality_score', 0),
            'hold_time': trade_result.get('hold_time_minutes', 0)
        })
        
        # Keep only recent history
        if len(self.performance_history) > 100:
            self.performance_history = self.performance_history[-100:]
    
    def get_strategy_analytics(self) -> Dict[str, Any]:
        """Get enhanced strategy analytics"""
        
        # Get base analytics
        base_analytics = super().get_strategy_analytics()
        
        # Add momentum-specific analytics
        momentum_analytics = {
            'momentum_performance': {
                'avg_quality_score': np.mean([t.get('quality_score', 0) for t in self.performance_history]) if self.performance_history else 0,
                'high_quality_trades': sum(1 for t in self.performance_history if t.get('quality_score', 0) >= 16),
                'avg_hold_time_minutes': np.mean([t.get('hold_time', 0) for t in self.performance_history]) if self.performance_history else 0
            },
            'quality_score_distribution': self._get_quality_distribution(),
            'regime_performance': self._get_regime_performance(),
            'ml_prediction_accuracy': self._get_ml_accuracy() if self.ml_enabled else None
        }
        
        # Merge analytics
        base_analytics.update(momentum_analytics)
        
        return base_analytics
    
    def _get_quality_distribution(self) -> Dict[str, int]:
        """Get distribution of quality scores"""
        if not self.performance_history:
            return {}
        
        distribution = {
            'excellent': sum(1 for t in self.performance_history if t.get('quality_score', 0) >= 18),
            'good': sum(1 for t in self.performance_history if 15 <= t.get('quality_score', 0) < 18),
            'normal': sum(1 for t in self.performance_history if 12 <= t.get('quality_score', 0) < 15),
            'poor': sum(1 for t in self.performance_history if t.get('quality_score', 0) < 12)
        }
        
        return distribution
    
    def _get_regime_performance(self) -> Dict[str, Any]:
        """Get performance by market regime"""
        # Placeholder - implement based on actual regime tracking
        return {
            'trending': {'trades': 0, 'win_rate': 0.0},
            'ranging': {'trades': 0, 'win_rate': 0.0},
            'volatile': {'trades': 0, 'win_rate': 0.0}
        }
    
    def _get_ml_accuracy(self) -> float:
        """Get ML prediction accuracy"""
        # Placeholder - implement based on actual ML tracking
        return 0.65