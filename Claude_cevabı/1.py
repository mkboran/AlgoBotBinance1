#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - FAZ 1: PARAMETER SPACES - HEDGE FUND+ LEVEL
💎 FIXED: Async/Sync Uyumsuzluğu Giderildi

ÇÖZÜMLER:
1. ✅ get_parameter_space fonksiyonu artık senkron
2. ✅ MockPortfolio ile senkron test ortamı
3. ✅ Robust backtest simülasyonu
4. ✅ Hata toleranslı veri işleme
"""

import optuna
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np

logger = logging.getLogger("ParameterSpaces")

class MockPosition:
    """Test için mock pozisyon"""
    def __init__(self, quantity_btc, entry_cost_usdt_total):
        self.quantity_btc = quantity_btc
        self.entry_cost_usdt_total = entry_cost_usdt_total

class MockPortfolio:
    """Senkron test portfolio - Optuna uyumlu"""
    def __init__(self, initial_capital_usdt):
        self.balance = initial_capital_usdt
        self.initial_capital_usdt = initial_capital_usdt
        self.positions = []
        self.closed_trades = []
        self.cumulative_pnl = 0.0

    def execute_buy(self, strategy_name, symbol, current_price, timestamp, reason, amount_usdt_override):
        """Senkron alım işlemi"""
        if self.balance < amount_usdt_override:
            return None
        
        cost = amount_usdt_override
        self.balance -= cost
        position = MockPosition(amount_usdt_override / current_price, cost)
        self.positions.append(position)
        return position

    def execute_sell(self, position_to_close, current_price, timestamp, reason):
        """Senkron satış işlemi"""
        if position_to_close not in self.positions:
            return False
            
        profit = (current_price * position_to_close.quantity_btc) - position_to_close.entry_cost_usdt_total
        self.cumulative_pnl += profit
        self.balance += (current_price * position_to_close.quantity_btc)
        self.positions.remove(position_to_close)
        self.closed_trades.append({
            'profit': profit,
            'timestamp': timestamp
        })
        return True

class ParameterSpaceRegistry:
    """🎯 Central registry for all strategy parameter spaces"""
    
    @staticmethod
    def get_parameter_space(strategy_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        """Get parameter space for specified strategy - SYNCHRONOUS"""
        
        parameter_functions = {
            "momentum": get_momentum_parameter_space,
            "bollinger_rsi": get_bollinger_rsi_parameter_space,
            "rsi_ml": get_rsi_ml_parameter_space,
            "macd_ml": get_macd_ml_parameter_space,
        }
        
        if strategy_name not in parameter_functions:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        # Call the synchronous function directly
        return parameter_functions[strategy_name](trial)

def get_parameter_space(strategy_name: str, trial: optuna.Trial) -> Dict[str, Any]:
    """Main entry point - SYNCHRONOUS"""
    return ParameterSpaceRegistry.get_parameter_space(strategy_name, trial)

def get_momentum_parameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    🚀 MOMENTUM STRATEGY PARAMETER SPACE - FULLY SYNCHRONOUS
    💎 Hedge Fund Level Parameter Optimization
    """
    
    parameters = {}
    
    # ✅ EMA PARAMETERS - Optimized ranges
    parameters['ema_short'] = trial.suggest_int('ema_short', 12, 16)
    parameters['ema_medium'] = trial.suggest_int('ema_medium', 20, 24) 
    parameters['ema_long'] = trial.suggest_int('ema_long', 55, 60)
    
    # ✅ MOMENTUM INDICATORS
    parameters['rsi_period'] = trial.suggest_int('rsi_period', 12, 16)
    parameters['rsi_oversold'] = trial.suggest_int('rsi_oversold', 25, 35)
    parameters['rsi_overbought'] = trial.suggest_int('rsi_overbought', 65, 75)
    
    # ✅ ADX TREND STRENGTH
    parameters['adx_period'] = trial.suggest_int('adx_period', 12, 16)
    parameters['adx_threshold'] = trial.suggest_int('adx_threshold', 22, 28)
    
    # ✅ ATR VOLATILITY
    parameters['atr_period'] = trial.suggest_int('atr_period', 12, 16)
    parameters['atr_multiplier'] = trial.suggest_float('atr_multiplier', 1.8, 2.5)
    
    # ✅ VOLUME ANALYSIS
    parameters['volume_sma_period'] = trial.suggest_int('volume_sma_period', 18, 22)
    parameters['volume_multiplier'] = trial.suggest_float('volume_multiplier', 1.4, 1.8)
    
    # ✅ MOMENTUM SCORING
    parameters['momentum_lookback'] = trial.suggest_int('momentum_lookback', 3, 6)
    parameters['momentum_threshold'] = trial.suggest_float('momentum_threshold', 0.008, 0.015)
    
    # ✅ QUALITY SCORE WEIGHTS
    parameters['quality_trend_weight'] = trial.suggest_float('quality_trend_weight', 0.25, 0.35)
    parameters['quality_volume_weight'] = trial.suggest_float('quality_volume_weight', 0.15, 0.25)
    parameters['quality_volatility_weight'] = trial.suggest_float('quality_volatility_weight', 0.15, 0.25)
    parameters['quality_momentum_weight'] = trial.suggest_float('quality_momentum_weight', 0.25, 0.35)
    
    # ✅ SIGNAL FILTERING
    parameters['min_quality_score'] = trial.suggest_int('min_quality_score', 12, 16)
    parameters['trend_alignment_required'] = trial.suggest_categorical('trend_alignment_required', [True, False])
    
    # ✅ MACHINE LEARNING
    parameters['ml_enabled'] = trial.suggest_categorical('ml_enabled', [True, False])
    parameters['ml_confidence_threshold'] = trial.suggest_float('ml_confidence_threshold', 0.6, 0.75)
    
    # ✅ KELLY CRITERION
    parameters['kelly_enabled'] = trial.suggest_categorical('kelly_enabled', [True, False])
    parameters['kelly_multiplier'] = trial.suggest_float('kelly_multiplier', 0.2, 0.35)
    
    # ✅ DYNAMIC EXITS
    parameters['dynamic_exit_enabled'] = trial.suggest_categorical('dynamic_exit_enabled', [True, False])
    parameters['trailing_stop_activation_pct'] = trial.suggest_float('trailing_stop_activation_pct', 0.015, 0.025)
    parameters['trailing_stop_distance_pct'] = trial.suggest_float('trailing_stop_distance_pct', 0.008, 0.015)
    
    # ✅ RISK MANAGEMENT
    parameters['max_positions'] = trial.suggest_int('max_positions', 2, 4)
    parameters['position_size_pct'] = trial.suggest_float('position_size_pct', 0.2, 0.35)
    parameters['max_drawdown_pct'] = trial.suggest_float('max_drawdown_pct', 0.08, 0.12)
    
    # ✅ ADAPTIVE PARAMETERS
    parameters['adaptive_enabled'] = trial.suggest_categorical('adaptive_enabled', [True, False])
    parameters['learning_rate'] = trial.suggest_float('learning_rate', 0.05, 0.15)
    
    # 🚀 SIMULATED BACKTEST FOR SCORING
    try:
        # Create synthetic data for testing
        data_length = 1000
        base_price = 50000
        
        # Generate realistic price data with trend and volatility
        trend = np.linspace(0, 0.2, data_length)
        noise = np.random.normal(0, 0.02, data_length)
        prices = base_price * (1 + trend + noise)
        
        # Create DataFrame
        timestamps = pd.date_range(
            start=datetime.now(timezone.utc) - pd.Timedelta(days=50),
            periods=data_length,
            freq='15min'
        )
        
        df = pd.DataFrame({
            'timestamp': timestamps,
            'open': prices * 0.998,
            'high': prices * 1.002,
            'low': prices * 0.996,
            'close': prices,
            'volume': np.random.uniform(100, 1000, data_length)
        })
        
        # Run simulated backtest
        portfolio = MockPortfolio(initial_capital_usdt=10000.0)
        
        # Calculate indicators
        df['ema_short'] = df['close'].ewm(span=parameters['ema_short'], adjust=False).mean()
        df['ema_medium'] = df['close'].ewm(span=parameters['ema_medium'], adjust=False).mean()
        df['ema_long'] = df['close'].ewm(span=parameters['ema_long'], adjust=False).mean()
        
        # RSI calculation
        delta = df['close'].diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=parameters['rsi_period']).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=parameters['rsi_period']).mean()
        rs = gain / (loss + 1e-10)
        df['rsi'] = 100 - (100 / (1 + rs))
        
        # Simple trading logic
        position = None
        entry_bar = 0
        
        for i in range(100, len(df)):
            current_price = df['close'].iloc[i]
            
            # Entry conditions
            if position is None and i - entry_bar > 10:  # Cooldown period
                ema_bullish = (df['ema_short'].iloc[i] > df['ema_medium'].iloc[i] > df['ema_long'].iloc[i])
                rsi_oversold = df['rsi'].iloc[i] < parameters['rsi_oversold']
                
                if ema_bullish and rsi_oversold:
                    # Buy signal
                    position = portfolio.execute_buy(
                        strategy_name="momentum",
                        symbol="BTC/USDT",
                        current_price=current_price,
                        timestamp=df['timestamp'].iloc[i],
                        reason="Momentum buy signal",
                        amount_usdt_override=portfolio.balance * parameters['position_size_pct']
                    )
                    entry_bar = i
            
            # Exit conditions
            elif position is not None:
                # Calculate profit
                current_value = current_price * position.quantity_btc
                profit_pct = (current_value - position.entry_cost_usdt_total) / position.entry_cost_usdt_total
                
                # Exit on profit target or stop loss
                if profit_pct > 0.02 or profit_pct < -0.01:
                    portfolio.execute_sell(
                        position_to_close=position,
                        current_price=current_price,
                        timestamp=df['timestamp'].iloc[i],
                        reason=f"Exit at {profit_pct:.2%}"
                    )
                    position = None
        
        # Calculate final metrics
        total_return = (portfolio.balance + portfolio.cumulative_pnl - 10000) / 10000
        win_rate = len([t for t in portfolio.closed_trades if t['profit'] > 0]) / max(1, len(portfolio.closed_trades))
        
        # Composite score
        score = (total_return * 0.4) + (win_rate * 0.3) + (0.3 / max(0.01, abs(total_return - 0.1)))
        
        return score
        
    except Exception as e:
        logger.error(f"Backtest simulation error: {e}")
        return 0.0  # Return worst score on error

def get_bollinger_rsi_parameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    📊 BOLLINGER BANDS + RSI STRATEGY PARAMETER SPACE
    🎯 Mean reversion with volatility bands
    """
    
    parameters = {}
    
    # ✅ BOLLINGER BANDS
    parameters['bb_period'] = trial.suggest_int('bb_period', 18, 22)
    parameters['bb_std_dev'] = trial.suggest_float('bb_std_dev', 1.8, 2.2)
    
    # ✅ RSI
    parameters['rsi_period'] = trial.suggest_int('rsi_period', 12, 16)
    parameters['rsi_oversold'] = trial.suggest_int('rsi_oversold', 25, 35)
    parameters['rsi_overbought'] = trial.suggest_int('rsi_overbought', 65, 75)
    
    # ✅ VOLUME FILTER
    parameters['volume_ma_period'] = trial.suggest_int('volume_ma_period', 18, 22)
    parameters['volume_threshold'] = trial.suggest_float('volume_threshold', 1.3, 1.7)
    
    # ✅ POSITION MANAGEMENT
    parameters['position_size_pct'] = trial.suggest_float('position_size_pct', 0.2, 0.35)
    parameters['max_positions'] = trial.suggest_int('max_positions', 2, 4)
    
    return parameters

def get_rsi_ml_parameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    🤖 RSI + MACHINE LEARNING PARAMETER SPACE
    🧠 AI-enhanced momentum detection
    """
    
    parameters = {}
    
    # ✅ RSI CORE
    parameters['rsi_period'] = trial.suggest_int('rsi_period', 8, 16)
    parameters['rsi_oversold'] = trial.suggest_int('rsi_oversold', 20, 35)
    parameters['rsi_overbought'] = trial.suggest_int('rsi_overbought', 65, 80)
    
    # ✅ MACHINE LEARNING
    parameters['ml_lookback_periods'] = trial.suggest_int('ml_lookback_periods', 20, 50)
    parameters['ml_confidence_threshold'] = trial.suggest_float('ml_confidence_threshold', 0.6, 0.8)
    
    # ✅ POSITION MANAGEMENT
    parameters['base_position_size_pct'] = trial.suggest_float('base_position_size_pct', 0.15, 0.35)
    
    return parameters

def get_macd_ml_parameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    📈 MACD + MACHINE LEARNING PARAMETER SPACE
    🎯 Trend following with AI confirmation
    """
    
    parameters = {}
    
    # ✅ MACD
    parameters['macd_fast'] = trial.suggest_int('macd_fast', 10, 14)
    parameters['macd_slow'] = trial.suggest_int('macd_slow', 24, 28)
    parameters['macd_signal'] = trial.suggest_int('macd_signal', 8, 10)
    
    # ✅ ML PARAMETERS
    parameters['ml_feature_count'] = trial.suggest_int('ml_feature_count', 20, 40)
    parameters['ml_prediction_threshold'] = trial.suggest_float('ml_prediction_threshold', 0.55, 0.75)
    
    return parameters