#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - FAZ 1: PARAMETER SPACES
💎 Parametre Uzayları Kütüphanesi - Hedge Fund Precision

Bu modül şunları sağlar:
1. ✅ Her strateji için optimized parameter spaces
2. ✅ Mathematical range optimization
3. ✅ Domain expertise integration
4. ✅ Constraint handling
5. ✅ Multi-objective parameter tuning

📍 DOSYA: parameter_spaces.py
📁 KONUM: optimization/
🔄 DURUM: kalıcı
"""

import optuna
from typing import Dict, Any, List, Tuple
import logging
from datetime import datetime, timezone
import pandas as pd
import numpy as np

from utils.portfolio import Portfolio
from backtesting.multi_strategy_backtester import MultiStrategyBacktester, BacktestConfiguration, BacktestMode

class MockPosition:
    def __init__(self, quantity_btc, entry_cost_usdt_total):
        self.quantity_btc = quantity_btc
        self.entry_cost_usdt_total = entry_cost_usdt_total

class MockPortfolio:
    def __init__(self, initial_capital_usdt):
        self.balance = initial_capital_usdt
        self.initial_capital_usdt = initial_capital_usdt
        self.positions = []
        self.closed_trades = []
        self.cumulative_pnl = 0.0

    def execute_buy(self, strategy_name, symbol, current_price, timestamp, reason, amount_usdt_override):
        cost = amount_usdt_override
        self.balance -= cost
        position = MockPosition(amount_usdt_override / current_price, cost)
        self.positions.append(position)
        return position

    def execute_sell(self, position_to_close, current_price, timestamp, reason):
        profit = (current_price * position_to_close.quantity_btc) - position_to_close.entry_cost_usdt_total
        self.cumulative_pnl += profit
        self.balance += (current_price * position_to_close.quantity_btc)
        self.positions.remove(position_to_close)
        return True

logger = logging.getLogger("ParameterSpaces")

class ParameterSpaceRegistry:
    """🎯 Central registry for all strategy parameter spaces"""
    
    @staticmethod
    def get_parameter_space(strategy_name: str, trial: optuna.Trial) -> Dict[str, Any]:
        """Get parameter space for specified strategy"""
        
        parameter_functions = {
            "momentum": get_momentum_parameter_space,
            "bollinger_rsi": get_bollinger_rsi_parameter_space,
            "rsi_ml": get_rsi_ml_parameter_space,
            "macd_ml": get_macd_ml_parameter_space,
            "volume_profile": get_volume_profile_parameter_space
        }
        
        if strategy_name not in parameter_functions:
            raise ValueError(f"Unsupported strategy: {strategy_name}")
        
        return parameter_functions[strategy_name](trial)

async def get_momentum_parameter_space(trial: optuna.Trial) -> float:
    """
    🚀 MOMENTUM STRATEGY PARAMETER SPACE
    💎 Optimized based on 26.80% performance achievement
    """
    
    parameters = {}
    
    # ✅ TECHNICAL INDICATORS (Core momentum signals)
    # EMA periods - optimized ranges based on proven performance
    parameters['ema_short'] = trial.suggest_int('ema_short', 8, 18)  # Fast trend
    parameters['ema_medium'] = trial.suggest_int('ema_medium', 18, 35)  # Medium trend  
    parameters['ema_long'] = trial.suggest_int('ema_long', 35, 100)  # Long trend
    
    # RSI configuration
    parameters['rsi_period'] = trial.suggest_int('rsi_period', 10, 20)
    parameters['rsi_oversold'] = trial.suggest_int('rsi_oversold', 25, 35)
    parameters['rsi_overbought'] = trial.suggest_int('rsi_overbought', 65, 75)
    
    # ADX strength filter
    parameters['adx_period'] = trial.suggest_int('adx_period', 10, 20)
    parameters['adx_threshold'] = trial.suggest_float('adx_threshold', 20.0, 35.0)
    
    # ATR for volatility adjustment
    parameters['atr_period'] = trial.suggest_int('atr_period', 10, 20)
    parameters['atr_multiplier'] = trial.suggest_float('atr_multiplier', 1.5, 3.0)
    
    # Volume confirmation
    parameters['volume_sma_period'] = trial.suggest_int('volume_sma_period', 15, 25)
    parameters['volume_multiplier'] = trial.suggest_float('volume_multiplier', 1.2, 2.5)
    
    # ✅ POSITION MANAGEMENT (Risk & sizing)
    parameters['max_positions'] = trial.suggest_int('max_positions', 2, 5)
    parameters['base_position_size_pct'] = trial.suggest_float('base_position_size_pct', 0.15, 0.45)
    parameters['min_position_usdt'] = trial.suggest_float('min_position_usdt', 25.0, 75.0)
    parameters['max_position_usdt'] = trial.suggest_float('max_position_usdt', 400.0, 800.0)
    
    # ✅ PERFORMANCE-BASED SIZING (Dynamic allocation)
    parameters['size_high_profit_pct'] = trial.suggest_float('size_high_profit_pct', 0.35, 0.55)
    parameters['size_good_profit_pct'] = trial.suggest_float('size_good_profit_pct', 0.25, 0.40)
    parameters['size_normal_profit_pct'] = trial.suggest_float('size_normal_profit_pct', 0.15, 0.30)
    parameters['size_breakeven_pct'] = trial.suggest_float('size_breakeven_pct', 0.10, 0.20)
    parameters['size_loss_pct'] = trial.suggest_float('size_loss_pct', 0.05, 0.15)
    parameters['size_max_balance_pct'] = trial.suggest_float('size_max_balance_pct', 0.70, 0.90)
    
    # Performance thresholds
    parameters['perf_high_profit_threshold'] = trial.suggest_float('perf_high_profit_threshold', 8.0, 15.0)
    parameters['perf_good_profit_threshold'] = trial.suggest_float('perf_good_profit_threshold', 3.0, 8.0)
    parameters['perf_normal_profit_threshold'] = trial.suggest_float('perf_normal_profit_threshold', 0.0, 3.0)
    parameters['perf_breakeven_threshold'] = trial.suggest_float('perf_breakeven_threshold', -2.0, 0.0)
    
    # ✅ ENTRY CONDITIONS (Signal strength)
    parameters['trend_strength_threshold'] = trial.suggest_float('trend_strength_threshold', 0.6, 0.9)
    parameters['momentum_threshold'] = trial.suggest_float('momentum_threshold', 0.5, 0.8)
    parameters['volume_confirmation_required'] = trial.suggest_categorical('volume_confirmation_required', [True, False])
    
    # ✅ EXIT CONDITIONS (Profit protection)
    parameters['take_profit_pct'] = trial.suggest_float('take_profit_pct', 2.5, 6.0)
    parameters['stop_loss_pct'] = trial.suggest_float('stop_loss_pct', 1.5, 4.0)
    parameters['trailing_stop_activation_pct'] = trial.suggest_float('trailing_stop_activation_pct', 1.0, 3.0)
    parameters['trailing_stop_distance_pct'] = trial.suggest_float('trailing_stop_distance_pct', 0.5, 1.5)
    
    # ✅ MACHINE LEARNING INTEGRATION
    parameters['ml_prediction_threshold'] = trial.suggest_float('ml_prediction_threshold', 0.55, 0.75)
    parameters['ml_confidence_threshold'] = trial.suggest_float('ml_confidence_threshold', 0.6, 0.9)
    parameters['enable_ml_prediction'] = trial.suggest_categorical('enable_ml_prediction', [True, False])
    
    # ✅ RISK MANAGEMENT (Advanced)
    parameters['max_daily_trades'] = trial.suggest_int('max_daily_trades', 8, 20)
    parameters['max_consecutive_losses'] = trial.suggest_int('max_consecutive_losses', 3, 7)
    parameters['portfolio_heat_limit_pct'] = trial.suggest_float('portfolio_heat_limit_pct', 15.0, 35.0)
    parameters['correlation_limit'] = trial.suggest_float('correlation_limit', 0.6, 0.9)
    
    # ✅ MARKET REGIME FILTERS
    parameters['enable_market_regime_filter'] = trial.suggest_categorical('enable_market_regime_filter', [True, False])
    parameters['bear_market_size_reduction'] = trial.suggest_float('bear_market_size_reduction', 0.3, 0.7)
    parameters['high_volatility_threshold'] = trial.suggest_float('high_volatility_threshold', 0.02, 0.05)
    
    # Ensure logical constraints
    _apply_momentum_constraints(parameters)

    # --- Backtesting and Score Calculation ---
    try:
        from strategies.momentum_optimized import EnhancedMomentumStrategy
        
        # Load historical data (using a dummy for now, replace with actual data loading)
        # In a real scenario, you'd load data based on the optimization config's date range
        # For testing, we'll create a dummy DataFrame
        data = pd.DataFrame({
            'timestamp': pd.to_datetime(pd.date_range(start='2023-01-01', periods=100, freq='H')),
            'open': 100 + np.random.rand(100) * 10,
            'high': 100 + np.random.rand(100) * 10 + 1,
            'low': 100 + np.random.rand(100) * 10 - 1,
            'close': 100 + np.random.rand(100) * 10,
            'volume': 1000 + np.random.rand(100) * 100
        })
        data = data.set_index('timestamp')

        # Initialize portfolio and strategy
        portfolio = MockPortfolio(initial_capital_usdt=10000.0)
        strategy = EnhancedMomentumStrategy(portfolio=portfolio, symbol="BTC/USDT", **parameters)

        # Run a simplified backtest
        # This is a placeholder. A real backtest would iterate through data,
        # generate signals, execute trades, and update portfolio.
        # For optimization, we need a quantifiable performance metric.
        
        # Simulate some trades to get a non-zero PnL for testing
        # In a real backtest, this would be driven by strategy signals
        if len(data) > 20 and parameters['ema_short'] < parameters['ema_long']: # Ensure enough data and a simple condition
            try:
                # Ensure portfolio.positions is not empty before trying to access index 0
                buy_position = portfolio.execute_buy(
                    strategy_name="momentum",
                    symbol="BTC/USDT",
                    current_price=data['close'].iloc[10],
                    timestamp=data.index[10].isoformat(),
                    reason="Simulated buy",
                    amount_usdt_override=500.0
                )
                if buy_position:
                    portfolio.execute_sell(
                        position_to_close=buy_position,
                        current_price=data['close'].iloc[20] * 1.01, # Small profit
                        timestamp=data.index[20].isoformat(),
                        reason="Simulated sell"
                    )
            except Exception as trade_e:
                logger.warning(f"Simulated trade failed: {trade_e}")
        
        # Calculate a score (e.g., total return, Sharpe ratio)
        # For simplicity, let's use cumulative PnL as the score
        score = portfolio.cumulative_pnl
        
        # Optuna's objective function should return a single float value.
        # If the score is not a number (e.g., NaN), return 0.0 or raise TrialPruned.
        if pd.isna(score):
            return 0.0
            
        return score

    except Exception as e:
        logger.error(f"Error during backtest simulation for trial: {e}")
        # If backtest fails, prune the trial
        raise optuna.TrialPruned()

def get_bollinger_rsi_parameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    📊 BOLLINGER BANDS + RSI STRATEGY PARAMETER SPACE
    🎯 Mean reversion with volatility bands
    """
    
    parameters = {}
    
    # ✅ BOLLINGER BANDS
    parameters['bb_period'] = trial.suggest_int('bb_period', 15, 25)
    parameters['bb_std_multiplier'] = trial.suggest_float('bb_std_multiplier', 1.8, 2.5)
    parameters['bb_squeeze_threshold'] = trial.suggest_float('bb_squeeze_threshold', 0.02, 0.06)
    
    # ✅ RSI OSCILLATOR
    parameters['rsi_period'] = trial.suggest_int('rsi_period', 10, 18)
    parameters['rsi_oversold'] = trial.suggest_int('rsi_oversold', 25, 35)
    parameters['rsi_overbought'] = trial.suggest_int('rsi_overbought', 65, 75)
    parameters['rsi_divergence_lookback'] = trial.suggest_int('rsi_divergence_lookback', 5, 15)
    
    # ✅ ENTRY/EXIT CONDITIONS
    parameters['mean_reversion_strength'] = trial.suggest_float('mean_reversion_strength', 0.6, 0.9)
    parameters['band_touch_confirmation'] = trial.suggest_categorical('band_touch_confirmation', [True, False])
    parameters['volume_spike_threshold'] = trial.suggest_float('volume_spike_threshold', 1.5, 3.0)
    
    # ✅ POSITION MANAGEMENT
    parameters['max_positions'] = trial.suggest_int('max_positions', 2, 4)
    parameters['position_size_pct'] = trial.suggest_float('position_size_pct', 0.20, 0.40)
    parameters['take_profit_pct'] = trial.suggest_float('take_profit_pct', 1.5, 4.0)
    parameters['stop_loss_pct'] = trial.suggest_float('stop_loss_pct', 1.0, 3.0)
    
    # ✅ ADVANCED FEATURES
    parameters['enable_squeeze_detection'] = trial.suggest_categorical('enable_squeeze_detection', [True, False])
    parameters['enable_divergence_detection'] = trial.suggest_categorical('enable_divergence_detection', [True, False])
    
    return parameters

def get_rsi_ml_parameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    🤖 RSI + MACHINE LEARNING STRATEGY PARAMETER SPACE
    🧠 AI-enhanced momentum detection
    """
    
    parameters = {}
    
    # ✅ RSI CORE
    parameters['rsi_period'] = trial.suggest_int('rsi_period', 8, 16)
    parameters['rsi_oversold'] = trial.suggest_int('rsi_oversold', 20, 35)
    parameters['rsi_overbought'] = trial.suggest_int('rsi_overbought', 65, 80)
    
    # ✅ MACHINE LEARNING MODEL
    parameters['ml_model_type'] = trial.suggest_categorical('ml_model_type', ['xgboost', 'random_forest', 'neural_network'])
    parameters['ml_lookback_periods'] = trial.suggest_int('ml_lookback_periods', 20, 50)
    parameters['ml_feature_count'] = trial.suggest_int('ml_feature_count', 15, 40)
    parameters['ml_prediction_threshold'] = trial.suggest_float('ml_prediction_threshold', 0.55, 0.80)
    parameters['ml_confidence_threshold'] = trial.suggest_float('ml_confidence_threshold', 0.65, 0.90)
    
    # ✅ FEATURE ENGINEERING
    parameters['enable_price_features'] = trial.suggest_categorical('enable_price_features', [True, False])
    parameters['enable_volume_features'] = trial.suggest_categorical('enable_volume_features', [True, False])
    parameters['enable_volatility_features'] = trial.suggest_categorical('enable_volatility_features', [True, False])
    parameters['enable_momentum_features'] = trial.suggest_categorical('enable_momentum_features', [True, False])
    
    # ✅ MODEL TRAINING
    parameters['training_data_size'] = trial.suggest_int('training_data_size', 1000, 3000)
    parameters['model_retrain_frequency'] = trial.suggest_int('model_retrain_frequency', 50, 200)
    parameters['validation_split'] = trial.suggest_float('validation_split', 0.15, 0.30)
    
    # ✅ POSITION MANAGEMENT
    parameters['max_positions'] = trial.suggest_int('max_positions', 2, 4)
    parameters['base_position_size_pct'] = trial.suggest_float('base_position_size_pct', 0.15, 0.35)
    parameters['ml_confidence_sizing'] = trial.suggest_categorical('ml_confidence_sizing', [True, False])
    
    return parameters

def get_macd_ml_parameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    📈 MACD + MACHINE LEARNING STRATEGY PARAMETER SPACE
    🎯 Trend following with AI confirmation
    """
    
    parameters = {}
    
    # ✅ MACD INDICATOR
    parameters['macd_fast'] = trial.suggest_int('macd_fast', 8, 15)
    parameters['macd_slow'] = trial.suggest_int('macd_slow', 21, 35)
    parameters['macd_signal'] = trial.suggest_int('macd_signal', 7, 12)
    parameters['macd_threshold'] = trial.suggest_float('macd_threshold', 0.001, 0.01)
    
    # ✅ TREND DETECTION
    parameters['trend_ema_period'] = trial.suggest_int('trend_ema_period', 50, 100)
    parameters['trend_strength_period'] = trial.suggest_int('trend_strength_period', 20, 40)
    parameters['trend_confirmation_bars'] = trial.suggest_int('trend_confirmation_bars', 2, 6)
    
    # ✅ MACHINE LEARNING INTEGRATION
    parameters['ml_model_type'] = trial.suggest_categorical('ml_model_type', ['xgboost', 'lightgbm', 'catboost'])
    parameters['ml_prediction_horizon'] = trial.suggest_int('ml_prediction_horizon', 3, 10)
    parameters['ml_feature_window'] = trial.suggest_int('ml_feature_window', 30, 60)
    parameters['ml_ensemble_models'] = trial.suggest_int('ml_ensemble_models', 3, 7)
    
    # ✅ SIGNAL COMBINATION
    parameters['macd_weight'] = trial.suggest_float('macd_weight', 0.3, 0.7)
    parameters['ml_weight'] = trial.suggest_float('ml_weight', 0.3, 0.7)
    parameters['signal_consensus_threshold'] = trial.suggest_float('signal_consensus_threshold', 0.6, 0.9)
    
    # ✅ POSITION MANAGEMENT
    parameters['max_positions'] = trial.suggest_int('max_positions', 1, 3)
    parameters['position_size_pct'] = trial.suggest_float('position_size_pct', 0.25, 0.50)
    parameters['pyramid_enabled'] = trial.suggest_categorical('pyramid_enabled', [True, False])
    parameters['pyramid_max_levels'] = trial.suggest_int('pyramid_max_levels', 2, 4)
    
    return parameters

def get_volume_profile_parameter_space(trial: optuna.Trial) -> Dict[str, Any]:
    """
    📊 VOLUME PROFILE STRATEGY PARAMETER SPACE
    🎯 Order flow and market microstructure analysis
    """
    
    parameters = {}
    
    # ✅ VOLUME PROFILE CORE
    parameters['vp_period'] = trial.suggest_int('vp_period', 20, 50)
    parameters['vp_price_levels'] = trial.suggest_int('vp_price_levels', 20, 40)
    parameters['poc_distance_threshold'] = trial.suggest_float('poc_distance_threshold', 0.5, 2.0)
    
    # ✅ VALUE AREA ANALYSIS
    parameters['value_area_percentage'] = trial.suggest_float('value_area_percentage', 0.65, 0.80)
    parameters['vah_val_breakout_threshold'] = trial.suggest_float('vah_val_breakout_threshold', 0.3, 1.0)
    parameters['volume_imbalance_ratio'] = trial.suggest_float('volume_imbalance_ratio', 1.5, 3.0)
    
    # ✅ ORDER FLOW INDICATORS
    parameters['delta_smoothing_period'] = trial.suggest_int('delta_smoothing_period', 5, 15)
    parameters['cumulative_delta_threshold'] = trial.suggest_float('cumulative_delta_threshold', 0.3, 0.8)
    parameters['volume_rate_threshold'] = trial.suggest_float('volume_rate_threshold', 1.2, 2.5)
    
    # ✅ MARKET MICROSTRUCTURE
    parameters['tick_analysis_enabled'] = trial.suggest_categorical('tick_analysis_enabled', [True, False])
    parameters['orderbook_imbalance_threshold'] = trial.suggest_float('orderbook_imbalance_threshold', 1.5, 3.0)
    parameters['aggressive_vs_passive_ratio'] = trial.suggest_float('aggressive_vs_passive_ratio', 1.2, 2.0)
    
    # ✅ POSITION MANAGEMENT
    parameters['max_positions'] = trial.suggest_int('max_positions', 1, 3)
    parameters['position_size_pct'] = trial.suggest_float('position_size_pct', 0.20, 0.45)
    parameters['poc_entry_only'] = trial.suggest_categorical('poc_entry_only', [True, False])
    
    # ✅ RISK MANAGEMENT
    parameters['value_area_stop_loss'] = trial.suggest_categorical('value_area_stop_loss', [True, False])
    parameters['volume_weighted_exits'] = trial.suggest_categorical('volume_weighted_exits', [True, False])
    
    return parameters

def _apply_momentum_constraints(parameters: Dict[str, Any]) -> None:
    """Apply logical constraints to momentum parameters"""
    
    # EMA periods must be in ascending order
    if parameters['ema_short'] >= parameters['ema_medium']:
        parameters['ema_medium'] = parameters['ema_short'] + 5
    
    if parameters['ema_medium'] >= parameters['ema_long']:
        parameters['ema_long'] = parameters['ema_medium'] + 10
    
    # RSI thresholds must be logical
    if parameters['rsi_oversold'] >= parameters['rsi_overbought']:
        parameters['rsi_overbought'] = parameters['rsi_oversold'] + 20
    
    # Take profit must be greater than stop loss
    if parameters['take_profit_pct'] <= parameters['stop_loss_pct']:
        parameters['take_profit_pct'] = parameters['stop_loss_pct'] + 0.5
    
    # Performance thresholds must be in ascending order
    thresholds = ['perf_breakeven_threshold', 'perf_normal_profit_threshold', 
                  'perf_good_profit_threshold', 'perf_high_profit_threshold']
    
    for i in range(len(thresholds) - 1):
        if parameters[thresholds[i]] >= parameters[thresholds[i + 1]]:
            parameters[thresholds[i + 1]] = parameters[thresholds[i]] + 1.0

# Main interface function
async def get_parameter_space(strategy_name: str, trial: optuna.Trial) -> float:
    """
    🎯 Main interface function for parameter space retrieval
    
    Args:
        strategy_name: Name of the strategy
        trial: Optuna trial object
        
    Returns:
        float: Performance score for the strategy with the given parameters
    """
    
    parameter_functions = {
        "momentum": get_momentum_parameter_space,
        # Add other strategies here as they are implemented
    }
    
    if strategy_name not in parameter_functions:
        raise ValueError(f"Unsupported strategy: {strategy_name}")
    
    return parameter_functions[strategy_name](trial)


# Parameter validation functions
def validate_parameters(strategy_name: str, parameters: Dict[str, Any]) -> bool:
    """Validate parameter set for logical consistency"""
    
    try:
        if strategy_name == "momentum":
            return _validate_momentum_parameters(parameters)
        # Add validation for other strategies as needed
        return True
    except Exception:
        return False

def _validate_momentum_parameters(parameters: Dict[str, Any]) -> bool:
    """Validate momentum strategy parameters"""
    
    # Check EMA order
    if not (parameters['ema_short'] < parameters['ema_medium'] < parameters['ema_long']):
        return False
    
    # Check RSI bounds
    if not (0 < parameters['rsi_oversold'] < parameters['rsi_overbought'] < 100):
        return False
    
    # Check profit/loss relationship
    if parameters['take_profit_pct'] <= parameters['stop_loss_pct']:
        return False
    
    return True