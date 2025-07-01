#!/usr/bin/env python3
"""
🎯 OBJECTIVE FIXED - OPTIMIZATION OBJECTIVE FUNCTION
💎 Optimizasyon için skor hesaplama fonksiyonu

Bu modül optimizasyon algoritmaları için objective function sağlar.
"""

import asyncio
import pandas as pd
import numpy as np
from typing import Dict, Any, Optional
from datetime import datetime, timezone, timedelta
from pathlib import Path
import logging

# Core imports with error handling
try:
    from utils.config import settings
    from utils.logger import logger
    from utils.portfolio import Portfolio
except ImportError as e:
    import logging
    logger = logging.getLogger(__name__)
    
    class DummySettings:
        SYMBOL = "BTC/USDT"
        INITIAL_CAPITAL_USDT = 1000.0
    
    settings = DummySettings()

def run_objective(trial, strategy_name: str = "momentum", **kwargs) -> float:
    """
    🎯 Optimization objective function
    
    Args:
        trial: Optuna trial object
        strategy_name: Strategy to optimize
        **kwargs: Additional parameters
        
    Returns:
        float: Objective score (higher is better)
    """
    
    try:
        # Suggest parameters based on strategy
        params = suggest_parameters(trial, strategy_name)
        
        # Run backtest with suggested parameters
        score = run_backtest_with_params(params, strategy_name)
        
        return score
        
    except Exception as e:
        logger.error(f"Objective function error: {e}")
        return -1000.0  # Penalty for failed trials

def suggest_parameters(trial, strategy_name: str) -> Dict[str, Any]:
    """
    🔧 Suggest parameters for optimization
    
    Args:
        trial: Optuna trial object
        strategy_name: Strategy name
        
    Returns:
        Dict[str, Any]: Suggested parameters
    """
    
    if strategy_name == "momentum":
        return suggest_momentum_parameters(trial)
    elif strategy_name == "bollinger":
        return suggest_bollinger_parameters(trial)
    elif strategy_name == "rsi":
        return suggest_rsi_parameters(trial)
    else:
        return suggest_default_parameters(trial)

def suggest_momentum_parameters(trial) -> Dict[str, Any]:
    """Momentum strategy parameters"""
    
    return {
        # Technical indicators
        'ema_short': trial.suggest_int('ema_short', 8, 15),
        'ema_medium': trial.suggest_int('ema_medium', 18, 25),
        'ema_long': trial.suggest_int('ema_long', 45, 65),
        'rsi_period': trial.suggest_int('rsi_period', 10, 16),
        'adx_period': trial.suggest_int('adx_period', 20, 30),
        
        # Position sizing
        'base_position_size_pct': trial.suggest_float('base_position_size_pct', 50.0, 80.0),
        'max_positions': trial.suggest_int('max_positions', 1, 4),
        
        # Risk management
        'max_loss_pct': trial.suggest_float('max_loss_pct', 2.0, 6.0),
        'min_profit_target_usdt': trial.suggest_float('min_profit_target_usdt', 15.0, 40.0),
        
        # ML settings
        'ml_enabled': trial.suggest_categorical('ml_enabled', [True, False]),
        'ml_confidence_threshold': trial.suggest_float('ml_confidence_threshold', 0.55, 0.75)
    }

def suggest_bollinger_parameters(trial) -> Dict[str, Any]:
    """Bollinger Bands strategy parameters"""
    
    return {
        'bb_period': trial.suggest_int('bb_period', 15, 25),
        'bb_std': trial.suggest_float('bb_std', 1.8, 2.2),
        'squeeze_threshold': trial.suggest_float('squeeze_threshold', 0.05, 0.15),
        'base_position_size_pct': trial.suggest_float('base_position_size_pct', 40.0, 70.0)
    }

def suggest_rsi_parameters(trial) -> Dict[str, Any]:
    """RSI strategy parameters"""
    
    return {
        'rsi_period': trial.suggest_int('rsi_period', 10, 18),
        'rsi_overbought': trial.suggest_float('rsi_overbought', 70.0, 85.0),
        'rsi_oversold': trial.suggest_float('rsi_oversold', 15.0, 30.0),
        'base_position_size_pct': trial.suggest_float('base_position_size_pct', 45.0, 75.0)
    }

def suggest_default_parameters(trial) -> Dict[str, Any]:
    """Default parameters for unknown strategies"""
    
    return {
        'base_position_size_pct': trial.suggest_float('base_position_size_pct', 50.0, 70.0),
        'max_positions': trial.suggest_int('max_positions', 1, 3),
        'ml_enabled': trial.suggest_categorical('ml_enabled', [True, False])
    }

def run_backtest_with_params(params: Dict[str, Any], strategy_name: str) -> float:
    """
    🧪 Run backtest with given parameters
    
    Args:
        params: Strategy parameters
        strategy_name: Strategy name
        
    Returns:
        float: Backtest score
    """
    
    try:
        # Mock backtest for now - replace with real implementation
        base_score = 100.0
        
        # Penalize extreme values
        if params.get('base_position_size_pct', 50) > 75:
            base_score -= 20.0
        
        if params.get('max_positions', 2) > 3:
            base_score -= 15.0
        
        # Reward ML usage
        if params.get('ml_enabled', False):
            base_score += 10.0
        
        # Add some randomness to simulate real backtest variance
        import random
        random_factor = random.uniform(0.8, 1.2)
        
        final_score = base_score * random_factor
        
        logger.debug(f"Backtest score for {strategy_name}: {final_score:.2f}")
        return final_score
        
    except Exception as e:
        logger.error(f"Backtest error: {e}")
        return -100.0

def calculate_composite_score(metrics: Dict[str, float]) -> float:
    """
    📊 Calculate composite score from backtest metrics
    
    Args:
        metrics: Backtest metrics dictionary
        
    Returns:
        float: Composite score
    """
    
    try:
        # Extract metrics with defaults
        total_return = metrics.get('total_return_pct', 0.0)
        sharpe_ratio = metrics.get('sharpe_ratio', 0.0)
        max_drawdown = metrics.get('max_drawdown_pct', 100.0)
        win_rate = metrics.get('win_rate_pct', 0.0)
        profit_factor = metrics.get('profit_factor', 0.0)
        
        # Composite score calculation
        return_score = min(total_return / 100.0, 2.0) * 40  # Max 80 points
        sharpe_score = min(sharpe_ratio / 2.0, 1.0) * 25   # Max 25 points
        dd_score = max(0, (20 - max_drawdown) / 20) * 20   # Max 20 points
        win_score = (win_rate / 100.0) * 10                # Max 10 points
        pf_score = min(profit_factor / 3.0, 1.0) * 5       # Max 5 points
        
        composite = return_score + sharpe_score + dd_score + win_score + pf_score
        
        return max(0.0, composite)
        
    except Exception as e:
        logger.error(f"Composite score calculation error: {e}")
        return 0.0

# Async wrapper for compatibility
async def run_objective_async(trial, strategy_name: str = "momentum", **kwargs) -> float:
    """Async version of run_objective"""
    return run_objective(trial, strategy_name, **kwargs)

# Export main functions
__all__ = ['run_objective', 'run_objective_async', 'suggest_parameters', 'calculate_composite_score']
