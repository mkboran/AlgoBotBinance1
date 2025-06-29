#!/usr/bin/env python3
"""
🔥 HIZLI DÜZELTME: OBJECTIVE FUNCTION - SYNC VERSION
💎 Asyncio loop çakışmasını önlemek için sync versiyonu
"""

import sys
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional
import optuna
from datetime import datetime, timezone
import traceback

# Add project root to path
sys.path.append(str(Path(__file__).parent.parent))

# Import project modules
from utils.portfolio import Portfolio
from utils.config import settings
from utils.logger import logger
from parameter_spaces import get_parameter_space

# Strategy imports
try:
    from strategies.momentum_optimized import EnhancedMomentumStrategy
    MOMENTUM_AVAILABLE = True
except ImportError:
    MOMENTUM_AVAILABLE = False

# Backtester import
try:
    from backtesting.backtest_runner import MomentumBacktester
    BACKTESTER_AVAILABLE = True
except ImportError:
    BACKTESTER_AVAILABLE = False

objective_logger = logging.getLogger("ObjectiveFunction")

def calculate_composite_score(results: Dict[str, Any]) -> float:
    """🎯 Risk-adjusted composite score calculation"""
    
    # Extract metrics with safe defaults
    sharpe_ratio = results.get('sharpe_ratio', 0.0)
    calmar_ratio = results.get('calmar_ratio', 0.0)
    total_return_pct = results.get('total_return_pct', 0.0)
    win_rate_pct = results.get('win_rate_pct', 0.0)
    max_drawdown_pct = results.get('max_drawdown_pct', 100.0)
    profit_factor = results.get('profit_factor', 0.0)
    total_trades = results.get('total_trades', 0)
    
    # Normalize metrics
    normalized_sharpe = min(1.0, max(0.0, (sharpe_ratio + 2) / 7))
    normalized_calmar = min(1.0, max(0.0, (calmar_ratio + 1) / 4))
    normalized_return = min(1.0, max(0.0, total_return_pct / 50))
    normalized_winrate = min(1.0, max(0.0, win_rate_pct / 100))
    normalized_profit_factor = min(1.0, max(0.0, (profit_factor - 0.5) / 4.5))
    
    # Base composite score
    composite_score = (
        normalized_sharpe * 0.35 +
        normalized_calmar * 0.25 +
        normalized_return * 0.20 +
        normalized_winrate * 0.10 +
        normalized_profit_factor * 0.10
    )
    
    # Apply penalties
    penalties = 0.0
    
    if max_drawdown_pct > 15.0:
        penalties += (max_drawdown_pct - 15.0) * 0.02
    
    if total_trades < 10:
        penalties += 0.1 * (10 - total_trades) / 10
    
    if total_return_pct < 0:
        penalties += abs(total_return_pct) * 0.01
    
    if win_rate_pct < 40.0:
        penalties += (40.0 - win_rate_pct) * 0.005
    
    final_score = max(0.0, composite_score - penalties)
    return final_score

def run_objective(
    trial: optuna.Trial,
    strategy_name: str,
    data_file: str,
    start_date: str,
    end_date: str
) -> float:
    """🎯 MAIN OBJECTIVE FUNCTION - SYNC VERSION"""
    
    try:
        objective_logger.debug(f"Running objective for {strategy_name}, trial {trial.number}")
        
        # Get parameter space for strategy
        parameters = get_parameter_space(strategy_name, trial)
        
        # Create portfolio
        portfolio = Portfolio(initial_balance=1000.0)
        
        # Create strategy instance with parameters
        if strategy_name == "momentum":
            if not MOMENTUM_AVAILABLE:
                raise ValueError("Momentum strategy not available")
            strategy = EnhancedMomentumStrategy(portfolio=portfolio, **parameters)
        else:
            raise ValueError(f"Strategy {strategy_name} not supported in sync mode")
        
        # Create backtester
        if not BACKTESTER_AVAILABLE:
            raise ValueError("Backtester not available")
        
        backtester = MomentumBacktester(
            csv_path=data_file,
            initial_capital=1000.0,
            start_date=start_date,
            end_date=end_date,
            symbol="BTC/USDT",
            portfolio_instance=portfolio,
            strategy_instance=strategy
        )
        
        # Run backtest - try sync first, then async if needed
        import inspect
        import asyncio
        
        if inspect.iscoroutinefunction(backtester.run_backtest):
            # Run async in sync context - CAREFUL!
            try:
                # Check if there's already a running loop
                current_loop = asyncio.get_running_loop()
                # If we get here, there's a running loop, so we can't create a new one
                # This is the problematic case - return failure
                objective_logger.error("Cannot run async backtest in running event loop context")
                return -1.0
            except RuntimeError:
                # No running loop, safe to create new one
                results = asyncio.run(backtester.run_backtest())
        else:
            # Sync method, safe to call
            results = backtester.run_backtest()
        
        if not results or 'error_in_backtest' in results:
            objective_logger.warning(f"Trial {trial.number} backtest failed")
            return -1.0
        
        # Calculate composite score
        composite_score = calculate_composite_score(results)
        
        # Log results
        return_pct = results.get('total_return_pct', 0)
        sharpe = results.get('sharpe_ratio', 0)
        drawdown = results.get('max_drawdown_pct', 0)
        
        objective_logger.info(
            f"Trial {trial.number}: Score={composite_score:.4f}, "
            f"Return={return_pct:.1f}%, Sharpe={sharpe:.2f}, DD={drawdown:.1f}%"
        )
        
        # Prune bad trials
        if composite_score < 0.1 and trial.number > 10:
            raise optuna.TrialPruned()
        
        return composite_score
        
    except optuna.TrialPruned:
        raise
    except Exception as e:
        objective_logger.error(f"Trial {trial.number} error: {e}")
        return -1.0