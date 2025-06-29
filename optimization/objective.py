#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - FAZ 1: OBJECTIVE FUNCTION
💎 Evrensel Başarı Metriği - Mathematical Precision

Bu modül şunları sağlar:
1. ✅ Risk-adjusted composite scoring
2. ✅ Multi-metric evaluation system
3. ✅ Robust performance assessment
4. ✅ Overfitting prevention
5. ✅ Production-ready validation

📍 DOSYA: objective.py
📁 KONUM: optimization/
🔄 DURUM: kalıcı
"""

import sys
import asyncio
import logging
import numpy as np
from pathlib import Path
from typing import Dict, Any, Optional, Tuple
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

# Setup logger
objective_logger = logging.getLogger("ObjectiveFunction")

class ObjectiveMetrics:
    """📊 Comprehensive performance metrics calculator"""
    
    @staticmethod
    def calculate_composite_score(results: Dict[str, Any]) -> float:
        """
        🎯 COMPOSITE SCORE CALCULATION
        💎 Risk-adjusted performance metric combining multiple factors
        
        Formula:
        Score = (Sharpe * 0.4) + (Calmar * 0.3) + (Return * 0.2) + (WinRate * 0.1) - Penalties
        """
        
        # Extract metrics with safe defaults
        sharpe_ratio = results.get('sharpe_ratio', 0.0)
        calmar_ratio = results.get('calmar_ratio', 0.0)
        total_return_pct = results.get('total_return_pct', 0.0)
        win_rate_pct = results.get('win_rate_pct', 0.0)
        max_drawdown_pct = results.get('max_drawdown_pct', 100.0)
        profit_factor = results.get('profit_factor', 0.0)
        total_trades = results.get('total_trades', 0)
        
        # Normalize metrics to 0-1 scale for fair combination
        
        # Sharpe ratio (typical range: -2 to 5, target: >2)
        normalized_sharpe = min(1.0, max(0.0, (sharpe_ratio + 2) / 7))
        
        # Calmar ratio (typical range: -1 to 3, target: >1)
        normalized_calmar = min(1.0, max(0.0, (calmar_ratio + 1) / 4))
        
        # Total return (scale: 0-100%, target: >20%)
        normalized_return = min(1.0, max(0.0, total_return_pct / 50))
        
        # Win rate (scale: 0-100%, target: >60%)
        normalized_winrate = min(1.0, max(0.0, win_rate_pct / 100))
        
        # Profit factor (typical range: 0-5, target: >1.5)
        normalized_profit_factor = min(1.0, max(0.0, (profit_factor - 0.5) / 4.5))
        
        # Base composite score
        composite_score = (
            normalized_sharpe * 0.35 +           # Risk-adjusted return
            normalized_calmar * 0.25 +          # Drawdown-adjusted return
            normalized_return * 0.20 +          # Absolute return
            normalized_winrate * 0.10 +         # Consistency
            normalized_profit_factor * 0.10     # Risk/reward efficiency
        )
        
        # Apply penalties for poor risk characteristics
        penalties = 0.0
        
        # Drawdown penalty (exponential penalty for high drawdowns)
        if max_drawdown_pct > 15.0:
            penalties += (max_drawdown_pct - 15.0) * 0.02
        
        # Low trade count penalty (insufficient data)
        if total_trades < 10:
            penalties += 0.1 * (10 - total_trades) / 10
        
        # Negative return penalty
        if total_return_pct < 0:
            penalties += abs(total_return_pct) * 0.01
        
        # Very low win rate penalty
        if win_rate_pct < 40.0:
            penalties += (40.0 - win_rate_pct) * 0.005
        
        # Final score with penalties
        final_score = max(0.0, composite_score - penalties)
        
        objective_logger.debug(f"Composite Score Calculation:")
        objective_logger.debug(f"  Sharpe: {sharpe_ratio:.3f} -> {normalized_sharpe:.3f}")
        objective_logger.debug(f"  Calmar: {calmar_ratio:.3f} -> {normalized_calmar:.3f}")
        objective_logger.debug(f"  Return: {total_return_pct:.1f}% -> {normalized_return:.3f}")
        objective_logger.debug(f"  WinRate: {win_rate_pct:.1f}% -> {normalized_winrate:.3f}")
        objective_logger.debug(f"  Base Score: {composite_score:.4f}")
        objective_logger.debug(f"  Penalties: {penalties:.4f}")
        objective_logger.debug(f"  Final Score: {final_score:.4f}")
        
        return final_score
    
    @staticmethod
    def calculate_robustness_score(results: Dict[str, Any]) -> float:
        """🛡️ Calculate strategy robustness based on consistency metrics"""
        
        # Robustness factors
        drawdown = results.get('max_drawdown_pct', 100.0)
        volatility = results.get('volatility_annual', 50.0)
        win_rate = results.get('win_rate_pct', 0.0)
        profit_factor = results.get('profit_factor', 0.0)
        
        # Robustness = inverse of risk metrics
        drawdown_factor = max(0.0, 1.0 - (drawdown / 20.0))  # Lower drawdown = higher robustness
        volatility_factor = max(0.0, 1.0 - (volatility / 40.0))  # Lower volatility = higher robustness
        consistency_factor = win_rate / 100.0  # Higher win rate = higher robustness
        stability_factor = min(1.0, profit_factor / 2.0)  # Profit factor around 2 = optimal
        
        robustness = (drawdown_factor * 0.4 + volatility_factor * 0.2 + 
                     consistency_factor * 0.2 + stability_factor * 0.2)
        
        return max(0.0, min(1.0, robustness))

class StrategyExecutor:
    """🎯 Strategy execution and backtesting coordinator"""
    
    def __init__(self):
        self.data_cache = {}  # Cache loaded data for performance
    
    def execute_strategy_backtest_sync(
        self, 
        strategy_name: str, 
        parameters: Dict[str, Any],
        data_file: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Execute strategy backtest synchronously"""
        
        try:
            objective_logger.debug(f"Executing {strategy_name} backtest: {start_date} to {end_date}")
            
            # Validate strategy support
            if strategy_name == "momentum" and not MOMENTUM_AVAILABLE:
                raise ValueError("Momentum strategy not available")
            
            if not BACKTESTER_AVAILABLE:
                raise ValueError("Backtester not available")
            
            # Create portfolio with standard initial capital
            portfolio = Portfolio(initial_balance=1000.0)
            
            # Create strategy instance with parameters
            if strategy_name == "momentum":
                if not MOMENTUM_AVAILABLE:
                    raise ValueError("Enhanced Momentum Strategy not available")
                
                # Apply parameters to strategy
                strategy = EnhancedMomentumStrategy(portfolio=portfolio, **parameters)
            else:
                raise ValueError(f"Strategy {strategy_name} not implemented for sync execution")
            
            # Create backtester  
            backtester = MomentumBacktester(
                csv_path=data_file,
                initial_capital=1000.0,
                start_date=start_date,
                end_date=end_date,
                symbol="BTC/USDT",
                portfolio_instance=portfolio,
                strategy_instance=strategy
            )
            
            # Check if run_backtest is async
            # For now, let's try to run it sync - if it fails, we'll handle
            import inspect
            if inspect.iscoroutinefunction(backtester.run_backtest):
                # It's async, we need to handle this differently
                raise ValueError("Backtester is async, cannot run synchronously")
            else:
                # It's sync, safe to call
                results = backtester.run_backtest()
            
            if not results or 'error_in_backtest' in results:
                error_msg = results.get('error_in_backtest', 'Unknown backtest error') if results else 'No results returned'
                objective_logger.error(f"Backtest failed: {error_msg}")
                return self._create_failure_result(error_msg)
            
            # Validate results
            if not self._validate_backtest_results(results):
                objective_logger.warning("Backtest results validation failed")
                return self._create_failure_result("Invalid backtest results")
            
            objective_logger.debug(f"Backtest completed successfully: Return={results.get('total_return_pct', 0):.2f}%")
            
            return results
            
        except Exception as e:
            objective_logger.error(f"Strategy execution failed: {e}")
            objective_logger.error(traceback.format_exc())
            return self._create_failure_result(str(e))
    
    async def execute_strategy_backtest(
        self, 
        strategy_name: str, 
        parameters: Dict[str, Any],
        data_file: str,
        start_date: str,
        end_date: str
    ) -> Dict[str, Any]:
        """Execute strategy backtest with given parameters"""
        
        try:
            objective_logger.debug(f"Executing {strategy_name} backtest: {start_date} to {end_date}")
            
            # Validate strategy support
            if strategy_name == "momentum" and not MOMENTUM_AVAILABLE:
                raise ValueError("Momentum strategy not available")
            
            if not BACKTESTER_AVAILABLE:
                raise ValueError("Backtester not available")
            
            # Create portfolio with standard initial capital
            portfolio = Portfolio(initial_balance=1000.0)
            
            # Create strategy instance with parameters
            strategy = await self._create_strategy_instance(strategy_name, portfolio, parameters)
            
            # Create backtester
            backtester = MomentumBacktester(
                csv_path=data_file,
                initial_capital=1000.0,
                start_date=start_date,
                end_date=end_date,
                symbol="BTC/USDT",
                portfolio_instance=portfolio,
                strategy_instance=strategy
            )
            
            # Run backtest
            results = await backtester.run_backtest()
            
            if not results or 'error_in_backtest' in results:
                error_msg = results.get('error_in_backtest', 'Unknown backtest error') if results else 'No results returned'
                objective_logger.error(f"Backtest failed: {error_msg}")
                return self._create_failure_result(error_msg)
            
            # Validate results
            if not self._validate_backtest_results(results):
                objective_logger.warning("Backtest results validation failed")
                return self._create_failure_result("Invalid backtest results")
            
            objective_logger.debug(f"Backtest completed successfully: Return={results.get('total_return_pct', 0):.2f}%")
            
            return results
            
        except Exception as e:
            objective_logger.error(f"Strategy execution failed: {e}")
            objective_logger.error(traceback.format_exc())
            return self._create_failure_result(str(e))
    
    async def _create_strategy_instance(self, strategy_name: str, portfolio: Portfolio, parameters: Dict[str, Any]):
        """Create strategy instance with parameters"""
        
        if strategy_name == "momentum":
            if not MOMENTUM_AVAILABLE:
                raise ValueError("Enhanced Momentum Strategy not available")
            
            # Apply parameters to strategy
            strategy = EnhancedMomentumStrategy(portfolio=portfolio, **parameters)
            return strategy
        
        # Add other strategies here when implemented
        elif strategy_name == "bollinger_rsi":
            raise NotImplementedError("Bollinger RSI strategy not yet implemented")
        elif strategy_name == "rsi_ml":
            raise NotImplementedError("RSI ML strategy not yet implemented")
        elif strategy_name == "macd_ml":
            raise NotImplementedError("MACD ML strategy not yet implemented")
        elif strategy_name == "volume_profile":
            raise NotImplementedError("Volume Profile strategy not yet implemented")
        else:
            raise ValueError(f"Unknown strategy: {strategy_name}")
    
    def _validate_backtest_results(self, results: Dict[str, Any]) -> bool:
        """Validate backtest results for completeness"""
        
        required_fields = [
            'total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 
            'win_rate_pct', 'total_trades'
        ]
        
        for field in required_fields:
            if field not in results:
                objective_logger.warning(f"Missing required field: {field}")
                return False
            
            value = results[field]
            if value is None or (isinstance(value, float) and np.isnan(value)):
                objective_logger.warning(f"Invalid value for field {field}: {value}")
                return False
        
        return True
    
    def _create_failure_result(self, error_message: str) -> Dict[str, Any]:
        """Create failure result with minimal viable metrics"""
        
        return {
            'total_return_pct': -100.0,
            'sharpe_ratio': -10.0,
            'calmar_ratio': -10.0,
            'max_drawdown_pct': 100.0,
            'win_rate_pct': 0.0,
            'profit_factor': 0.0,
            'total_trades': 0,
            'volatility_annual': 100.0,
            'error_message': error_message
        }

class WalkForwardObjective:
    """🚶 Walk-Forward Analysis objective function"""
    
    def __init__(self, data_file: str):
        self.data_file = data_file
        self.executor = StrategyExecutor()
        self.metrics = ObjectiveMetrics()
    
    async def evaluate_period(
        self, 
        strategy_name: str,
        parameters: Dict[str, Any],
        start_date: str,
        end_date: str
    ) -> float:
        """Evaluate strategy performance for a specific time period"""
        
        results = await self.executor.execute_strategy_backtest(
            strategy_name=strategy_name,
            parameters=parameters,
            data_file=self.data_file,
            start_date=start_date,
            end_date=end_date
        )
        
        return self.metrics.calculate_composite_score(results)

# Main objective function
def run_objective(
    trial: optuna.Trial,
    strategy_name: str,
    data_file: str,
    start_date: str,
    end_date: str
) -> float:
    """
    🎯 MAIN OBJECTIVE FUNCTION
    💎 Universal performance evaluation for all strategies
    
    This function:
    1. Gets parameter space for the strategy
    2. Executes backtest with parameters
    3. Calculates composite score
    4. Returns optimization target
    """
    
    try:
        objective_logger.debug(f"Running objective for {strategy_name}, trial {trial.number}")
        
        # Get parameter space for strategy
        parameters = get_parameter_space(strategy_name, trial)
        
        objective_logger.debug(f"Trial {trial.number} parameters: {len(parameters)} params")
        
        # Execute strategy backtest
        executor = StrategyExecutor()
        
        # Run backtest synchronously (Optuna requirement)
        # Use existing event loop if available, otherwise create new one
        try:
            current_loop = asyncio.get_running_loop()
            # If loop is running, use asyncio.create_task (but this is tricky in sync context)
            # Better solution: make the execution sync
            results = executor.execute_strategy_backtest_sync(
                strategy_name=strategy_name,
                parameters=parameters,
                data_file=data_file,
                start_date=start_date,
                end_date=end_date
            )
        except RuntimeError:
            # No running loop, safe to create new one
            loop = asyncio.new_event_loop()
            asyncio.set_event_loop(loop)
            try:
                results = loop.run_until_complete(
                    executor.execute_strategy_backtest(
                        strategy_name=strategy_name,
                        parameters=parameters,
                        data_file=data_file,
                        start_date=start_date,
                        end_date=end_date
                    )
                )
            finally:
                loop.close()
        
        # Calculate composite score
        metrics = ObjectiveMetrics()
        composite_score = metrics.calculate_composite_score(results)
        
        # Log trial results
        return_pct = results.get('total_return_pct', 0)
        sharpe = results.get('sharpe_ratio', 0)
        drawdown = results.get('max_drawdown_pct', 0)
        
        objective_logger.info(
            f"Trial {trial.number}: Score={composite_score:.4f}, "
            f"Return={return_pct:.1f}%, Sharpe={sharpe:.2f}, DD={drawdown:.1f}%"
        )
        
        # Handle failures
        if 'error_message' in results:
            objective_logger.warning(f"Trial {trial.number} failed: {results['error_message']}")
            # Return very low score for failed trials
            return -1.0
        
        # Prune obviously bad trials early
        if composite_score < 0.1 and trial.number > 10:
            raise optuna.TrialPruned()
        
        return composite_score
        
    except optuna.TrialPruned:
        objective_logger.debug(f"Trial {trial.number} pruned")
        raise
    
    except Exception as e:
        objective_logger.error(f"Objective function error in trial {trial.number}: {e}")
        objective_logger.error(traceback.format_exc())
        
        # Return very low score for error cases
        return -1.0

# Utility functions for advanced optimization
def run_multi_period_objective(
    trial: optuna.Trial,
    strategy_name: str,
    data_file: str,
    time_periods: list
) -> float:
    """Run objective across multiple time periods for robustness"""
    
    scores = []
    
    for start_date, end_date in time_periods:
        try:
            score = run_objective(trial, strategy_name, data_file, start_date, end_date)
            scores.append(score)
        except optuna.TrialPruned:
            raise
        except Exception:
            scores.append(-1.0)  # Penalty for failed periods
    
    if not scores:
        return -1.0
    
    # Return average score across periods (robustness measure)
    avg_score = np.mean(scores)
    std_score = np.std(scores)
    
    # Penalize high variance (inconsistent performance)
    consistency_penalty = std_score * 0.1
    
    return max(-1.0, avg_score - consistency_penalty)

def validate_objective_function() -> bool:
    """Validate objective function dependencies"""
    
    validation_results = {
        "momentum_strategy": MOMENTUM_AVAILABLE,
        "backtester": BACKTESTER_AVAILABLE,
    }
    
    objective_logger.info("Objective Function Validation:")
    for component, available in validation_results.items():
        status = "✅" if available else "❌"
        objective_logger.info(f"  {status} {component}")
    
    return all(validation_results.values())

# Test function for development
async def test_objective_function():
    """Test objective function with sample parameters"""
    
    if not validate_objective_function():
        print("❌ Objective function validation failed")
        return
    
    # Create mock trial
    study = optuna.create_study(direction="maximize")
    
    def mock_objective(trial):
        return run_objective(
            trial=trial,
            strategy_name="momentum",
            data_file="historical_data/BTCUSDT_15m_20210101_20241231.csv",
            start_date="2024-01-01",
            end_date="2024-03-31"
        )
    
    try:
        study.optimize(mock_objective, n_trials=3)
        print(f"✅ Test completed. Best score: {study.best_value:.4f}")
        print(f"📊 Best parameters: {study.best_params}")
    except Exception as e:
        print(f"❌ Test failed: {e}")

if __name__ == "__main__":
    # Run validation and test
    if validate_objective_function():
        print("✅ Objective function validation passed")
        # Uncomment to run test
        # asyncio.run(test_objective_function())
    else:
        print("❌ Objective function validation failed")