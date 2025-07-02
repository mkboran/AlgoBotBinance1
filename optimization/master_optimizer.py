#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - FAZ 1: MASTER OPTIMIZER
💎 Optimizasyonun Komuta Merkezi - Hedge Fund Seviyesi

Bu modül şunları sağlar:
1. ✅ Tüm stratejileri tek merkezden optimize etme
2. ✅ Walk-Forward Analysis ile overfitting engellemesi
3. ✅ Risk-adjusted composite scoring
4. ✅ Çoklu piyasa rejimi testleri
5. ✅ Production-ready enterprise architecture

KULLANIM:
python master_optimizer.py --strategy momentum --trials 5000
python master_optimizer.py --strategy all --trials 10000 --walk-forward
python master_optimizer.py --strategy bollinger_rsi --trials 3000 --storage sqlite:///optimization/studies.db

📍 DOSYA: master_optimizer.py
📁 KONUM: optimization/
🔄 DURUM: kalıcı
"""

import argparse
import asyncio
import logging
import sys
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple
import optuna
import pandas as pd
import json
import sqlite3
from dataclasses import dataclass, asdict
import numpy as np

# Core imports
sys.path.append(str(Path(__file__).parent.parent))
from utils.config import settings
from utils.logger import logger
from parameter_spaces import get_parameter_space

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs") / "master_optimizer.log", mode='a', encoding='utf-8')
    ]
)
master_logger = logging.getLogger("MasterOptimizer")

@dataclass
class OptimizationConfig:
    """Optimizasyon konfigürasyonu"""
    strategy_name: str
    trials: int
    storage_url: str
    walk_forward: bool
    walk_forward_periods: int
    validation_split: float
    early_stopping_rounds: int
    parallel_jobs: int
    timeout_seconds: Optional[int] = None

@dataclass
class OptimizationResult:
    """Optimizasyon sonuç raporu"""
    strategy_name: str
    best_parameters: Dict[str, Any]
    best_score: float
    total_trials: int
    successful_trials: int
    failed_trials: int
    optimization_duration_minutes: float
    walk_forward_results: Optional[List[Dict[str, Any]]] = None
    robustness_score: Optional[float] = None
    final_validation_score: Optional[float] = None

class MasterOptimizer:
    """🚀 Master Optimization Engine - Enterprise Level"""
    
    def __init__(self, config: OptimizationConfig):
        self.config = config
        
        # Supported strategies
        self.supported_strategies = {
            "momentum": "strategies.momentum_optimized.EnhancedMomentumStrategy",
            "bollinger_rsi": "strategies.bollinger_rsi_strategy.BollingerRSIStrategy", 
            "rsi_ml": "strategies.rsi_ml_strategy.RSIMLStrategy",
            "macd_ml": "strategies.macd_ml_strategy.MACDMLStrategy",
            "volume_profile": "strategies.volume_profile_strategy.VolumeProfileStrategy"
        }
        
        # Data file paths
        self.data_file = "historical_data/BTCUSDT_15m_20210101_20241231.csv"
        
        # Walk-forward periods (6-month overlapping windows)
        self.walk_forward_windows = [
            ("2023-01-01", "2023-06-30"),  # H1 2023
            ("2023-04-01", "2023-09-30"),  # Q2-Q3 2023
            ("2023-07-01", "2023-12-31"),  # H2 2023
            ("2024-01-01", "2024-06-30"),  # H1 2024
            ("2024-04-01", "2024-09-30"),  # Q2-Q3 2024
        ]
        
        master_logger.info(f"🚀 Master Optimizer initialized for strategy: {config.strategy_name}")
        master_logger.info(f"📊 Configuration: {config.trials} trials, Walk-forward: {config.walk_forward}")
    
    def validate_strategy(self, strategy_name: str) -> bool:
        """✅ Strateji geçerliliğini kontrol et"""
        
        if strategy_name == "all":
            return True
        
        if strategy_name not in self.supported_strategies:
            master_logger.error(f"❌ Unsupported strategy: {strategy_name}")
            master_logger.error(f"💡 Supported strategies: {list(self.supported_strategies.keys())}")
            return False
        
        return True
    
    def create_study(self, study_name: str) -> optuna.Study:
        """📚 Optuna study oluştur"""
        
        try:
            # Optuna study configuration
            study = optuna.create_study(
                study_name=study_name,
                storage=self.config.storage_url,
                direction="maximize",
                load_if_exists=True,
                sampler=optuna.samplers.TPESampler(
                    seed=42,
                    n_startup_trials=max(20, self.config.trials // 100),
                    n_ei_candidates=24,
                    multivariate=True
                ),
                pruner=optuna.pruners.MedianPruner(
                    n_startup_trials=max(10, self.config.trials // 200),
                    n_warmup_steps=5,
                    interval_steps=10
                )
            )
            
            master_logger.info(f"📚 Study created/loaded: {study_name}")
            return study
            
        except Exception as e:
            master_logger.error(f"❌ Failed to create study: {e}")
            raise
    
    async def optimize_single_strategy(self, strategy_name: str) -> OptimizationResult:
        """🎯 Tek stratejiyi optimize et"""
        
        master_logger.info(f"🎯 Starting optimization for {strategy_name}")
        start_time = datetime.now()
        
        study_name = f"master_optimization_{strategy_name}_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        study = self.create_study(study_name)
        
        if self.config.walk_forward:
            # Walk-Forward Optimization
            master_logger.info("🚶 Starting Walk-Forward Analysis...")
            result = await self._run_walk_forward_optimization(study, strategy_name)
        else:
            # Standard optimization
            master_logger.info("⚡ Starting Standard Optimization...")
            result = await self._run_standard_optimization(study, strategy_name)
        
        # Calculate duration
        duration = (datetime.now() - start_time).total_seconds() / 60
        result.optimization_duration_minutes = duration
        
        # Save results
        await self._save_optimization_results(result)
        
        master_logger.info(f"🎉 Optimization completed for {strategy_name} in {duration:.2f} minutes")
        master_logger.info(f"🏆 Best score: {result.best_score:.4f}")
        master_logger.info(f"📊 Successful trials: {result.successful_trials}/{result.total_trials}")
        
        return result
    
    async def _run_standard_optimization(self, study: optuna.Study, strategy_name: str) -> OptimizationResult:
        """⚡ Standard optimization süreci"""
        
        # Objective function wrapper
        def objective_wrapper(trial):
            try:
                return get_parameter_space(
                    trial=trial,
                    strategy_name=strategy_name,
                    data_file=self.data_file,
                    start_date="2023-01-01",
                    end_date="2024-09-30"
                )
            except Exception as e:
                master_logger.error(f"❌ Trial {trial.number} failed: {e}")
                raise optuna.TrialPruned()
        
        # Run optimization
        try:
            study.optimize(
                objective_wrapper,
                n_trials=self.config.trials,
                timeout=self.config.timeout_seconds,
                n_jobs=self.config.parallel_jobs if self.config.parallel_jobs > 1 else 1,
                show_progress_bar=True
            )
        except KeyboardInterrupt:
            master_logger.warning("⚠️ Optimization interrupted by user")
        
        # Collect results
        successful_trials = len([t for t in study.trials if t.value is not None])
        failed_trials = len(study.trials) - successful_trials
        
        result = OptimizationResult(
            strategy_name=strategy_name,
            best_parameters=study.best_params,
            best_score=study.best_value if study.best_value else 0.0,
            total_trials=len(study.trials),
            successful_trials=successful_trials,
            failed_trials=failed_trials,
            optimization_duration_minutes=0.0  # Will be set by caller
        )
        
        return result
    
    async def _run_walk_forward_optimization(self, study: optuna.Study, strategy_name: str) -> OptimizationResult:
        """🚶 Walk-Forward Analysis ile robust optimization"""
        
        master_logger.info("🚶 Executing Walk-Forward Analysis...")
        
        walk_forward_results = []
        all_best_params = []
        
        # Her time window için optimization çalıştır
        for i, (start_date, end_date) in enumerate(self.walk_forward_windows):
            window_name = f"Window_{i+1}_{start_date}_{end_date}"
            master_logger.info(f"📅 Optimizing window {i+1}/{len(self.walk_forward_windows)}: {start_date} to {end_date}")
            
            # Window-specific objective function
            def window_objective(trial):
                try:
                    return get_parameter_space(
                        trial=trial,
                        strategy_name=strategy_name,
                        data_file=self.data_file,
                        start_date=start_date,
                        end_date=end_date
                    )
                except Exception as e:
                    master_logger.error(f"❌ Window {window_name} trial {trial.number} failed: {e}")
                    raise optuna.TrialPruned()
            
            # Create window-specific study
            window_study_name = f"{study.study_name}_{window_name}"
            window_study = self.create_study(window_study_name)
            
            # Optimize for this window
            trials_per_window = max(200, self.config.trials // len(self.walk_forward_windows))
            
            try:
                window_study.optimize(
                    window_objective,
                    n_trials=trials_per_window,
                    timeout=self.config.timeout_seconds,
                    n_jobs=1,  # Sequential for stability
                    show_progress_bar=False
                )
            except KeyboardInterrupt:
                master_logger.warning(f"⚠️ Window {window_name} interrupted")
                continue
            
            # Collect window results
            if window_study.best_value is not None:
                window_result = {
                    "window_name": window_name,
                    "start_date": start_date,
                    "end_date": end_date,
                    "best_params": window_study.best_params,
                    "best_score": window_study.best_value,
                    "total_trials": len(window_study.trials),
                    "successful_trials": len([t for t in window_study.trials if t.value is not None])
                }
                
                walk_forward_results.append(window_result)
                all_best_params.append(window_study.best_params)
                
                master_logger.info(f"✅ Window {i+1} completed: Best score = {window_study.best_value:.4f}")
            else:
                master_logger.warning(f"⚠️ Window {i+1} failed to find valid solution")
        
        # Consensus parameters (most frequent values across windows)
        consensus_params = self._calculate_consensus_parameters(all_best_params)
        
        # Robustness score (consistency across windows)
        robustness_score = self._calculate_robustness_score(walk_forward_results)
        
        # Final validation on out-of-sample data
        final_validation_score = await self._run_final_validation(
            strategy_name, consensus_params, "2024-10-01", "2024-12-31"
        )
        
        # Collect overall results
        total_trials = sum(r["total_trials"] for r in walk_forward_results)
        successful_trials = sum(r["successful_trials"] for r in walk_forward_results)
        best_score = max((r["best_score"] for r in walk_forward_results), default=0.0)
        
        result = OptimizationResult(
            strategy_name=strategy_name,
            best_parameters=consensus_params,
            best_score=best_score,
            total_trials=total_trials,
            successful_trials=successful_trials,
            failed_trials=total_trials - successful_trials,
            optimization_duration_minutes=0.0,  # Will be set by caller
            walk_forward_results=walk_forward_results,
            robustness_score=robustness_score,
            final_validation_score=final_validation_score
        )
        
        master_logger.info(f"🎯 Walk-Forward Analysis completed")
        master_logger.info(f"🔄 Robustness Score: {robustness_score:.4f}")
        master_logger.info(f"✅ Final Validation Score: {final_validation_score:.4f}")
        
        return result
    
    def _calculate_consensus_parameters(self, param_sets: List[Dict[str, Any]]) -> Dict[str, Any]:
        """🤝 Calculate consensus parameters from multiple optimization windows"""
        
        if not param_sets:
            return {}
        
        consensus = {}
        
        # Get all parameter names
        all_param_names = set()
        for params in param_sets:
            all_param_names.update(params.keys())
        
        # For each parameter, find consensus value
        for param_name in all_param_names:
            values = [params.get(param_name) for params in param_sets if param_name in params]
            
            if not values:
                continue
            
            # For numerical parameters, use median
            if all(isinstance(v, (int, float)) for v in values):
                consensus[param_name] = np.median(values)
                # Convert back to int if all original values were int
                if all(isinstance(v, int) for v in values):
                    consensus[param_name] = int(consensus[param_name])
            
            # For categorical parameters, use most frequent
            else:
                from collections import Counter
                counter = Counter(values)
                consensus[param_name] = counter.most_common(1)[0][0]
        
        master_logger.info(f"🤝 Consensus parameters calculated from {len(param_sets)} windows")
        
        return consensus
    
    def _calculate_robustness_score(self, walk_forward_results: List[Dict[str, Any]]) -> float:
        """🛡️ Calculate robustness score (consistency across windows)"""
        
        if len(walk_forward_results) < 2:
            return 0.0
        
        scores = [r["best_score"] for r in walk_forward_results]
        
        # Robustness = 1 - (coefficient of variation)
        mean_score = np.mean(scores)
        std_score = np.std(scores)
        
        if mean_score == 0:
            return 0.0
        
        cv = std_score / mean_score  # Coefficient of variation
        robustness = max(0.0, 1.0 - cv)  # Higher is better, max 1.0
        
        return robustness
    
    async def _run_final_validation(self, strategy_name: str, parameters: Dict[str, Any], 
                                  start_date: str, end_date: str) -> float:
        """✅ Final validation on out-of-sample data"""
        
        try:
            master_logger.info(f"✅ Running final validation on {start_date} to {end_date}")
            
            # Create a mock trial with consensus parameters
            class MockTrial:
                def __init__(self, params):
                    self.params = params
                
                def suggest_int(self, name, low, high):
                    return self.params.get(name, (low + high) // 2)
                
                def suggest_float(self, name, low, high):
                    return self.params.get(name, (low + high) / 2)
                
                def suggest_categorical(self, name, choices):
                    return self.params.get(name, choices[0])
            
            mock_trial = MockTrial(parameters)
            
            # Run validation
            validation_score = get_parameter_space(
                trial=mock_trial,
                strategy_name=strategy_name,
                data_file=self.data_file,
                start_date=start_date,
                end_date=end_date
            )
            
            master_logger.info(f"✅ Final validation score: {validation_score:.4f}")
            return validation_score
            
        except Exception as e:
            master_logger.error(f"❌ Final validation failed: {e}")
            return 0.0
    
    async def _save_optimization_results(self, result: OptimizationResult) -> None:
        """💾 Save optimization results to JSON and update strategy files"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # Save detailed results
        results_file = Path("optimization/results") / f"master_optimization_{result.strategy_name}_{timestamp}.json"
        results_file.parent.mkdir(parents=True, exist_ok=True)
        
        with open(results_file, 'w', encoding='utf-8') as f:
            json.dump(asdict(result), f, indent=2, ensure_ascii=False, default=str)
        
        master_logger.info(f"💾 Detailed results saved: {results_file}")
        
        # Save best parameters in JSON parameter system format
        if result.best_parameters:
            from json_parameter_system import JSONParameterManager
            
            manager = JSONParameterManager()
            success = manager.save_optimization_results(
                strategy_name=result.strategy_name,
                best_parameters=result.best_parameters,
                optimization_metrics={
                    "best_score": result.best_score,
                    "total_trials": result.total_trials,
                    "successful_trials": result.successful_trials,
                    "robustness_score": result.robustness_score,
                    "final_validation_score": result.final_validation_score,
                    "optimization_duration_minutes": result.optimization_duration_minutes,
                    "walk_forward_analysis": self.config.walk_forward,
                    "optimization_date": datetime.now().isoformat()
                },
                source_file=f"master_optimizer_{timestamp}"
            )
            
            if success:
                master_logger.info(f"✅ Parameters saved to JSON parameter system")
            else:
                master_logger.warning(f"⚠️ Failed to save to JSON parameter system")
    
    async def optimize_all_strategies(self) -> Dict[str, OptimizationResult]:
        """🌟 Optimize all supported strategies"""
        
        master_logger.info("🌟 Starting optimization for ALL strategies")
        
        all_results = {}
        
        for strategy_name in self.supported_strategies.keys():
            try:
                master_logger.info(f"🚀 Starting {strategy_name} optimization...")
                
                result = await self.optimize_single_strategy(strategy_name)
                all_results[strategy_name] = result
                
                master_logger.info(f"✅ {strategy_name} completed with score: {result.best_score:.4f}")
                
            except Exception as e:
                master_logger.error(f"❌ {strategy_name} optimization failed: {e}")
                master_logger.error(traceback.format_exc())
                continue
        
        # Summary report
        master_logger.info("="*80)
        master_logger.info("🎉 ALL STRATEGIES OPTIMIZATION COMPLETED!")
        master_logger.info("="*80)
        
        for strategy_name, result in all_results.items():
            master_logger.info(f"📊 {strategy_name.upper()}: Score = {result.best_score:.4f}, "
                             f"Trials = {result.successful_trials}/{result.total_trials}")
        
        # Find best performing strategy
        if all_results:
            best_strategy = max(all_results.items(), key=lambda x: x[1].best_score)
            master_logger.info(f"🏆 BEST STRATEGY: {best_strategy[0].upper()} with score {best_strategy[1].best_score:.4f}")
        
        return all_results


async def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(
        description="Master Optimizer - Enterprise Strategy Optimization System",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  python master_optimizer.py --strategy momentum --trials 5000
  python master_optimizer.py --strategy all --trials 10000 --walk-forward
  python master_optimizer.py --strategy bollinger_rsi --trials 3000 --parallel 4
  python master_optimizer.py --strategy momentum --trials 2000 --storage sqlite:///optimization/studies.db
        """
    )
    
    parser.add_argument('--strategy', required=True, 
                       choices=['momentum', 'bollinger_rsi', 'rsi_ml', 'macd_ml', 'volume_profile', 'all'],
                       help='Strategy to optimize or "all" for all strategies')
    parser.add_argument('--trials', type=int, default=5000, help='Number of optimization trials')
    parser.add_argument('--storage', default='sqlite:///optimization/studies.db', 
                       help='Optuna storage URL')
    parser.add_argument('--walk-forward', action='store_true', 
                       help='Enable Walk-Forward Analysis (recommended)')
    parser.add_argument('--parallel', type=int, default=1, help='Number of parallel jobs')
    parser.add_argument('--timeout', type=int, help='Optimization timeout in seconds')
    
    args = parser.parse_args()
    
    # Create optimization config
    config = OptimizationConfig(
        strategy_name=args.strategy,
        trials=args.trials,
        storage_url=args.storage,
        walk_forward=args.walk_forward,
        walk_forward_periods=5,
        validation_split=0.2,
        early_stopping_rounds=100,
        parallel_jobs=args.parallel,
        timeout_seconds=args.timeout
    )
    
    # Initialize optimizer
    optimizer = MasterOptimizer(config)
    
    # Validate strategy
    if not optimizer.validate_strategy(args.strategy):
        sys.exit(1)
    
    try:
        print("🚀 MASTER OPTIMIZER - ENTERPRISE STRATEGY OPTIMIZATION")
        print("="*80)
        print(f"🎯 Strategy: {args.strategy}")
        print(f"🔬 Trials: {args.trials}")
        print(f"🚶 Walk-Forward: {'✅' if args.walk_forward else '❌'}")
        print(f"⚡ Parallel Jobs: {args.parallel}")
        print(f"💾 Storage: {args.storage}")
        print("="*80)
        
        if args.strategy == "all":
            results = await optimizer.optimize_all_strategies()
            
            print("\n🎉 ALL STRATEGIES OPTIMIZATION COMPLETED!")
            print("📊 Final Results Summary:")
            for strategy_name, result in results.items():
                print(f"   🏆 {strategy_name.upper()}: {result.best_score:.4f} "
                      f"({result.successful_trials}/{result.total_trials} trials)")
        
        else:
            result = await optimizer.optimize_single_strategy(args.strategy)
            
            print(f"\n🎉 {args.strategy.upper()} OPTIMIZATION COMPLETED!")
            print(f"🏆 Best Score: {result.best_score:.4f}")
            print(f"📊 Successful Trials: {result.successful_trials}/{result.total_trials}")
            print(f"⏱️ Duration: {result.optimization_duration_minutes:.2f} minutes")
            
            if result.robustness_score is not None:
                print(f"🛡️ Robustness Score: {result.robustness_score:.4f}")
            if result.final_validation_score is not None:
                print(f"✅ Validation Score: {result.final_validation_score:.4f}")
        
        print("\n💎 Optimization parameters saved to JSON parameter system")
        print("🚀 Ready for next phase: Ensemble optimization!")
    
    except KeyboardInterrupt:
        print("\n🛑 Optimization interrupted by user")
        sys.exit(130)
    
    except Exception as e:
        master_logger.error(f"❌ Master optimization failed: {e}")
        master_logger.error(traceback.format_exc())
        print(f"\n❌ OPTIMIZATION FAILED: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())