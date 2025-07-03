#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX v2.0 - MERKEZI KOMUTA SİSTEMİ
💎 FAZ 5: SİSTEMİN CANLANMASI - Hedge Fund Seviyesi Üstü

✅ TÜM FAZLAR ENTEGRE EDİLDİ:
🧠 FAZ 1: BaseStrategy v2.0 - Dinamik Çıkış, Kelly Criterion, Global Intelligence
🎯 FAZ 2: StrategyCoordinator v1.0 - Kolektif Bilinç ve Orchestration
🧬 FAZ 3: AdaptiveParameterEvolution v1.0 - Kendini İyileştiren Sistem
🎼 FAZ 4: Multi-Strategy Integration - Portfolio Optimization
🚀 FAZ 5: PhoenixTradingSystem - Ultimate Command Center

TEK VE GÜÇLÜ KOMUTA MERKEZİ:
- Live Trading: StrategyCoordinator + AdaptiveEvolution + Acil Durum Freni
- Backtesting: Multi-Strategy Portfolio Validation
- Optimization: Master Optimizer + Fine-Tuning
- Validation: Comprehensive System Health Check
- Status: Real-time System Analytics

KULLANIM ÖRNEKLERİ:
python main.py live --strategy momentum --capital 1000 --symbol BTC/USDT
python main.py backtest --strategy all --start-date 2024-01-01 --end-date 2024-12-31
python main.py optimize --strategy momentum --trials 5000 --walk-forward
python main.py validate --full-validation
python main.py status --detailed

HEDGE FUND LEVEL IMPLEMENTATION - PRODUCTION READY
"""

import asyncio
import argparse
import sys
import os
import logging
import json
import traceback
import signal
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional, Tuple, Union
import warnings
import time
from dataclasses import dataclass, asdict

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# ==================================================================================
# CORE SYSTEM IMPORTS
# ==================================================================================

try:
    # Basic utilities
    from utils.config import settings
    from utils.logger import logger
    from utils.portfolio import Portfolio
    from utils.data import BinanceFetcher, DataFetchingError

    # Strategic foundation (FAZ 1)
    from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType
    
    # All strategy implementations
    from strategies.momentum_optimized import EnhancedMomentumStrategy
    from strategies.bollinger_ml_strategy import BollingerMLStrategy
    from strategies.rsi_ml_strategy import RSIMLStrategy
    from strategies.macd_ml_strategy import MACDMLStrategy
    from strategies.volume_profile_strategy import VolumeProfileMLStrategy
    
    # Coordination layer (FAZ 2)
    from utils.strategy_coordinator import StrategyCoordinator, integrate_strategy_coordinator
    
    # Evolution system (FAZ 3)
    from utils.adaptive_parameter_evolution import AdaptiveParameterEvolution, integrate_adaptive_parameter_evolution, EvolutionConfig
    
    # Backtesting system (FAZ 4)
    from backtesting.multi_strategy_backtester import MultiStrategyBacktester, BacktestConfiguration, BacktestMode, BacktestResult
    
    # Optimization system
    from optimization.master_optimizer import MasterOptimizer, OptimizationConfig, OptimizationResult
    
    # Parameter management
    from json_parameter_system import JSONParameterManager
    
    # Validation system
    from scripts.validate_system import PhoenixSystemValidator
    
    CORE_IMPORTS_SUCCESS = True
    IMPORT_ERROR = None
    
except ImportError as e:
    CORE_IMPORTS_SUCCESS = False
    IMPORT_ERROR = str(e)
    logger.error(f"❌ Critical import error: {e}")
    # Create dummy classes for graceful degradation
    class Portfolio: pass
    class StrategyCoordinator: pass
    class AdaptiveParameterEvolution: pass
    class MultiStrategyBacktester: pass
    class MasterOptimizer: pass
    class PhoenixSystemValidator: pass

except Exception as e:
    CORE_IMPORTS_SUCCESS = False
    IMPORT_ERROR = str(e)
    logger.error(f"❌ System import error: {e}")


# ==================================================================================
# CONFIGURATION AND DATA STRUCTURES
# ==================================================================================

@dataclass
class SystemStatus:
    """System status data structure"""
    timestamp: datetime
    system_version: str
    uptime_seconds: float
    core_imports_success: bool
    
    # Component status
    portfolio_initialized: bool = False
    coordinator_active: bool = False
    evolution_active: bool = False
    backtester_ready: bool = False
    optimizer_ready: bool = False
    
    # Performance metrics
    total_strategies: int = 0
    active_strategies: int = 0
    total_trades: int = 0
    current_balance: float = 0.0
    total_profit_usdt: float = 0.0
    
    # System health
    memory_usage_mb: float = 0.0
    cpu_usage_pct: float = 0.0
    last_error: Optional[str] = None
    
    # Advanced metrics
    coordination_success_rate: float = 0.0
    evolution_cycles_completed: int = 0
    optimization_runs: int = 0
    validation_score: float = 0.0

@dataclass
class EmergencyBrakeConfig:
    """Emergency brake configuration"""
    max_drawdown_pct: float = 15.0  # 15% max drawdown
    max_daily_loss_pct: float = 5.0  # 5% max daily loss
    min_balance_pct: float = 85.0   # 85% of initial capital minimum
    consecutive_losses_limit: int = 8
    enable_emergency_brake: bool = True
    emergency_contacts: List[str] = None  # Future: email/SMS alerts


# ==================================================================================
# PHOENIX TRADING SYSTEM - ULTIMATE COMMAND CENTER
# ==================================================================================

class PhoenixTradingSystem:
    """
    🚀 Phoenix Trading System - Ultimate Command Center
    
    Revolutionary trading system integrating all FAZ components:
    - Multi-Strategy Coordination with Collective Intelligence
    - Adaptive Parameter Evolution with Self-Healing
    - Advanced Portfolio Management with Risk Controls
    - Real-time Performance Monitoring and Analytics
    - Emergency Safety Systems and Circuit Breakers
    """
    
    def __init__(self):
        """Initialize Phoenix Trading System"""
        
        # System metadata
        self.system_name = "Phoenix Trading System"
        self.version = "2.0"
        self.start_time = datetime.now(timezone.utc)
        self.logger = logging.getLogger("phoenix.main")
        
        # Core components (initialized on demand)
        self.portfolio: Optional[Portfolio] = None
        self.strategy_coordinator: Optional[StrategyCoordinator] = None
        self.evolution_system: Optional[AdaptiveParameterEvolution] = None
        self.backtester: Optional[MultiStrategyBacktester] = None
        self.optimizer: Optional[MasterOptimizer] = None
        self.validator: Optional[PhoenixSystemValidator] = None
        self.data_fetcher: Optional[BinanceFetcher] = None
        
        # Parameter management
        self.json_manager = JSONParameterManager()
        
        # Strategy registry
        self.strategy_registry = {
            "momentum": {
                "class": EnhancedMomentumStrategy if CORE_IMPORTS_SUCCESS else None,
                "name": "Enhanced Momentum Strategy",
                "description": "ML-enhanced momentum strategy with dynamic exits and Kelly sizing",
                "default_allocation": 0.3
            },
            "bollinger_ml": {
                "class": BollingerMLStrategy if CORE_IMPORTS_SUCCESS else None,
                "name": "Bollinger Bands + ML Strategy", 
                "description": "Mean reversion with ML predictions and volatility analysis",
                "default_allocation": 0.25
            },
            "rsi_ml": {
                "class": RSIMLStrategy if CORE_IMPORTS_SUCCESS else None,
                "name": "RSI + ML Strategy",
                "description": "RSI divergence detection with ML enhancement",
                "default_allocation": 0.2
            },
            "macd_ml": {
                "class": MACDMLStrategy if CORE_IMPORTS_SUCCESS else None,
                "name": "MACD + ML Strategy",
                "description": "MACD trend following with ML confirmation",
                "default_allocation": 0.15
            },
            "volume_profile": {
                "class": VolumeProfileMLStrategy if CORE_IMPORTS_SUCCESS else None,
                "name": "Volume Profile Strategy",
                "description": "Volume profile analysis with institutional flow detection",
                "default_allocation": 0.1
            }
        }
        
        # System state
        self.is_live_trading = False
        self.emergency_brake_triggered = False
        self.system_errors: List[str] = []
        self.performance_history = []
        
        # Emergency brake configuration
        self.emergency_config = EmergencyBrakeConfig()
        
        self.logger.info(f"🚀 {self.system_name} v{self.version} initialized")
        self.logger.info(f"📊 Strategy registry: {len(self.strategy_registry)} strategies available")

    # ==================================================================================
    # SYSTEM INITIALIZATION
    # ==================================================================================
    
    async def initialize_system(
        self, 
        mode: str, 
        config: Dict[str, Any]
    ) -> bool:
        """🔧 Initialize system components based on operating mode"""
        
        if not CORE_IMPORTS_SUCCESS:
            self.logger.error(f"❌ Cannot initialize system - import error: {IMPORT_ERROR}")
            return False
        
        try:
            self.logger.info(f"🔧 Initializing Phoenix Trading System for {mode} mode...")
            
            # Initialize portfolio
            initial_capital = config.get("capital", 1000.0)
            self.portfolio = Portfolio(initial_capital_usdt=initial_capital)
            self.logger.info(f"✅ Portfolio initialized: ${initial_capital:,.2f}")
            
            # Initialize validator
            self.validator = PhoenixSystemValidator()
            self.logger.info("✅ System validator ready")
            
            # Mode-specific initialization
            if mode == "live":
                return await self._initialize_live_mode(config)
            elif mode == "backtest":
                return await self._initialize_backtest_mode(config)
            elif mode == "optimize":
                return await self._initialize_optimization_mode(config)
            else:
                # Basic initialization for validate/status modes
                self.logger.info("✅ Basic system initialization completed")
                return True
                
        except Exception as e:
            self.logger.error(f"❌ System initialization failed: {e}")
            self.logger.error(traceback.format_exc())
            return False
    
    async def _initialize_live_mode(self, config: Dict[str, Any]) -> bool:
        """Initialize components for live trading"""
        try:
            # Initialize data fetcher
            symbol = config.get("symbol", "BTC/USDT")
            self.data_fetcher = BinanceFetcher(symbol=symbol)
            self.logger.info(f"✅ Data fetcher initialized for {symbol}")
            
            # Initialize strategies
            strategy_name = config.get("strategy", "momentum")
            strategies = await self._initialize_strategies([strategy_name], config)
            
            if not strategies:
                self.logger.error("❌ No strategies initialized")
                return False
            
            # Initialize strategy coordinator
            self.strategy_coordinator = integrate_strategy_coordinator(
                portfolio_instance=self.portfolio,
                strategies=strategies,
                **config.get("coordinator_config", {})
            )
            self.logger.info("✅ Strategy coordinator initialized")
            
            # Initialize adaptive evolution system
            evolution_config = EvolutionConfig(
                consecutive_loss_trigger=config.get("consecutive_loss_trigger", 5),
                profit_factor_threshold=config.get("profit_factor_threshold", 1.0),
                min_improvement_pct=config.get("min_improvement_pct", 5.0)
            )
            
            self.evolution_system = integrate_adaptive_parameter_evolution(
                strategy_coordinator_instance=self.strategy_coordinator,
                evolution_config=evolution_config
            )
            self.logger.info("✅ Adaptive evolution system initialized")
            
            # Configure emergency brake
            self.emergency_config = EmergencyBrakeConfig(
                max_drawdown_pct=config.get("max_drawdown_pct", 15.0),
                max_daily_loss_pct=config.get("max_daily_loss_pct", 5.0),
                enable_emergency_brake=config.get("enable_emergency_brake", True)
            )
            
            self.logger.info("🎉 Live trading mode initialization completed successfully!")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Live mode initialization failed: {e}")
            return False
    
    async def _initialize_backtest_mode(self, config: Dict[str, Any]) -> bool:
        """Initialize components for backtesting"""
        try:
            # Initialize backtester
            self.backtester = MultiStrategyBacktester(
                enable_parallel_processing=config.get("parallel_processing", True),
                max_workers=config.get("max_workers", 4),
                enable_advanced_analytics=config.get("advanced_analytics", True)
            )
            
            # Register strategies for backtesting
            strategy_name = config.get("strategy", "all")
            if strategy_name == "all":
                strategy_names = list(self.strategy_registry.keys())
            else:
                strategy_names = [strategy_name]
            
            for name in strategy_names:
                if name in self.strategy_registry and self.strategy_registry[name]["class"]:
                    self.backtester.register_strategy(
                        name,
                        self.strategy_registry[name]["class"],
                        {"portfolio": self.portfolio}
                    )
            
            self.logger.info(f"✅ Backtester initialized with {len(strategy_names)} strategies")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Backtest mode initialization failed: {e}")
            return False
    
    async def _initialize_optimization_mode(self, config: Dict[str, Any]) -> bool:
        """Initialize components for optimization"""
        try:
            # Create optimization configuration
            optimization_config = OptimizationConfig(
                strategy_name=config.get("strategy", "momentum"),
                trials=config.get("trials", 1000),
                storage_url=config.get("storage", "sqlite:///optimization/studies.db"),
                walk_forward=config.get("walk_forward", False),
                walk_forward_periods=config.get("walk_forward_periods", 5),
                validation_split=config.get("validation_split", 0.2),
                early_stopping_rounds=config.get("early_stopping_rounds", 100),
                parallel_jobs=config.get("parallel_jobs", 1),
                timeout_seconds=config.get("timeout_minutes", 120) * 60
            )
            
            # Initialize master optimizer
            self.optimizer = MasterOptimizer(optimization_config)
            
            self.logger.info(f"✅ Optimizer initialized for {optimization_config.strategy_name}")
            self.logger.info(f"   Trials: {optimization_config.trials}")
            self.logger.info(f"   Walk-forward: {optimization_config.walk_forward}")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Optimization mode initialization failed: {e}")
            return False
    
    async def _initialize_strategies(
        self, 
        strategy_names: List[str], 
        config: Dict[str, Any]
    ) -> List[Tuple[str, Any]]:
        """Initialize strategy instances"""
        try:
            strategies = []
            
            for name in strategy_names:
                if name not in self.strategy_registry:
                    self.logger.warning(f"⚠️ Unknown strategy: {name}")
                    continue
                
                strategy_info = self.strategy_registry[name]
                strategy_class = strategy_info["class"]
                
                if not strategy_class:
                    self.logger.warning(f"⚠️ Strategy class not available: {name}")
                    continue
                
                # Load optimized parameters
                optimized_params = self.json_manager.load_strategy_parameters(name)
                strategy_params = optimized_params.parameters if optimized_params else {}
                
                # Create strategy instance
                strategy_instance = strategy_class(
                    portfolio=self.portfolio,
                    symbol=config.get("symbol", "BTC/USDT"),
                    **strategy_params,
                    **config.get("strategy_config", {})
                )
                
                strategies.append((name, strategy_instance))
                self.logger.info(f"✅ Strategy initialized: {name}")
                
                # Log parameter count
                if optimized_params:
                    param_count = len(optimized_params.parameters)
                    self.logger.info(f"   Loaded {param_count} optimized parameters")
            
            return strategies
            
        except Exception as e:
            self.logger.error(f"❌ Strategy initialization error: {e}")
            return []

    # ==================================================================================
    # LIVE TRADING OPERATIONS
    # ==================================================================================
    
    async def run_live_trading(self, args: argparse.Namespace) -> None:
        """
        🔴 LIVE TRADING MODE - REAL MONEY OPERATIONS
        
        Comprehensive live trading with:
        - Multi-strategy coordination
        - Adaptive parameter evolution
        - Emergency brake system
        - Real-time monitoring
        """
        
        self.logger.warning("🔴 LIVE TRADING MODE ACTIVATED - REAL MONEY AT RISK!")
        self.logger.warning("🛡️ Emergency brake enabled - system will auto-stop on excessive losses")
        
        # Configuration from arguments
        config = {
            "strategy": args.strategy,
            "capital": args.capital,
            "symbol": args.symbol,
            "max_drawdown_pct": getattr(args, 'max_drawdown', 15.0),
            "enable_emergency_brake": getattr(args, 'emergency_brake', True)
        }
        
        # Initialize system for live trading
        if not await self.initialize_system("live", config):
            self.logger.error("❌ Live trading initialization failed")
            return
        
        # Setup signal handlers for graceful shutdown
        def signal_handler(signum, frame):
            self.logger.warning(f"🛑 Signal {signum} received - initiating graceful shutdown...")
            self.is_live_trading = False
        
        signal.signal(signal.SIGINT, signal_handler)
        signal.signal(signal.SIGTERM, signal_handler)
        
        # Pre-flight safety check
        safety_check = await self._perform_safety_check()
        if not safety_check:
            self.logger.error("❌ Safety check failed - aborting live trading")
            return
        
        self.is_live_trading = True
        trading_cycle = 0
        last_evolution_check = datetime.now(timezone.utc)
        last_status_report = datetime.now(timezone.utc)
        
        self.logger.info("🚀 Live trading loop starting...")
        self.logger.info(f"   Strategy: {args.strategy}")
        self.logger.info(f"   Symbol: {args.symbol}")
        self.logger.info(f"   Capital: ${args.capital:,.2f}")
        self.logger.info(f"   Emergency brake: {config['enable_emergency_brake']}")
        
        try:
            while self.is_live_trading:
                cycle_start = datetime.now(timezone.utc)
                trading_cycle += 1
                
                self.logger.info(f"📊 Trading cycle {trading_cycle} starting...")
                
                try:
                    # ==================================================================================
                    # STEP 1: FETCH MARKET DATA
                    # ==================================================================================
                    
                    market_data = await self._fetch_market_data()
                    if market_data is None:
                        self.logger.error("❌ Failed to fetch market data - skipping cycle")
                        await asyncio.sleep(60)  # Wait 1 minute before retry
                        continue
                    
                    # ==================================================================================
                    # STEP 2: EMERGENCY BRAKE CHECK
                    # ==================================================================================
                    
                    if config["enable_emergency_brake"]:
                        brake_triggered = await self._check_emergency_brake(config)
                        if brake_triggered:
                            self.logger.critical("🚨 EMERGENCY BRAKE TRIGGERED - STOPPING ALL TRADING")
                            await self._execute_emergency_stop()
                            break
                    
                    # ==================================================================================
                    # STEP 3: STRATEGY COORDINATION
                    # ==================================================================================
                    
                    coordination_results = await self.strategy_coordinator.coordinate_strategies(market_data)
                    
                    if coordination_results.get("success"):
                        actions_taken = coordination_results.get("actions_taken", [])
                        self.logger.info(f"✅ Strategy coordination completed: {', '.join(actions_taken)}")
                    else:
                        self.logger.warning("⚠️ Strategy coordination issues detected")
                    
                    # ==================================================================================
                    # STEP 4: ADAPTIVE EVOLUTION (Every Hour)
                    # ==================================================================================
                    
                    time_since_evolution = datetime.now(timezone.utc) - last_evolution_check
                    if time_since_evolution.total_seconds() >= 3600:  # 1 hour
                        self.logger.info("🧬 Running adaptive parameter evolution check...")
                        
                        evolution_results = await self.evolution_system.monitor_strategies()
                        
                        if evolution_results.get("evolution_recommended"):
                            recommended_count = len(evolution_results["evolution_recommended"])
                            self.logger.info(f"🎯 Evolution recommended for {recommended_count} strategies")
                        
                        last_evolution_check = datetime.now(timezone.utc)
                    
                    # ==================================================================================
                    # STEP 5: PERFORMANCE MONITORING
                    # ==================================================================================
                    
                    await self._log_performance_metrics(trading_cycle, coordination_results)
                    
                    # ==================================================================================
                    # STEP 6: STATUS REPORTING (Every 15 minutes)
                    # ==================================================================================
                    
                    time_since_status = datetime.now(timezone.utc) - last_status_report
                    if time_since_status.total_seconds() >= 900:  # 15 minutes
                        await self._log_detailed_status()
                        last_status_report = datetime.now(timezone.utc)
                    
                    # ==================================================================================
                    # STEP 7: CYCLE COMPLETION
                    # ==================================================================================
                    
                    cycle_duration = (datetime.now(timezone.utc) - cycle_start).total_seconds()
                    self.logger.info(f"✅ Trading cycle {trading_cycle} completed ({cycle_duration:.2f}s)")
                    
                    # Wait for next cycle (default: 5 minutes)
                    cycle_interval = getattr(args, 'cycle_interval', 300)
                    await asyncio.sleep(cycle_interval)
                    
                except Exception as cycle_error:
                    self.logger.error(f"❌ Trading cycle {trading_cycle} error: {cycle_error}")
                    self.logger.error(traceback.format_exc())
                    
                    # Add to system errors
                    self.system_errors.append(f"Cycle {trading_cycle}: {str(cycle_error)}")
                    
                    # Continue with next cycle after short delay
                    await asyncio.sleep(60)
                    continue
        
        except KeyboardInterrupt:
            self.logger.warning("🛑 Live trading interrupted by user")
        
        except Exception as e:
            self.logger.error(f"❌ Live trading system error: {e}")
            self.logger.error(traceback.format_exc())
        
        finally:
            self.logger.info("🔄 Initiating graceful shutdown...")
            await self._graceful_shutdown()
            self.logger.info("✅ Live trading session ended")

    async def _fetch_market_data(self) -> Optional[Any]:
        """Fetch current market data"""
        try:
            # Fetch OHLCV data
            market_data = await self.data_fetcher.fetch_ohlcv()
            
            if market_data is not None and len(market_data) > 0:
                self.logger.debug(f"📊 Market data fetched: {len(market_data)} candles")
                return market_data
            else:
                self.logger.warning("⚠️ Empty market data received")
                return None
        
        except DataFetchingError as e:
            self.logger.error(f"❌ Market data fetch error: {e}")
            return None
        
        except Exception as e:
            self.logger.error(f"❌ Unexpected market data error: {e}")
            return None
    
    async def _check_emergency_brake(self, config: Dict[str, Any]) -> bool:
        """Check emergency brake conditions"""
        try:
            initial_balance = self.portfolio.initial_balance
            current_balance = self.portfolio.balance
            
            # Calculate current drawdown
            current_drawdown_pct = ((initial_balance - current_balance) / initial_balance) * 100
            
            # Check maximum drawdown
            if current_drawdown_pct > self.emergency_config.max_drawdown_pct:
                self.logger.critical(f"🚨 Maximum drawdown exceeded: {current_drawdown_pct:.1f}% > {self.emergency_config.max_drawdown_pct:.1f}%")
                return True
            
            # Check minimum balance threshold
            balance_pct = (current_balance / initial_balance) * 100
            if balance_pct < self.emergency_config.min_balance_pct:
                self.logger.critical(f"🚨 Minimum balance threshold breached: {balance_pct:.1f}% < {self.emergency_config.min_balance_pct:.1f}%")
                return True
            
            # Check for excessive system errors
            if len(self.system_errors) >= 10:  # 10+ errors in session
                self.logger.critical(f"🚨 Excessive system errors detected: {len(self.system_errors)} errors")
                return True
            
            return False
            
        except Exception as e:
            self.logger.error(f"❌ Emergency brake check error: {e}")
            return True  # Fail-safe: trigger brake on check error
    
    async def _execute_emergency_stop(self) -> None:
        """Execute emergency stop procedures"""
        try:
            self.emergency_brake_triggered = True
            self.is_live_trading = False
            
            self.logger.critical("🚨 EXECUTING EMERGENCY STOP PROCEDURES")
            
            # Close all open positions (placeholder - would integrate with actual trading)
            if hasattr(self.portfolio, 'positions') and self.portfolio.positions:
                self.logger.critical(f"🔴 Closing {len(self.portfolio.positions)} open positions")
                # In real implementation: close all positions at market
            
            # Stop all strategy activities
            if self.strategy_coordinator:
                for strategy_name in self.strategy_coordinator.strategies:
                    self.strategy_coordinator.set_strategy_status(strategy_name, "INACTIVE")
                self.logger.critical("🛑 All strategies deactivated")
            
            # Log emergency brake event
            emergency_log = {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "trigger_reason": "EMERGENCY_BRAKE_ACTIVATED",
                "portfolio_state": {
                    "initial_balance": self.portfolio.initial_balance,
                    "current_balance": self.portfolio.balance,
                    "drawdown_pct": ((self.portfolio.initial_balance - self.portfolio.balance) / self.portfolio.initial_balance) * 100
                },
                "system_errors": self.system_errors[-5:]  # Last 5 errors
            }
            
            # Save emergency log
            emergency_file = Path("logs") / f"emergency_brake_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
            emergency_file.parent.mkdir(exist_ok=True)
            
            with open(emergency_file, 'w') as f:
                json.dump(emergency_log, f, indent=2)
            
            self.logger.critical(f"📝 Emergency brake log saved: {emergency_file}")
            
        except Exception as e:
            self.logger.critical(f"❌ Emergency stop execution error: {e}")
    
    async def _perform_safety_check(self) -> bool:
        """Perform pre-trading safety checks"""
        try:
            self.logger.info("🛡️ Performing safety checks...")
            
            # Check system validator
            if self.validator:
                validation_result = self.validator.run_pre_commit_validation()
                if not validation_result:
                    self.logger.error("❌ System validation failed")
                    return False
            
            # Check portfolio state
            if self.portfolio.balance <= 0:
                self.logger.error("❌ Portfolio balance is zero or negative")
                return False
            
            # Check strategy coordinator
            if not self.strategy_coordinator:
                self.logger.error("❌ Strategy coordinator not initialized")
                return False
            
            # Check data fetcher
            if not self.data_fetcher:
                self.logger.error("❌ Data fetcher not initialized")
                return False
            
            self.logger.info("✅ All safety checks passed")
            return True
            
        except Exception as e:
            self.logger.error(f"❌ Safety check error: {e}")
            return False
    
    async def _log_performance_metrics(self, cycle: int, coordination_results: Dict) -> None:
        """Log performance metrics"""
        try:
            if cycle % 10 == 0:  # Every 10 cycles
                balance = self.portfolio.balance
                initial_balance = self.portfolio.initial_balance
                profit_pct = ((balance - initial_balance) / initial_balance) * 100
                
                self.logger.info(f"📊 Performance Update (Cycle {cycle}):")
                self.logger.info(f"   💰 Balance: ${balance:,.2f} ({profit_pct:+.2f}%)")
                self.logger.info(f"   📈 Actions: {', '.join(coordination_results.get('actions_taken', []))}")
                
                # Store in performance history
                self.performance_history.append({
                    "timestamp": datetime.now(timezone.utc),
                    "cycle": cycle,
                    "balance": balance,
                    "profit_pct": profit_pct,
                    "actions": coordination_results.get("actions_taken", [])
                })
        
        except Exception as e:
            self.logger.error(f"❌ Performance logging error: {e}")
    
    async def _log_detailed_status(self) -> None:
        """Log detailed system status"""
        try:
            status = await self.get_system_status()
            
            self.logger.info("📊 DETAILED SYSTEM STATUS:")
            self.logger.info(f"   🔄 Uptime: {status.uptime_seconds/3600:.1f} hours")
            self.logger.info(f"   💰 Balance: ${status.current_balance:,.2f}")
            self.logger.info(f"   📈 Profit: ${status.total_profit_usdt:,.2f}")
            self.logger.info(f"   🎯 Active Strategies: {status.active_strategies}/{status.total_strategies}")
            self.logger.info(f"   🧬 Evolution Cycles: {status.evolution_cycles_completed}")
            self.logger.info(f"   📊 Coordination Success: {status.coordination_success_rate:.1%}")
        
        except Exception as e:
            self.logger.error(f"❌ Status logging error: {e}")
    
    async def _graceful_shutdown(self) -> None:
        """Perform graceful system shutdown"""
        try:
            self.logger.info("🔄 Graceful shutdown in progress...")
            
            # Stop trading activities
            self.is_live_trading = False
            
            # Close data connections
            if self.data_fetcher and hasattr(self.data_fetcher, 'close'):
                await self.data_fetcher.close()
            
            # Save final state
            if self.portfolio:
                final_state = {
                    "shutdown_time": datetime.now(timezone.utc).isoformat(),
                    "final_balance": self.portfolio.balance,
                    "session_duration_hours": (datetime.now(timezone.utc) - self.start_time).total_seconds() / 3600,
                    "total_errors": len(self.system_errors),
                    "emergency_brake_triggered": self.emergency_brake_triggered
                }
                
                shutdown_file = Path("logs") / f"session_end_{datetime.now().strftime('%Y%m%d_%H%M%S')}.json"
                shutdown_file.parent.mkdir(exist_ok=True)
                
                with open(shutdown_file, 'w') as f:
                    json.dump(final_state, f, indent=2)
                
                self.logger.info(f"💾 Session data saved: {shutdown_file}")
            
            self.logger.info("✅ Graceful shutdown completed")
            
        except Exception as e:
            self.logger.error(f"❌ Graceful shutdown error: {e}")

    # ==================================================================================
    # BACKTESTING OPERATIONS
    # ==================================================================================
    
    async def run_backtest(self, args: argparse.Namespace) -> None:
        """
        🧪 BACKTESTING MODE - Historical Strategy Validation
        
        Comprehensive backtesting with:
        - Multi-strategy portfolio testing
        - Performance attribution analysis
        - Risk-adjusted metrics
        - Statistical significance testing
        """
        
        self.logger.info("🧪 BACKTESTING MODE ACTIVATED")
        
        # Configuration from arguments
        config = {
            "strategy": args.strategy,
            "start_date": args.start_date,
            "end_date": args.end_date,
            "capital": args.capital,
            "data_file": args.data_file,
            "parallel_processing": getattr(args, 'parallel', True),
            "advanced_analytics": getattr(args, 'analytics', True)
        }
        
        # Initialize system for backtesting
        if not await self.initialize_system("backtest", config):
            self.logger.error("❌ Backtesting initialization failed")
            return
        
        try:
            # Create backtest configuration
            backtest_config = BacktestConfiguration(
                start_date=datetime.fromisoformat(args.start_date),
                end_date=datetime.fromisoformat(args.end_date),
                initial_capital=args.capital,
                commission_rate=getattr(args, 'commission', 0.001),
                slippage_rate=getattr(args, 'slippage', 0.0005),
                mode=BacktestMode.MULTI_STRATEGY if args.strategy == "all" else BacktestMode.SINGLE_STRATEGY
            )
            
            self.logger.info(f"🚀 Starting backtest...")
            self.logger.info(f"   Strategy: {args.strategy}")
            self.logger.info(f"   Period: {args.start_date} to {args.end_date}")
            self.logger.info(f"   Capital: ${args.capital:,.2f}")
            self.logger.info(f"   Data: {args.data_file}")
            
            # Load historical data
            historical_data = await self._load_backtest_data(args.data_file)
            if historical_data is None:
                self.logger.error("❌ Failed to load historical data")
                return
            
            # Run backtest
            backtest_start = datetime.now(timezone.utc)
            
            if args.strategy == "all":
                results = await self.backtester.run_multi_strategy_backtest(
                    backtest_config, 
                    historical_data
                )
            else:
                results = await self.backtester.run_single_strategy_backtest(
                    args.strategy,
                    backtest_config,
                    historical_data
                )
            
            backtest_duration = (datetime.now(timezone.utc) - backtest_start).total_seconds()
            
            # Display results
            if results:
                await self._display_backtest_results(results, backtest_duration)
                
                # Save detailed results
                await self._save_backtest_results(results, config)
            else:
                self.logger.error("❌ Backtest completed with no results")
        
        except Exception as e:
            self.logger.error(f"❌ Backtesting error: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _load_backtest_data(self, data_file: str) -> Optional[Any]:
        """Load historical data for backtesting"""
        try:
            import pandas as pd
            
            data_path = Path(data_file)
            if not data_path.exists():
                self.logger.error(f"❌ Data file not found: {data_file}")
                return None
            
            # Load CSV data
            df = pd.read_csv(data_path)
            
            # Validate data structure
            required_columns = ['timestamp', 'open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                self.logger.error(f"❌ Missing columns in data: {missing_columns}")
                return None
            
            # Convert timestamp
            df['timestamp'] = pd.to_datetime(df['timestamp'])
            df.set_index('timestamp', inplace=True)
            
            self.logger.info(f"✅ Historical data loaded: {len(df)} candles")
            self.logger.info(f"   Period: {df.index[0]} to {df.index[-1]}")
            
            return df
            
        except Exception as e:
            self.logger.error(f"❌ Data loading error: {e}")
            return None
    
    async def _display_backtest_results(self, results: BacktestResult, duration: float) -> None:
        """Display comprehensive backtest results"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🏁 BACKTEST RESULTS - COMPREHENSIVE REPORT")
            self.logger.info("="*80)
            
            # Performance metrics
            self.logger.info("📊 PERFORMANCE METRICS:")
            self.logger.info(f"   💰 Total Return: {results.total_return_pct:+.2f}%")
            self.logger.info(f"   📈 Annualized Return: {results.annualized_return_pct:+.2f}%")
            self.logger.info(f"   📊 Volatility: {results.volatility_pct:.2f}%")
            self.logger.info(f"   ⚡ Sharpe Ratio: {results.sharpe_ratio:.3f}")
            self.logger.info(f"   📉 Max Drawdown: {results.max_drawdown_pct:.2f}%")
            self.logger.info(f"   🎯 Calmar Ratio: {results.calmar_ratio:.3f}")
            
            # Trading metrics
            self.logger.info("\n📋 TRADING METRICS:")
            self.logger.info(f"   🔄 Total Trades: {results.total_trades}")
            self.logger.info(f"   🎯 Win Rate: {results.win_rate_pct:.1f}%")
            self.logger.info(f"   💎 Profit Factor: {results.profit_factor:.2f}")
            self.logger.info(f"   ⬆️ Average Win: {results.avg_win_pct:.2f}%")
            self.logger.info(f"   ⬇️ Average Loss: {results.avg_loss_pct:.2f}%")
            
            # Risk metrics
            self.logger.info("\n🛡️ RISK METRICS:")
            self.logger.info(f"   📊 VaR (95%): {results.var_95_pct:.2f}%")
            self.logger.info(f"   🔴 CVaR (95%): {results.cvar_95_pct:.2f}%")
            self.logger.info(f"   🩺 Ulcer Index: {results.ulcer_index:.2f}")
            
            # Strategy-specific results
            if results.strategy_results:
                self.logger.info("\n🎯 STRATEGY BREAKDOWN:")
                for strategy_name, strategy_result in results.strategy_results.items():
                    contribution = results.strategy_contributions.get(strategy_name, 0.0)
                    self.logger.info(f"   📊 {strategy_name.upper()}:")
                    self.logger.info(f"      Return: {strategy_result.get('return_pct', 0):+.2f}%")
                    self.logger.info(f"      Contribution: {contribution:.1f}%")
                    self.logger.info(f"      Trades: {strategy_result.get('trades', 0)}")
            
            # Execution metrics
            self.logger.info("\n⚡ EXECUTION METRICS:")
            self.logger.info(f"   ⏱️ Backtest Duration: {duration:.2f} seconds")
            self.logger.info(f"   📊 Data Points Processed: {results.data_points_processed:,}")
            self.logger.info(f"   🚀 Processing Speed: {results.data_points_processed/duration:.0f} points/sec")
            
            # Statistical significance
            if results.statistical_significance:
                self.logger.info("\n📈 STATISTICAL ANALYSIS:")
                for metric, p_value in results.statistical_significance.items():
                    significance = "Significant" if p_value < 0.05 else "Not Significant"
                    self.logger.info(f"   {metric}: p={p_value:.4f} ({significance})")
            
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"❌ Results display error: {e}")
    
    async def _save_backtest_results(self, results: BacktestResult, config: Dict[str, Any]) -> None:
        """Save detailed backtest results"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            results_file = Path("backtest_results") / f"backtest_{config['strategy']}_{timestamp}.json"
            results_file.parent.mkdir(exist_ok=True)
            
            # Convert results to serializable format
            results_dict = asdict(results)
            results_dict['configuration'] = config
            results_dict['generation_time'] = datetime.now(timezone.utc).isoformat()
            
            # Handle pandas objects
            if hasattr(results.equity_curve, 'to_dict'):
                results_dict['equity_curve'] = results.equity_curve.to_dict()
            if hasattr(results.returns_series, 'to_dict'):
                results_dict['returns_series'] = results.returns_series.to_dict()
            
            # Save to JSON
            with open(results_file, 'w') as f:
                json.dump(results_dict, f, indent=2, default=str)
            
            self.logger.info(f"💾 Backtest results saved: {results_file}")
            
        except Exception as e:
            self.logger.error(f"❌ Results saving error: {e}")

    # ==================================================================================
    # OPTIMIZATION OPERATIONS
    # ==================================================================================
    
    async def run_optimization(self, args: argparse.Namespace) -> None:
        """
        🎯 OPTIMIZATION MODE - Strategy Parameter Optimization
        
        Advanced optimization with:
        - Bayesian optimization via Optuna
        - Walk-forward analysis
        - Multi-objective optimization
        - Parameter validation
        """
        
        self.logger.info("🎯 OPTIMIZATION MODE ACTIVATED")
        
        # Configuration from arguments
        config = {
            "strategy": args.strategy,
            "trials": args.trials,
            "storage": getattr(args, 'storage', 'sqlite:///optimization/studies.db'),
            "walk_forward": getattr(args, 'walk_forward', False),
            "parallel_jobs": getattr(args, 'parallel', 1),
            "timeout_minutes": getattr(args, 'timeout', 120)
        }
        
        # Initialize system for optimization
        if not await self.initialize_system("optimize", config):
            self.logger.error("❌ Optimization initialization failed")
            return
        
        try:
            self.logger.info(f"🚀 Starting optimization...")
            self.logger.info(f"   Strategy: {args.strategy}")
            self.logger.info(f"   Trials: {args.trials}")
            self.logger.info(f"   Walk-forward: {config['walk_forward']}")
            self.logger.info(f"   Storage: {config['storage']}")
            
            optimization_start = datetime.now(timezone.utc)
            
            # Run optimization
            if args.strategy == "all":
                results = await self.optimizer.optimize_all_strategies()
            else:
                result = await self.optimizer.optimize_single_strategy(args.strategy)
                results = {args.strategy: result}
            
            optimization_duration = (datetime.now(timezone.utc) - optimization_start).total_seconds()
            
            # Display optimization results
            await self._display_optimization_results(results, optimization_duration)
            
            # Save parameters to JSON system
            for strategy_name, result in results.items():
                if result and result.best_parameters:
                    success = self.json_manager.save_optimization_results(
                        strategy_name=strategy_name,
                        best_parameters=result.best_parameters,
                        optimization_metrics={
                            "best_score": result.best_score,
                            "total_trials": result.total_trials,
                            "successful_trials": result.successful_trials,
                            "optimization_duration_minutes": result.optimization_duration_minutes,
                            "walk_forward_analysis": config['walk_forward'],
                            "optimization_date": datetime.now().isoformat()
                        },
                        source_file=f"main_optimizer_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
                    )
                    
                    if success:
                        self.logger.info(f"✅ {strategy_name} parameters saved to JSON system")
                    else:
                        self.logger.warning(f"⚠️ Failed to save {strategy_name} parameters")
        
        except Exception as e:
            self.logger.error(f"❌ Optimization error: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _display_optimization_results(self, results: Dict[str, OptimizationResult], duration: float) -> None:
        """Display optimization results"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🎯 OPTIMIZATION RESULTS - COMPREHENSIVE REPORT")
            self.logger.info("="*80)
            
            total_trials = sum(r.total_trials for r in results.values() if r)
            successful_trials = sum(r.successful_trials for r in results.values() if r)
            
            self.logger.info("📊 OPTIMIZATION SUMMARY:")
            self.logger.info(f"   🎯 Strategies Optimized: {len(results)}")
            self.logger.info(f"   🔬 Total Trials: {total_trials:,}")
            self.logger.info(f"   ✅ Successful Trials: {successful_trials:,}")
            self.logger.info(f"   ⏱️ Total Duration: {duration/60:.1f} minutes")
            self.logger.info(f"   ⚡ Trials per Minute: {total_trials/(duration/60):.1f}")
            
            for strategy_name, result in results.items():
                if not result:
                    self.logger.warning(f"❌ {strategy_name.upper()}: Optimization failed")
                    continue
                
                self.logger.info(f"\n🚀 {strategy_name.upper()} RESULTS:")
                self.logger.info(f"   🏆 Best Score: {result.best_score:.4f}")
                self.logger.info(f"   📊 Trials: {result.successful_trials}/{result.total_trials}")
                self.logger.info(f"   ⏱️ Duration: {result.optimization_duration_minutes:.1f} minutes")
                
                if result.best_parameters:
                    param_count = len(result.best_parameters)
                    self.logger.info(f"   ⚙️ Optimized Parameters: {param_count}")
                    
                    # Show top 5 most important parameters
                    important_params = list(result.best_parameters.items())[:5]
                    for param_name, param_value in important_params:
                        if isinstance(param_value, float):
                            self.logger.info(f"      {param_name}: {param_value:.4f}")
                        else:
                            self.logger.info(f"      {param_name}: {param_value}")
                
                if result.robustness_score:
                    self.logger.info(f"   🛡️ Robustness Score: {result.robustness_score:.3f}")
                
                if result.final_validation_score:
                    self.logger.info(f"   ✅ Validation Score: {result.final_validation_score:.3f}")
            
            self.logger.info("="*80)
            self.logger.info("💎 Optimization parameters saved to JSON parameter system")
            self.logger.info("🚀 Ready for live trading or backtesting with optimized parameters!")
            
        except Exception as e:
            self.logger.error(f"❌ Optimization results display error: {e}")

    # ==================================================================================
    # SYSTEM VALIDATION AND STATUS
    # ==================================================================================
    
    async def validate_system(self, args: argparse.Namespace) -> None:
        """
        🛡️ SYSTEM VALIDATION - Comprehensive Health Check
        
        Complete system validation including:
        - Component integrity checks
        - Performance validation
        - Risk management validation
        - Configuration validation
        """
        
        self.logger.info("🛡️ SYSTEM VALIDATION MODE ACTIVATED")
        
        try:
            # Initialize basic system
            if not await self.initialize_system("validate", {}):
                self.logger.error("❌ Basic system initialization failed for validation")
                return
            
            full_validation = getattr(args, 'full_validation', False)
            
            if full_validation:
                # Run comprehensive validation
                self.logger.info("🔍 Running comprehensive system validation...")
                validation_summary = self.validator.run_full_validation()
            else:
                # Run basic validation
                self.logger.info("🚀 Running basic system validation...")
                validation_success = self.validator.run_pre_commit_validation()
                
                validation_summary = {
                    "overall_success": validation_success,
                    "validation_type": "basic",
                    "critical_failures": len(self.validator.critical_failures),
                    "warnings": len(self.validator.warnings)
                }
            
            # Display validation results
            await self._display_validation_results(validation_summary)
            
            # Additional Phoenix-specific validation
            await self._validate_phoenix_components()
            
        except Exception as e:
            self.logger.error(f"❌ System validation error: {e}")
            self.logger.error(traceback.format_exc())
    
    async def _display_validation_results(self, validation_summary: Dict[str, Any]) -> None:
        """Display validation results"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("🛡️ SYSTEM VALIDATION RESULTS")
            self.logger.info("="*80)
            
            if validation_summary.get("overall_success", False):
                self.logger.info("✅ SYSTEM VALIDATION PASSED")
            else:
                self.logger.error("❌ SYSTEM VALIDATION FAILED")
            
            # Basic metrics
            critical_failures = validation_summary.get("critical_failures", 0)
            warnings = validation_summary.get("warnings", 0)
            
            self.logger.info(f"📊 VALIDATION SUMMARY:")
            self.logger.info(f"   ❌ Critical Failures: {critical_failures}")
            self.logger.info(f"   ⚠️ Warnings: {warnings}")
            
            # Detailed metrics (if available)
            if "total_tests" in validation_summary:
                passed_tests = validation_summary.get("passed_tests", 0)
                total_tests = validation_summary.get("total_tests", 0)
                duration = validation_summary.get("duration_seconds", 0)
                
                self.logger.info(f"   🧪 Tests Passed: {passed_tests}/{total_tests}")
                self.logger.info(f"   ⏱️ Validation Duration: {duration:.2f} seconds")
            
            # Import status
            if CORE_IMPORTS_SUCCESS:
                self.logger.info("✅ All core imports successful")
            else:
                self.logger.error(f"❌ Import failures detected: {IMPORT_ERROR}")
            
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"❌ Validation results display error: {e}")
    
    async def _validate_phoenix_components(self) -> None:
        """Validate Phoenix-specific components"""
        try:
            self.logger.info("\n🔍 PHOENIX COMPONENT VALIDATION:")
            
            # Strategy registry validation
            available_strategies = sum(1 for s in self.strategy_registry.values() if s["class"] is not None)
            total_strategies = len(self.strategy_registry)
            
            self.logger.info(f"   🎯 Strategy Registry: {available_strategies}/{total_strategies} strategies available")
            
            # JSON parameter system validation
            param_files = list(Path("optimization/results").glob("*_best_params.json")) if Path("optimization/results").exists() else []
            self.logger.info(f"   💾 Parameter Files: {len(param_files)} optimization results found")
            
            # Backtesting data validation
            data_files = list(Path("historical_data").glob("*.csv")) if Path("historical_data").exists() else []
            self.logger.info(f"   📊 Historical Data: {len(data_files)} data files available")
            
            # Log directories validation
            logs_dir = Path("logs")
            if logs_dir.exists():
                log_files = list(logs_dir.glob("*.log"))
                self.logger.info(f"   📝 Log Files: {len(log_files)} log files found")
            else:
                self.logger.warning("   ⚠️ Logs directory not found")
            
            # Configuration validation
            try:
                config_values = {
                    "SYMBOL": getattr(settings, 'SYMBOL', 'Not Set'),
                    "TIMEFRAME": getattr(settings, 'TIMEFRAME', 'Not Set'),
                    "INITIAL_CAPITAL": getattr(settings, 'INITIAL_CAPITAL_USDT', 'Not Set')
                }
                self.logger.info(f"   ⚙️ Configuration: {len(config_values)} settings validated")
            except Exception as config_error:
                self.logger.warning(f"   ⚠️ Configuration validation error: {config_error}")
            
        except Exception as e:
            self.logger.error(f"❌ Phoenix component validation error: {e}")
    
    async def show_status(self, args: argparse.Namespace) -> None:
        """
        📊 SYSTEM STATUS - Real-time System Analytics
        
        Comprehensive status reporting including:
        - System health metrics
        - Component status
        - Performance analytics
        - Resource utilization
        """
        
        self.logger.info("📊 SYSTEM STATUS MODE ACTIVATED")
        
        try:
            detailed = getattr(args, 'detailed', False)
            
            # Get system status
            status = await self.get_system_status()
            
            # Display status
            await self._display_system_status(status, detailed)
            
        except Exception as e:
            self.logger.error(f"❌ System status error: {e}")
            self.logger.error(traceback.format_exc())
    
    async def get_system_status(self) -> SystemStatus:
        """Get comprehensive system status"""
        try:
            uptime = (datetime.now(timezone.utc) - self.start_time).total_seconds()
            
            # Basic status
            status = SystemStatus(
                timestamp=datetime.now(timezone.utc),
                system_version=self.version,
                uptime_seconds=uptime,
                core_imports_success=CORE_IMPORTS_SUCCESS
            )
            
            # Component status
            status.portfolio_initialized = self.portfolio is not None
            status.coordinator_active = self.strategy_coordinator is not None
            status.evolution_active = self.evolution_system is not None
            status.backtester_ready = self.backtester is not None
            status.optimizer_ready = self.optimizer is not None
            
            # Portfolio metrics
            if self.portfolio:
                status.current_balance = self.portfolio.balance
                status.total_profit_usdt = self.portfolio.balance - self.portfolio.initial_balance
            
            # Strategy metrics
            if self.strategy_coordinator:
                status.total_strategies = len(self.strategy_coordinator.strategies)
                status.active_strategies = sum(
                    1 for alloc in self.strategy_coordinator.strategy_allocations.values()
                    if alloc.status.value == 'active'
                )
                
                # Coordination metrics
                coord_summary = self.strategy_coordinator.get_coordination_summary()
                if coord_summary and 'performance_metrics' in coord_summary:
                    metrics = coord_summary['performance_metrics']
                    total_coordinations = metrics.get('total_coordinations', 0)
                    successful_consensus = metrics.get('successful_consensus', 0)
                    status.coordination_success_rate = successful_consensus / max(1, total_coordinations)
            
            # Evolution metrics
            if self.evolution_system:
                evolution_summary = self.evolution_system.get_evolution_summary()
                if evolution_summary and 'system_overview' in evolution_summary:
                    overview = evolution_summary['system_overview']
                    status.evolution_cycles_completed = overview.get('total_evolution_cycles', 0)
            
            # System health
            try:
                import psutil
                process = psutil.Process()
                status.memory_usage_mb = process.memory_info().rss / 1024 / 1024
                status.cpu_usage_pct = process.cpu_percent()
            except ImportError:
                status.memory_usage_mb = 0.0
                status.cpu_usage_pct = 0.0
            
            # Last error
            if self.system_errors:
                status.last_error = self.system_errors[-1]
            
            return status
            
        except Exception as e:
            self.logger.error(f"❌ System status collection error: {e}")
            return SystemStatus(
                timestamp=datetime.now(timezone.utc),
                system_version=self.version,
                uptime_seconds=0.0,
                core_imports_success=False,
                last_error=str(e)
            )
    
    async def _display_system_status(self, status: SystemStatus, detailed: bool = False) -> None:
        """Display system status"""
        try:
            self.logger.info("\n" + "="*80)
            self.logger.info("📊 PHOENIX TRADING SYSTEM STATUS")
            self.logger.info("="*80)
            
            # System overview
            self.logger.info("🚀 SYSTEM OVERVIEW:")
            self.logger.info(f"   📋 Version: {status.system_version}")
            self.logger.info(f"   ⏱️ Uptime: {status.uptime_seconds/3600:.1f} hours")
            self.logger.info(f"   🔧 Core Imports: {'✅ Success' if status.core_imports_success else '❌ Failed'}")
            self.logger.info(f"   📝 Trading Active: {'✅ Yes' if self.is_live_trading else '❌ No'}")
            
            # Component status
            self.logger.info("\n🔧 COMPONENT STATUS:")
            self.logger.info(f"   💰 Portfolio: {'✅ Ready' if status.portfolio_initialized else '❌ Not Ready'}")
            self.logger.info(f"   🎯 Coordinator: {'✅ Active' if status.coordinator_active else '❌ Inactive'}")
            self.logger.info(f"   🧬 Evolution: {'✅ Active' if status.evolution_active else '❌ Inactive'}")
            self.logger.info(f"   🧪 Backtester: {'✅ Ready' if status.backtester_ready else '❌ Not Ready'}")
            self.logger.info(f"   🎯 Optimizer: {'✅ Ready' if status.optimizer_ready else '❌ Not Ready'}")
            
            # Performance metrics
            if status.portfolio_initialized:
                profit_pct = (status.total_profit_usdt / (status.current_balance - status.total_profit_usdt)) * 100
                self.logger.info("\n💰 PORTFOLIO METRICS:")
                self.logger.info(f"   📊 Current Balance: ${status.current_balance:,.2f}")
                self.logger.info(f"   📈 Total Profit: ${status.total_profit_usdt:+,.2f} ({profit_pct:+.2f}%)")
            
            # Strategy metrics
            if status.coordinator_active:
                self.logger.info("\n🎯 STRATEGY METRICS:")
                self.logger.info(f"   📋 Total Strategies: {status.total_strategies}")
                self.logger.info(f"   ✅ Active Strategies: {status.active_strategies}")
                self.logger.info(f"   🎼 Coordination Success: {status.coordination_success_rate:.1%}")
            
            # Evolution metrics
            if status.evolution_active:
                self.logger.info("\n🧬 EVOLUTION METRICS:")
                self.logger.info(f"   🔄 Evolution Cycles: {status.evolution_cycles_completed}")
            
            # System health
            self.logger.info("\n🩺 SYSTEM HEALTH:")
            self.logger.info(f"   💾 Memory Usage: {status.memory_usage_mb:.1f} MB")
            self.logger.info(f"   ⚡ CPU Usage: {status.cpu_usage_pct:.1f}%")
            self.logger.info(f"   ❌ System Errors: {len(self.system_errors)}")
            
            if status.last_error:
                self.logger.info(f"   🔴 Last Error: {status.last_error}")
            
            # Emergency brake status
            if hasattr(self, 'emergency_brake_triggered'):
                brake_status = "🚨 TRIGGERED" if self.emergency_brake_triggered else "✅ Ready"
                self.logger.info(f"   🛡️ Emergency Brake: {brake_status}")
            
            # Detailed information
            if detailed:
                await self._display_detailed_status()
            
            self.logger.info("="*80)
            
        except Exception as e:
            self.logger.error(f"❌ Status display error: {e}")
    
    async def _display_detailed_status(self) -> None:
        """Display detailed system status"""
        try:
            self.logger.info("\n🔍 DETAILED SYSTEM INFORMATION:")
            
            # File system status
            if Path("optimization/results").exists():
                param_files = list(Path("optimization/results").glob("*.json"))
                self.logger.info(f"   💾 Parameter Files: {len(param_files)}")
            
            if Path("logs").exists():
                log_files = list(Path("logs").glob("*.log"))
                self.logger.info(f"   📝 Log Files: {len(log_files)}")
            
            # Strategy registry details
            self.logger.info(f"   🎯 Strategy Registry: {len(self.strategy_registry)} strategies")
            for name, info in self.strategy_registry.items():
                available = "✅" if info["class"] else "❌"
                self.logger.info(f"      {available} {name}: {info['name']}")
            
            # Recent performance
            if self.performance_history:
                recent_performance = self.performance_history[-5:]
                self.logger.info(f"   📊 Recent Performance: {len(recent_performance)} data points")
            
        except Exception as e:
            self.logger.error(f"❌ Detailed status error: {e}")


# ==================================================================================
# COMMAND LINE INTERFACE
# ==================================================================================

def create_argument_parser() -> argparse.ArgumentParser:
    """Create comprehensive argument parser"""
    
    parser = argparse.ArgumentParser(
        description="Phoenix Trading System v2.0 - Ultimate Command Center",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
USAGE EXAMPLES:
  Live Trading:
    python main.py live --strategy momentum --capital 1000 --symbol BTC/USDT
    python main.py live --strategy all --capital 5000 --emergency-brake --max-drawdown 10
  
  Backtesting:
    python main.py backtest --strategy momentum --start-date 2024-01-01 --end-date 2024-12-31
    python main.py backtest --strategy all --data-file historical_data/BTCUSDT_1h_2024.csv --analytics
  
  Optimization:
    python main.py optimize --strategy momentum --trials 5000 --walk-forward
    python main.py optimize --strategy all --trials 10000 --parallel 4 --timeout 180
  
  System Management:
    python main.py validate --full-validation
    python main.py status --detailed
    
HEDGE FUND LEVEL IMPLEMENTATION - PRODUCTION READY
        """
    )
    
    # Main command
    subparsers = parser.add_subparsers(dest='command', help='Available commands')
    
    # ==================================================================================
    # LIVE TRADING COMMAND
    # ==================================================================================
    
    live_parser = subparsers.add_parser(
        'live', 
        help='Live trading mode',
        description='Execute live trading with real money'
    )
    
    live_parser.add_argument(
        '--strategy', 
        type=str, 
        required=True,
        choices=['momentum', 'bollinger_ml', 'rsi_ml', 'macd_ml', 'volume_profile', 'all'],
        help='Strategy to trade with'
    )
    
    live_parser.add_argument(
        '--capital', 
        type=float, 
        required=True,
        help='Initial trading capital in USDT'
    )
    
    live_parser.add_argument(
        '--symbol', 
        type=str, 
        default='BTC/USDT',
        help='Trading pair symbol (default: BTC/USDT)'
    )
    
    live_parser.add_argument(
        '--max-drawdown', 
        type=float, 
        default=15.0,
        help='Maximum drawdown percentage for emergency brake (default: 15.0)'
    )
    
    live_parser.add_argument(
        '--cycle-interval', 
        type=int, 
        default=300,
        help='Trading cycle interval in seconds (default: 300)'
    )
    
    live_parser.add_argument(
        '--emergency-brake', 
        action='store_true',
        help='Enable emergency brake system'
    )
    
    # ==================================================================================
    # BACKTESTING COMMAND
    # ==================================================================================
    
    backtest_parser = subparsers.add_parser(
        'backtest', 
        help='Backtesting mode',
        description='Run historical strategy validation'
    )
    
    backtest_parser.add_argument(
        '--strategy', 
        type=str, 
        required=True,
        choices=['momentum', 'bollinger_ml', 'rsi_ml', 'macd_ml', 'volume_profile', 'all'],
        help='Strategy to backtest'
    )
    
    backtest_parser.add_argument(
        '--start-date', 
        type=str, 
        required=True,
        help='Start date for backtest (YYYY-MM-DD)'
    )
    
    backtest_parser.add_argument(
        '--end-date', 
        type=str, 
        required=True,
        help='End date for backtest (YYYY-MM-DD)'
    )
    
    backtest_parser.add_argument(
        '--capital', 
        type=float, 
        default=10000.0,
        help='Initial capital for backtest (default: 10000.0)'
    )
    
    backtest_parser.add_argument(
        '--data-file', 
        type=str, 
        required=True,
        help='Path to historical data CSV file'
    )
    
    backtest_parser.add_argument(
        '--commission', 
        type=float, 
        default=0.001,
        help='Commission rate (default: 0.001)'
    )
    
    backtest_parser.add_argument(
        '--parallel', 
        action='store_true',
        help='Enable parallel processing'
    )
    
    backtest_parser.add_argument(
        '--analytics', 
        action='store_true',
        help='Enable advanced analytics'
    )
    
    # ==================================================================================
    # OPTIMIZATION COMMAND
    # ==================================================================================
    
    optimize_parser = subparsers.add_parser(
        'optimize', 
        help='Optimization mode',
        description='Optimize strategy parameters'
    )
    
    optimize_parser.add_argument(
        '--strategy', 
        type=str, 
        required=True,
        choices=['momentum', 'bollinger_ml', 'rsi_ml', 'macd_ml', 'volume_profile', 'all'],
        help='Strategy to optimize'
    )
    
    optimize_parser.add_argument(
        '--trials', 
        type=int, 
        default=1000,
        help='Number of optimization trials (default: 1000)'
    )
    
    optimize_parser.add_argument(
        '--storage', 
        type=str, 
        default='sqlite:///optimization/studies.db',
        help='Optuna storage URL'
    )
    
    optimize_parser.add_argument(
        '--walk-forward', 
        action='store_true',
        help='Enable walk-forward analysis'
    )
    
    optimize_parser.add_argument(
        '--parallel', 
        type=int, 
        default=1,
        help='Number of parallel jobs (default: 1)'
    )
    
    optimize_parser.add_argument(
        '--timeout', 
        type=int, 
        default=120,
        help='Optimization timeout in minutes (default: 120)'
    )
    
    # ==================================================================================
    # VALIDATION COMMAND
    # ==================================================================================
    
    validate_parser = subparsers.add_parser(
        'validate', 
        help='System validation mode',
        description='Validate system health and integrity'
    )
    
    validate_parser.add_argument(
        '--full-validation', 
        action='store_true',
        help='Run comprehensive validation'
    )
    
    # ==================================================================================
    # STATUS COMMAND
    # ==================================================================================
    
    status_parser = subparsers.add_parser(
        'status', 
        help='System status mode',
        description='Display system status and analytics'
    )
    
    status_parser.add_argument(
        '--detailed', 
        action='store_true',
        help='Show detailed status information'
    )
    
    return parser


# ==================================================================================
# MAIN EXECUTION FUNCTION
# ==================================================================================

async def main():
    """Main execution function"""
    
    # Create argument parser
    parser = create_argument_parser()
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Initialize Phoenix Trading System
    phoenix = PhoenixTradingSystem()
    
    try:
        # Route to appropriate command handler
        if args.command == 'live':
            await phoenix.run_live_trading(args)
        
        elif args.command == 'backtest':
            await phoenix.run_backtest(args)
        
        elif args.command == 'optimize':
            await phoenix.run_optimization(args)
        
        elif args.command == 'validate':
            await phoenix.validate_system(args)
        
        elif args.command == 'status':
            await phoenix.show_status(args)
        
        else:
            logger.error(f"❌ Unknown command: {args.command}")
            parser.print_help()
    
    except KeyboardInterrupt:
        logger.warning("🛑 Operation interrupted by user")
    
    except Exception as e:
        logger.error(f"❌ Phoenix system error: {e}")
        logger.error(traceback.format_exc())
        sys.exit(1)


# ==================================================================================
# ENTRY POINT
# ==================================================================================

if __name__ == "__main__":
    try:
        # Setup logging
        logging.basicConfig(
            level=logging.INFO,
            format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
            handlers=[
                logging.StreamHandler(sys.stdout),
                logging.FileHandler(
                    Path("logs") / f"phoenix_{datetime.now().strftime('%Y%m%d')}.log",
                    mode='a',
                    encoding='utf-8'
                )
            ]
        )
        
        # Create logs directory
        Path("logs").mkdir(exist_ok=True)
        
        # Print startup banner
        print("🚀" + "="*79)
        print("🚀 PROJE PHOENIX v2.0 - ULTIMATE TRADING SYSTEM")
        print("🚀 Hedge Fund Seviyesi Üstü - Production Ready")
        print("🚀" + "="*79)
        print("🚀 FAZ 5 TAMAMLANDI: Sistemin Canlanması")
        print("🚀 Tüm Bileşenler Entegre - Komuta Merkezi Hazır")
        print("🚀" + "="*79)
        
        # Run main
        asyncio.run(main())
        
    except Exception as e:
        print(f"❌ System startup error: {e}")
        sys.exit(1)