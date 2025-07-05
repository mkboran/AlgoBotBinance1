#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX v2.0 - MAIN.PY IMPORT FIX
💎 FIXED: CORE_IMPORTS_SUCCESS tanımlama hatası giderildi

ÇÖZÜMLER:
1. ✅ CORE_IMPORTS_SUCCESS ve IMPORT_ERROR global scope'a taşındı
2. ✅ Import error handling geliştirildi
3. ✅ Graceful degradation eklendi
"""

import asyncio
from typing import Optional, Any, Dict
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
# GLOBAL VARIABLES - DEFINED BEFORE ANY TRY BLOCKS
# ==================================================================================

CORE_IMPORTS_SUCCESS = False
IMPORT_ERROR = None
ADVANCED_BACKTEST_AVAILABLE = False

# ==================================================================================
# CORE SYSTEM IMPORTS WITH PROPER ERROR HANDLING
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
    from strategies.bollinger_rsi_strategy import BollingerRSIStrategy
    from strategies.bollinger_ml_strategy import BollingerMLStrategy
    from strategies.rsi_ml_strategy import RSIMLStrategy
    from strategies.macd_ml_strategy import MACDMLStrategy
    from strategies.volume_profile_strategy import VolumeProfileMLStrategy
    
    # Coordination layer (FAZ 2)
    from utils.strategy_coordinator import StrategyCoordinator
    
    # Evolution system (FAZ 3)
    from utils.adaptive_parameter_evolution import AdaptiveParameterEvolution, EvolutionConfig
    
    # Backtesting system (FAZ 4)
    try:
        from backtesting.multi_strategy_backtester import (
            MultiStrategyBacktester, 
            BacktestResult, 
            BacktestConfiguration, 
            BacktestMode
        )
        ADVANCED_BACKTEST_AVAILABLE = True
    except ImportError as e:
        logger.warning(f"⚠️ Advanced backtest not available: {e}")
        ADVANCED_BACKTEST_AVAILABLE = False
        
        # Dummy classes for fallback
        class BacktestResult:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
        
        class BacktestConfiguration:
            def __init__(self, **kwargs):
                for k, v in kwargs.items():
                    setattr(self, k, v)
                # Add missing attribute with default
                if not hasattr(self, 'enable_position_sizing'):
                    self.enable_position_sizing = False
        
        class BacktestMode:
            SINGLE_STRATEGY = "single"
            MULTI_STRATEGY = "multi"
        
        class MultiStrategyBacktester:
            def __init__(self, **kwargs):
                pass
    
    # Optimization system
    from optimization.master_optimizer import MasterOptimizer, OptimizationConfig, OptimizationResult
    
    # Parameter management
    from json_parameter_system import JSONParameterManager
    
    # Validation system
    try:
        from scripts.validate_system import PhoenixSystemValidator
    except ImportError:
        logger.warning("⚠️ System validator not available")
        class PhoenixSystemValidator:
            pass
    
    CORE_IMPORTS_SUCCESS = True
    IMPORT_ERROR = None
    
except ImportError as e:
    CORE_IMPORTS_SUCCESS = False
    IMPORT_ERROR = str(e)
    logger = logging.getLogger("phoenix.main")
    logger.error(f"❌ Critical import error: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")
    
    # Create dummy classes for graceful degradation
    class Portfolio: 
        def __init__(self, **kwargs): pass
    class StrategyCoordinator: 
        def __init__(self, **kwargs): pass
    class AdaptiveParameterEvolution: 
        def __init__(self, **kwargs): pass
    class MultiStrategyBacktester: 
        def __init__(self, **kwargs): pass
    class MasterOptimizer: 
        def __init__(self, **kwargs): pass
    class PhoenixSystemValidator: 
        def __init__(self, **kwargs): pass
    class JSONParameterManager:
        def __init__(self, **kwargs): pass
    class BaseStrategy:
        def __init__(self, **kwargs): pass
    class TradingSignal:
        def __init__(self, **kwargs): pass
    class SignalType:
        BUY = "buy"
        SELL = "sell"
        HOLD = "hold"

except Exception as e:
    CORE_IMPORTS_SUCCESS = False
    IMPORT_ERROR = str(e)
    logger = logging.getLogger("phoenix.main")
    logger.error(f"❌ System import error: {e}")
    logger.error(f"Traceback: {traceback.format_exc()}")


# ==================================================================================
# CONFIGURATION AND DATA STRUCTURES
# ==================================================================================

@dataclass
class SystemStatus:
    """System durumu"""
    system_name: str = "Phoenix Trading System"
    system_version: str = "2.0"
    core_imports_success: bool = False
    import_error: Optional[str] = None
    active_mode: Optional[str] = None
    total_strategies: int = 0
    active_strategies: List[str] = None
    coordinator_active: bool = False
    evolution_active: bool = False
    backtester_active: bool = False
    optimizer_active: bool = False
    portfolio_value: float = 0.0
    total_positions: int = 0
    system_health: str = "unknown"
    uptime_seconds: float = 0.0
    last_update: str = ""

    def __post_init__(self):
        if self.active_strategies is None:
            self.active_strategies = []
        self.last_update = datetime.now(timezone.utc).isoformat()


# ==================================================================================
# PHOENIX TRADING SYSTEM - MAIN CLASS
# ==================================================================================

class PhoenixTradingSystem:
    """
    🚀 PHOENIX TRADING SYSTEM v2.0
    💎 Hedge Fund Level Automated Trading Platform
    
    Entegre edilmiş tüm modüller:
    - Strategy execution and coordination
    - Adaptive parameter evolution
    - Multi-strategy backtesting
    - Master optimization
    - System validation
    - Real-time monitoring
    """
    
    def __init__(self):
        """Initialize Phoenix Trading System"""
        self.logger = logging.getLogger("phoenix.main")
        self.start_time = datetime.now(timezone.utc)
        
        # Core components
        self.portfolio: Optional[Portfolio] = None
        self.coordinator: Optional[StrategyCoordinator] = None
        self.evolution: Optional[AdaptiveParameterEvolution] = None
        self.backtester: Optional[MultiStrategyBacktester] = None
        self.optimizer: Optional[MasterOptimizer] = None
        self.validator: Optional[PhoenixSystemValidator] = None
        self.parameter_manager: Optional[JSONParameterManager] = None
        
        # System state
        self.is_running = False
        self.current_mode = None
        self.emergency_stop = False
        
        # Strategy registry
        self.strategy_registry = self._build_strategy_registry()
        
        # Initialize parameter manager
        try:
            self.parameter_manager = JSONParameterManager()
        except Exception as e:
            self.logger.warning(f"Parameter manager initialization failed: {e}")
        
        self.logger.info("=" * 80)
        self.logger.info("🚀 PHOENIX TRADING SYSTEM v2.0 INITIALIZED")
        self.logger.info(f"📅 {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}")
        self.logger.info(f"✅ Core imports: {'SUCCESS' if CORE_IMPORTS_SUCCESS else 'FAILED'}")
        if not CORE_IMPORTS_SUCCESS:
            self.logger.error(f"❌ Import error: {IMPORT_ERROR}")
        self.logger.info("=" * 80)
    
    def _build_strategy_registry(self) -> Dict[str, Dict[str, Any]]:
        """Build strategy registry with available strategies"""
        registry = {}
        
        if CORE_IMPORTS_SUCCESS:
            registry = {
                "momentum": {
                    "class": EnhancedMomentumStrategy if 'EnhancedMomentumStrategy' in globals() else None,
                    "description": "Enhanced momentum with ML, Kelly, and dynamic exits",
                    "risk_level": "medium",
                    "recommended_capital": 1000.0
                },
                "bollinger_rsi": {
                    "class": BollingerRSIStrategy if 'BollingerRSIStrategy' in globals() else None,
                    "description": "Mean reversion with Bollinger Bands and RSI",
                    "risk_level": "low",
                    "recommended_capital": 500.0
                },
                "bollinger_ml": {
                    "class": BollingerMLStrategy if 'BollingerMLStrategy' in globals() else None,
                    "description": "Bollinger Bands with machine learning",
                    "risk_level": "medium",
                    "recommended_capital": 1000.0
                },
                "rsi_ml": {
                    "class": RSIMLStrategy if 'RSIMLStrategy' in globals() else None,
                    "description": "RSI with ML predictions",
                    "risk_level": "medium",
                    "recommended_capital": 1000.0
                },
                "macd_ml": {
                    "class": MACDMLStrategy if 'MACDMLStrategy' in globals() else None,
                    "description": "MACD with ML trend confirmation",
                    "risk_level": "medium-high",
                    "recommended_capital": 1500.0
                },
                "volume_profile": {
                    "class": VolumeProfileMLStrategy if 'VolumeProfileMLStrategy' in globals() else None,
                    "description": "Volume profile with ML analysis",
                    "risk_level": "high",
                    "recommended_capital": 2000.0
                }
            }
        
        return registry
    
    async def get_system_status(self) -> SystemStatus:
        """Get current system status"""
        status = SystemStatus(
            core_imports_success=CORE_IMPORTS_SUCCESS,
            import_error=IMPORT_ERROR,
            active_mode=self.current_mode,
            total_strategies=len(self.strategy_registry),
            coordinator_active=self.coordinator is not None,
            evolution_active=self.evolution is not None,
            backtester_active=self.backtester is not None,
            optimizer_active=self.optimizer is not None,
            uptime_seconds=(datetime.now(timezone.utc) - self.start_time).total_seconds()
        )
        
        # Portfolio status
        if self.portfolio:
            status.portfolio_value = self.portfolio.get_total_portfolio_value_usdt(50000.0)  # Dummy price
            status.total_positions = len(self.portfolio.positions)
        
        # Active strategies
        if self.coordinator:
            status.active_strategies = list(self.coordinator.strategies.keys())
        
        # System health
        if CORE_IMPORTS_SUCCESS and not self.emergency_stop:
            status.system_health = "healthy"
        elif not CORE_IMPORTS_SUCCESS:
            status.system_health = "degraded"
        else:
            status.system_health = "emergency_stop"
        
        return status
    
    # ... rest of the PhoenixTradingSystem class continues with all other methods ...
    # (I'm showing the critical fix - the rest of the class remains the same)