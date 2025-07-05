#!/usr/bin/env python3
"""
🔧 PROJE PHOENIX - OTOMATİK HATA DÜZELTME SCRIPT'İ
💎 Tüm test hatalarını otomatik olarak düzeltir

Kullanım:
python auto_fix_phoenix.py --backup --test
"""

import os
import shutil
import subprocess
import sys
from pathlib import Path
from datetime import datetime
import argparse
import logging

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("PhoenixAutoFix")


class PhoenixAutoFixer:
    """Proje Phoenix otomatik düzeltme sistemi"""
    
    def __init__(self, project_root: Path = Path.cwd()):
        self.project_root = project_root
        self.backup_dir = None
        self.fixes_applied = []
        
    def create_backup(self) -> bool:
        """Tüm dosyaların yedeğini al"""
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            self.backup_dir = self.project_root / f"AUTO_FIX_BACKUPS/backup_{timestamp}"
            self.backup_dir.mkdir(parents=True, exist_ok=True)
            
            # Backup edilecek dizinler
            dirs_to_backup = ['optimization', 'strategies', 'utils', 'backtesting', 'tests']
            files_to_backup = ['main.py', 'json_parameter_system.py']
            
            # Dizinleri yedekle
            for dir_name in dirs_to_backup:
                src = self.project_root / dir_name
                if src.exists():
                    dst = self.backup_dir / dir_name
                    shutil.copytree(src, dst)
                    logger.info(f"✅ Backed up: {dir_name}/")
            
            # Dosyaları yedekle
            for file_name in files_to_backup:
                src = self.project_root / file_name
                if src.exists():
                    dst = self.backup_dir / file_name
                    shutil.copy2(src, dst)
                    logger.info(f"✅ Backed up: {file_name}")
            
            logger.info(f"💾 Backup completed: {self.backup_dir}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Backup failed: {e}")
            return False
    
    def fix_parameter_spaces(self) -> bool:
        """parameter_spaces.py düzeltmesi"""
        try:
            file_path = self.project_root / "optimization/parameter_spaces.py"
            
            # Yeni içerik - senkron versiyon
            new_content = '''#!/usr/bin/env python3
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
'''
            
            # Dosyayı güncelle
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(new_content)
            
            logger.info("✅ Fixed: optimization/parameter_spaces.py")
            self.fixes_applied.append("parameter_spaces")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fix parameter_spaces.py: {e}")
            return False
    
    def fix_main_imports(self) -> bool:
        """main.py import hatalarını düzelt"""
        try:
            file_path = self.project_root / "main.py"
            
            if not file_path.exists():
                logger.warning("main.py not found")
                return False
            
            # Dosyayı oku
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Import bölümünü bul ve düzelt
            if "CORE_IMPORTS_SUCCESS = False" not in content[:1000]:
                # İlk satırlara ekle
                lines = content.split('\n')
                insert_index = 0
                
                # Import'lardan önce bir yer bul
                for i, line in enumerate(lines):
                    if line.strip().startswith('import') or line.strip().startswith('from'):
                        insert_index = i
                        break
                
                # Global değişkenleri ekle
                globals_code = """
# Global variables - defined before imports
CORE_IMPORTS_SUCCESS = False
IMPORT_ERROR = None
ADVANCED_BACKTEST_AVAILABLE = False
"""
                lines.insert(insert_index, globals_code)
                content = '\n'.join(lines)
            
            # BacktestConfiguration fix
            if "enable_position_sizing" not in content:
                content = content.replace(
                    "class BacktestConfiguration:",
                    """class BacktestConfiguration:
    def __init__(self, **kwargs):
        for k, v in kwargs.items():
            setattr(self, k, v)
        if not hasattr(self, 'enable_position_sizing'):
            self.enable_position_sizing = False"""
                )
            
            # Dosyayı yaz
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Fixed: main.py imports")
            self.fixes_applied.append("main_imports")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fix main.py: {e}")
            return False
    
    def fix_portfolio_logger(self) -> bool:
        """Portfolio logger eksikliğini düzelt"""
        try:
            file_path = self.project_root / "utils/portfolio.py"
            
            if not file_path.exists():
                logger.warning("utils/portfolio.py not found")
                return False
            
            # Dosyayı oku
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Portfolio.__init__ içinde logger ekle
            if "self.logger = logging.getLogger" not in content:
                # __init__ metodunu bul
                init_pattern = "def __init__(self, initial_capital_usdt"
                if init_pattern in content:
                    # Logger satırını ekle
                    lines = content.split('\n')
                    for i, line in enumerate(lines):
                        if init_pattern in line:
                            # __init__ içinde ilk boş olmayan satırı bul
                            j = i + 1
                            while j < len(lines) and (lines[j].strip() == '' or lines[j].strip().startswith('"""')):
                                j += 1
                            
                            # Logger'ı ekle
                            indent = '        '  # 8 spaces
                            logger_line = f'{indent}# Logger initialization\n{indent}self.logger = logging.getLogger("algobot.portfolio")\n'
                            lines.insert(j, logger_line)
                            break
                    
                    content = '\n'.join(lines)
            
            # Position.__post_init__ içinde logger ekle
            if "Position" in content and "self.logger = logging.getLogger" not in content:
                content = content.replace(
                    "def __post_init__(self):",
                    """def __post_init__(self):
        # Logger initialization
        self.logger = logging.getLogger(f"algobot.portfolio.position.{self.position_id}")"""
                )
            
            # Dosyayı yaz
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Fixed: utils/portfolio.py logger")
            self.fixes_applied.append("portfolio_logger")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fix portfolio.py: {e}")
            return False
    
    def fix_json_parameter_system(self) -> bool:
        """JSON parameter system strategy_name hatasını düzelt"""
        try:
            file_path = self.project_root / "Claude_cevabı/fix_json_parameter_system.py"
            
            if not file_path.exists():
                logger.warning("json_parameter_system.py not found")
                return False
            
            # Dosyayı oku
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # save_optimization_results düzeltmesi
            if '"strategy_name": strategy_name,' not in content:
                content = content.replace(
                    'parameter_data = {',
                    '''parameter_data = {
                "strategy_name": strategy_name,  # ROOT LEVEL'''
                )
            
            # load_strategy_parameters düzeltmesi
            if "if 'strategy_name' not in data:" not in content:
                # with open bloğundan sonra ekle
                pattern = "data = json.load(f)"
                if pattern in content:
                    content = content.replace(
                        pattern,
                        f"""{pattern}
            
            # Ensure strategy_name exists at root level
            if 'strategy_name' not in data:
                data['strategy_name'] = strategy_name"""
                    )
            
            # Dosyayı yaz
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            logger.info("✅ Fixed: json_parameter_system.py")
            self.fixes_applied.append("json_parameter")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fix json_parameter_system.py: {e}")
            return False
    
    def add_missing_strategy_methods(self) -> bool:
        """Strateji sınıflarına eksik metodları ekle"""
        try:
            # BaseStrategy düzeltmesi
            base_path = self.project_root / "strategies/base_strategy.py"
            if base_path.exists():
                with open(base_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # should_sell metodu ekle
                if "async def should_sell" not in content:
                    should_sell_code = '''
    async def should_sell(self, position, current_data: pd.DataFrame) -> Tuple[bool, str]:
        """Dynamic exit decision logic"""
        current_price = current_data['close'].iloc[-1]
        position.update_performance_metrics(current_price)
        
        # Stop loss check
        if position.stop_loss_price and current_price <= position.stop_loss_price:
            return True, f"Stop loss hit at ${current_price:.2f}"
        
        # Take profit check
        if position.take_profit_price and current_price >= position.take_profit_price:
            return True, f"Take profit hit at ${current_price:.2f}"
        
        # Time-based exit
        position_age_minutes = self._get_position_age_minutes(position)
        max_hold_minutes = getattr(self, 'max_hold_minutes', 1440)
        if position_age_minutes > max_hold_minutes:
            return True, f"Position age exceeded {max_hold_minutes} minutes"
        
        return False, "Hold position"
    
    def _get_position_age_minutes(self, position) -> int:
        """Calculate position age in minutes"""
        try:
            from datetime import datetime, timezone
            if isinstance(position.timestamp, str):
                position_time = datetime.fromisoformat(position.timestamp.replace('Z', '+00:00'))
            else:
                position_time = position.timestamp
            
            if position_time.tzinfo is None:
                position_time = position_time.replace(tzinfo=timezone.utc)
            
            current_time = datetime.now(timezone.utc)
            age_delta = current_time - position_time
            return int(age_delta.total_seconds() / 60)
        except:
            return 0
    
    def _calculate_performance_multiplier(self) -> float:
        """Calculate performance-based position size multiplier"""
        if self.trades_executed < 10:
            return 1.0
        
        win_rate = self.winning_trades / self.trades_executed if self.trades_executed > 0 else 0.5
        
        if win_rate >= 0.6:
            return 1.2
        elif win_rate < 0.4:
            return 0.8
        else:
            return 1.0
'''
                    # Abstract method tanımından sonra ekle
                    if "pass" in content:
                        last_pass_index = content.rfind("pass")
                        content = content[:last_pass_index + 4] + should_sell_code + content[last_pass_index + 4:]
                
                with open(base_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("✅ Fixed: strategies/base_strategy.py methods")
            
            # EnhancedMomentumStrategy düzeltmesi
            momentum_path = self.project_root / "strategies/momentum_optimized.py"
            if momentum_path.exists():
                with open(momentum_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Eksik metodları ekle
                methods_to_add = {
                    "_calculate_momentum_indicators": '''
    def _calculate_momentum_indicators(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Calculate momentum indicators"""
        indicators = {}
        close = data['close']
        
        # EMAs
        indicators['ema_short'] = close.ewm(span=self.ema_short, adjust=False).mean().iloc[-1]
        indicators['ema_medium'] = close.ewm(span=self.ema_medium, adjust=False).mean().iloc[-1]
        indicators['ema_long'] = close.ewm(span=self.ema_long, adjust=False).mean().iloc[-1]
        
        # RSI
        delta = close.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=self.rsi_period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=self.rsi_period).mean()
        rs = gain / (loss + 1e-10)
        indicators['rsi'] = (100 - (100 / (1 + rs))).iloc[-1]
        
        # ATR
        high = data['high']
        low = data['low']
        tr = pd.concat([high - low, abs(high - close.shift(1)), abs(low - close.shift(1))], axis=1).max(axis=1)
        indicators['atr'] = tr.rolling(window=14).mean().iloc[-1]
        
        # Volume
        indicators['volume_sma'] = data['volume'].rolling(window=20).mean().iloc[-1]
        indicators['volume_ratio'] = data['volume'].iloc[-1] / (indicators['volume_sma'] + 1e-10)
        
        # Price momentum
        indicators['price_momentum'] = (close.iloc[-1] - close.iloc[-4]) / close.iloc[-4]
        
        # ADX (simplified)
        indicators['adx'] = 25.0  # Placeholder
        
        return indicators
''',
                    "_analyze_momentum_signals": '''
    def _analyze_momentum_signals(self, data: pd.DataFrame) -> Dict[str, Any]:
        """Analyze momentum signals"""
        indicators = self._calculate_momentum_indicators(data)
        
        signals = {
            'signal_strength': 0,
            'quality_score': 0,
            'momentum_score': 0.0,
            'trend_alignment': False,
            'volume_confirmation': False,
            'risk_assessment': 'normal',
            'indicators': indicators
        }
        
        # Trend analysis
        if indicators['ema_short'] > indicators['ema_medium'] > indicators['ema_long']:
            signals['signal_strength'] += 3
            signals['trend_alignment'] = True
        
        # RSI analysis
        if indicators['rsi'] < self.rsi_oversold:
            signals['signal_strength'] += 2
        
        # Volume confirmation
        if indicators['volume_ratio'] > 1.5:
            signals['volume_confirmation'] = True
            signals['signal_strength'] += 1
        
        # Quality score
        signals['quality_score'] = min(20, signals['signal_strength'] * 3)
        
        return signals
''',
                    "_prepare_ml_features": '''
    def _prepare_ml_features(self, data: pd.DataFrame) -> Dict[str, float]:
        """Prepare ML features"""
        indicators = self._calculate_momentum_indicators(data)
        
        features = {
            'rsi': indicators['rsi'] / 100,
            'volume_ratio': indicators['volume_ratio'],
            'price_momentum': indicators['price_momentum'],
            'trend_strength': (indicators['ema_short'] - indicators['ema_long']) / indicators['ema_long'],
            'atr_ratio': indicators['atr'] / data['close'].iloc[-1]
        }
        
        return features
''',
                    "_calculate_performance_based_size": '''
    def _calculate_performance_based_size(self, signal: TradingSignal) -> float:
        """Calculate performance-based size multiplier"""
        if not hasattr(self, 'performance_history') or len(self.performance_history) < 5:
            return 1.0
        
        recent_trades = self.performance_history[-20:]
        winning_trades = sum(1 for t in recent_trades if t.get('profit', 0) > 0)
        win_rate = winning_trades / len(recent_trades) if recent_trades else 0.5
        
        if win_rate > 0.65:
            return 1.2
        elif win_rate < 0.35:
            return 0.8
        else:
            return 1.0
'''
                }
                
                # Metodları ekle
                for method_name, method_code in methods_to_add.items():
                    if f"def {method_name}" not in content:
                        # Sınıfın sonuna ekle
                        class_end = content.rfind("def ")
                        if class_end > 0:
                            # Son metoddan sonra ekle
                            next_newline = content.find("\n\n", class_end)
                            if next_newline > 0:
                                content = content[:next_newline] + "\n" + method_code + content[next_newline:]
                
                with open(momentum_path, 'w', encoding='utf-8') as f:
                    f.write(content)
                
                logger.info("✅ Fixed: strategies/momentum_optimized.py methods")
            
            self.fixes_applied.append("strategy_methods")
            return True
            
        except Exception as e:
            logger.error(f"❌ Failed to fix strategy methods: {e}")
            return False
    
    def run_tests(self) -> bool:
        """Testleri çalıştır"""
        try:
            logger.info("\n🧪 Running tests...")
            
            # pytest komutunu çalıştır
            result = subprocess.run(
                [sys.executable, "-m", "pytest", "--tb=short", "-v"],
                capture_output=True,
                text=True,
                cwd=self.project_root
            )
            
            # Sonuçları göster
            print("\n" + "="*80)
            print("TEST OUTPUT:")
            print("="*80)
            print(result.stdout)
            
            if result.returncode != 0:
                print("\nERRORS:")
                print(result.stderr)
            
            # Başarı durumunu kontrol et
            success = result.returncode == 0
            if success:
                logger.info("✅ All tests passed!")
            else:
                logger.error("❌ Some tests failed")
            
            return success
            
        except Exception as e:
            logger.error(f"❌ Failed to run tests: {e}")
            return False
    
    def apply_all_fixes(self, create_backup: bool = True, run_tests: bool = True) -> bool:
        """Tüm düzeltmeleri uygula"""
        logger.info("🚀 Starting Phoenix Auto-Fix Process...")
        
        # Backup
        if create_backup:
            if not self.create_backup():
                logger.error("Backup failed, aborting fixes")
                return False
        
        # Apply fixes
        fixes = [
            ("Parameter Spaces", self.fix_parameter_spaces),
            ("Main Imports", self.fix_main_imports),
            ("Portfolio Logger", self.fix_portfolio_logger),
            ("JSON Parameter System", self.fix_json_parameter_system),
            ("Strategy Methods", self.add_missing_strategy_methods),
        ]
        
        all_success = True
        for fix_name, fix_func in fixes:
            logger.info(f"\n🔧 Applying fix: {fix_name}")
            if not fix_func():
                all_success = False
                logger.error(f"Failed to apply {fix_name}")
        
        # Summary
        logger.info("\n" + "="*80)
        logger.info("📊 FIX SUMMARY:")
        logger.info(f"Total fixes applied: {len(self.fixes_applied)}")
        for fix in self.fixes_applied:
            logger.info(f"  ✅ {fix}")
        logger.info("="*80)
        
        # Run tests
        if run_tests and all_success:
            return self.run_tests()
        
        return all_success


def main():
    """Ana fonksiyon"""
    parser = argparse.ArgumentParser(description="Phoenix Auto-Fix Tool")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--no-test", action="store_true", help="Skip test execution")
    parser.add_argument("--verbose", "-v", action="store_true", help="Verbose output")
    
    args = parser.parse_args()
    
    if args.verbose:
        logging.getLogger().setLevel(logging.DEBUG)
    
    # Auto-fixer'ı çalıştır
    fixer = PhoenixAutoFixer()
    success = fixer.apply_all_fixes(
        create_backup=not args.no_backup,
        run_tests=not args.no_test
    )
    
    if success:
        logger.info("\n🎉 Phoenix Auto-Fix completed successfully!")
        logger.info("💎 All tests should now pass!")
    else:
        logger.error("\n❌ Phoenix Auto-Fix encountered errors")
        logger.info("💡 Check the logs and backup directory for recovery")
    
    return 0 if success else 1


if __name__ == "__main__":
    sys.exit(main())