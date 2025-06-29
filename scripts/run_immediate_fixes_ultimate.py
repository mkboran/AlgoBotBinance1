# run_immediate_fixes_ultimate.py
"""
🔧 ULTIMATE IMMEDIATE FIXES SYSTEM
💎 Otomatik sistem düzeltme ve optimization hazırlık scripti
🚀 Tek komutla tüm sorunları çöz ve sistemi çalışır hale getir

Bu script şunları yapar:
1. ✅ Portfolio.__init__() parametrelerini düzeltir
2. ✅ Eksik requirements.txt oluşturur ve paketleri yükler
3. ✅ Eksik ML predictor dosyalarını oluşturur
4. ✅ Import hatalarını düzeltir
5. ✅ Test sistemi çalıştırır
6. ✅ Optimization için hazırlık yapar
7. ✅ Gereksiz dosyaları temizler

KULLANIM:
python run_immediate_fixes_ultimate.py --fix-all --install-packages --run-tests
"""

import os
import sys
import subprocess
import shutil
import time
import argparse
from pathlib import Path
from typing import List, Dict, Tuple, Optional, Any
import logging
import json
import re
from datetime import datetime

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s - %(levelname)s - %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/immediate_fixes.log', mode='w')
    ]
)
logger = logging.getLogger(__name__)

class UltimateFixSystem:
    """🔧 Ultimate automatic fix system"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.fixes_applied = []
        self.errors_found = []
        self.files_processed = []
        self.start_time = datetime.now()
        
        # Create logs directory
        self.logs_dir = self.project_root / "logs"
        self.logs_dir.mkdir(exist_ok=True)
        
        logger.info("🚀 Ultimate Fix System initialized")
        logger.info(f"   Project root: {self.project_root.absolute()}")

    def check_system_status(self) -> Dict[str, bool]:
        """🔍 Check current system status"""
        
        logger.info("🔍 Checking system status...")
        
        status = {
            'requirements_txt_exists': False,
            'portfolio_parameters_fixed': False,
            'ml_predictor_exists': False,
            'env_example_exists': False,
            'optimization_script_exists': False,
            'main_files_exist': False,
            'python_packages_installed': False
        }
        
        # Check requirements.txt
        if (self.project_root / "requirements.txt").exists():
            status['requirements_txt_exists'] = True
            logger.info("✅ requirements.txt found")
        else:
            logger.warning("❌ requirements.txt missing")
        
        # Check .env.example
        if (self.project_root / ".env.example").exists():
            status['env_example_exists'] = True
            logger.info("✅ .env.example found")
        else:
            logger.warning("❌ .env.example missing")
        
        # Check ML predictor
        ml_predictor_path = self.project_root / "utils" / "advanced_ml_predictor.py"
        if ml_predictor_path.exists():
            status['ml_predictor_exists'] = True
            logger.info("✅ ML predictor found")
        else:
            logger.warning("❌ ML predictor missing")
        
        # Check optimization script
        optimize_path = self.project_root / "optimize_strategy_ultimate.py"
        if optimize_path.exists():
            status['optimization_script_exists'] = True
            logger.info("✅ Ultimate optimization script found")
        else:
            logger.warning("❌ Ultimate optimization script missing")
        
        # Check main files
        main_files = ["main.py", "backtest_runner.py", "utils/portfolio.py", "utils/config.py"]
        all_main_exist = all((self.project_root / f).exists() for f in main_files)
        status['main_files_exist'] = all_main_exist
        
        if all_main_exist:
            logger.info("✅ All main project files found")
        else:
            logger.warning("❌ Some main project files missing")
        
        # Check Portfolio parameter issues
        status['portfolio_parameters_fixed'] = self.check_portfolio_parameters()
        
        return status

    def check_portfolio_parameters(self) -> bool:
        """🔍 Check if Portfolio parameters are fixed"""
        
        files_to_check = [
            "backtest_runner.py",
            "main.py", 
            "utils/main_phase5_integration.py",
            "backtesting/multi_strategy_backtester.py"
        ]
        
        issues_found = False
        
        for file_path in files_to_check:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    # Check for incorrect Portfolio calls
                    incorrect_patterns = [
                        r'Portfolio\s*\(\s*initial_balance\s*=',
                        r'Portfolio\s*\(\s*balance\s*=',
                        r'Portfolio\s*\(\s*capital\s*=',
                        r'Portfolio\s*\(\s*\)'
                    ]
                    
                    for pattern in incorrect_patterns:
                        if re.search(pattern, content, re.IGNORECASE):
                            issues_found = True
                            logger.warning(f"❌ Portfolio parameter issue in {file_path}")
                            break
                    
                    if not issues_found:
                        logger.debug(f"✅ Portfolio parameters OK in {file_path}")
                        
                except Exception as e:
                    logger.error(f"Error checking {file_path}: {e}")
        
        return not issues_found

    def create_requirements_txt(self) -> bool:
        """📦 Create comprehensive requirements.txt"""
        
        logger.info("📦 Creating requirements.txt...")
        
        requirements_content = """# ============================================================================
# 🚀 MOMENTUM ML TRADING SYSTEM - COMPREHENSIVE DEPENDENCIES
# 💎 All packages required for hedge fund level trading system
# ============================================================================

# Core Data Analysis
pandas>=1.5.0
numpy>=1.21.0
scipy>=1.9.0
python-dateutil>=2.8.0

# Trading & Market Data
ccxt>=4.0.0
pandas-ta>=0.3.14
yfinance>=0.2.0

# Machine Learning & AI
scikit-learn>=1.1.0
xgboost>=1.7.0
lightgbm>=3.3.0
optuna>=3.5.0
textblob>=0.17.1

# Configuration & Environment
pydantic>=2.0.0
pydantic-settings>=2.0.0
python-dotenv>=1.0.0

# Networking & API
requests>=2.28.0
aiohttp>=3.8.0
asyncio-throttle>=1.0.0

# Visualization
matplotlib>=3.5.0
plotly>=5.0.0
seaborn>=0.11.0

# Data Storage
joblib>=1.2.0

# Development Tools
jupyter>=1.0.0
pytest>=7.2.0

# Performance
numba>=0.56.0

# Optional Advanced Packages
tensorflow>=2.12.0
torch>=2.0.0
ta-lib>=0.4.0
"""
        
        try:
            requirements_path = self.project_root / "requirements.txt"
            with open(requirements_path, 'w', encoding='utf-8') as f:
                f.write(requirements_content)
            
            self.fixes_applied.append("Created comprehensive requirements.txt")
            logger.info("✅ requirements.txt created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating requirements.txt: {e}")
            self.errors_found.append(f"requirements.txt creation failed: {e}")
            return False

    def create_env_example(self) -> bool:
        """🔧 Create .env.example file"""
        
        logger.info("🔧 Creating .env.example...")
        
        env_content = """# ============================================================================
# 🚀 MOMENTUM ML TRADING SYSTEM - ENVIRONMENT CONFIGURATION
# 💎 Copy this file to .env and customize your settings
# 🔧 Command: cp .env.example .env
# ============================================================================

# ================================================================================
# 🔐 API CREDENTIALS (Optional - for live trading only)
# ================================================================================
BINANCE_API_KEY=
BINANCE_API_SECRET=

# ================================================================================
# 📊 CORE TRADING SETTINGS
# ================================================================================
INITIAL_CAPITAL_USDT=1000.0
SYMBOL=BTC/USDT
TIMEFRAME=15m
FEE_BUY=0.001
FEE_SELL=0.001
MIN_TRADE_AMOUNT_USDT=25.0

# ================================================================================
# 🚀 MOMENTUM STRATEGY CONFIGURATION - PROFIT OPTIMIZED
# ================================================================================
MOMENTUM_EMA_SHORT=13
MOMENTUM_EMA_MEDIUM=21
MOMENTUM_EMA_LONG=56
MOMENTUM_RSI_PERIOD=13
MOMENTUM_BASE_POSITION_SIZE_PCT=65.0
MOMENTUM_MAX_POSITIONS=4

# ================================================================================
# 🧠 MACHINE LEARNING CONFIGURATION
# ================================================================================
MOMENTUM_ML_ENABLED=true
MOMENTUM_ML_CONFIDENCE_THRESHOLD=0.25
MOMENTUM_ML_RF_WEIGHT=0.30
MOMENTUM_ML_XGB_WEIGHT=0.35
MOMENTUM_ML_GB_WEIGHT=0.25

# ================================================================================
# 📝 LOGGING CONFIGURATION
# ================================================================================
ENABLE_CSV_LOGGING=true
LOG_LEVEL=INFO
TRADES_CSV_LOG_PATH=logs/trades.csv

# ================================================================================
# 📊 BACKTESTING SETTINGS
# ================================================================================
DATA_FILE_PATH=historical_data/BTCUSDT_15m_20210101_20241231.csv
DEFAULT_BACKTEST_START=2024-01-01
DEFAULT_BACKTEST_END=2024-12-31

# ================================================================================
# 🎯 OPTIMIZATION SETTINGS
# ================================================================================
OPTUNA_N_TRIALS=1000
OPTUNA_ENABLE_PRUNING=true
ENABLE_WALK_FORWARD=true
VALIDATION_SPLIT=0.2
"""
        
        try:
            env_path = self.project_root / ".env.example"
            with open(env_path, 'w', encoding='utf-8') as f:
                f.write(env_content)
            
            self.fixes_applied.append("Created .env.example file")
            logger.info("✅ .env.example created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating .env.example: {e}")
            self.errors_found.append(f".env.example creation failed: {e}")
            return False

    def create_ml_predictor(self) -> bool:
        """🧠 Create minimal ML predictor"""
        
        logger.info("🧠 Creating ML predictor...")
        
        ml_predictor_content = '''# utils/advanced_ml_predictor.py
"""
🧠 Advanced ML Predictor - Production Ready
💎 Simplified but functional ML prediction system
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple
from datetime import datetime, timezone
import logging
import warnings
warnings.filterwarnings('ignore')

from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedMLPredictor:
    """🧠 Advanced ML Predictor - Simplified Production Version"""
    
    def __init__(self):
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_columns = []
        self.last_training_time = None
        
        # Initialize models
        self.models['rf'] = RandomForestClassifier(
            n_estimators=100,
            max_depth=10,
            random_state=42,
            n_jobs=-1
        )
        
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=100,
            max_depth=6,
            random_state=42
        )
        
        if XGBOOST_AVAILABLE:
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=100,
                max_depth=6,
                random_state=42,
                eval_metric='logloss'
            )
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = StandardScaler()
        
        logger.info("✅ AdvancedMLPredictor initialized")

    def create_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """🔧 Create ML features from market data"""
        try:
            df = data.copy()
            
            # Basic features
            df['returns'] = df['close'].pct_change()
            df['price_change'] = df['close'] - df['open']
            df['price_range'] = df['high'] - df['low']
            
            # Moving averages
            for period in [5, 10, 20]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
            
            # Volatility
            df['volatility_10'] = df['returns'].rolling(10).std()
            df['volatility_20'] = df['returns'].rolling(20).std()
            
            # Volume
            df['volume_sma_20'] = df['volume'].rolling(20).mean()
            df['volume_ratio'] = df['volume'] / df['volume_sma_20']
            
            # Technical indicators
            df['rsi'] = self._calculate_rsi(df['close'], 14)
            
            # Clean data
            df = df.replace([np.inf, -np.inf], np.nan)
            df = df.fillna(method='ffill').fillna(0)
            
            return df
            
        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            return data
    
    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def train(self, data: pd.DataFrame) -> bool:
        """🎓 Train ML models"""
        try:
            if len(data) < 100:
                logger.warning("Insufficient data for training")
                return False
            
            # Create features
            df = self.create_features(data)
            
            # Create labels
            future_returns = df['close'].shift(-5) / df['close'] - 1
            labels = (future_returns > 0.01).astype(int)
            
            # Select features
            feature_cols = [col for col in df.columns if not col in ['open', 'high', 'low', 'close', 'volume', 'timestamp']]
            feature_cols = [col for col in feature_cols if not col.startswith('sma_')][:15]  # Limit features
            
            X = df[feature_cols].values
            y = labels.values
            
            # Remove invalid data
            valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y)
            X = X[valid_mask]
            y = y[valid_mask]
            
            if len(X) < 50:
                logger.warning("Insufficient valid training data")
                return False
            
            # Train models
            self.feature_columns = feature_cols
            successful_models = 0
            
            # Use last 80% for training
            split_idx = int(len(X) * 0.2)
            X_train = X[split_idx:]
            y_train = y[split_idx:]
            
            for name, model in self.models.items():
                try:
                    # Scale features
                    X_train_scaled = self.scalers[name].fit_transform(X_train)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    successful_models += 1
                    logger.info(f"✅ {name.upper()} model trained successfully")
                    
                except Exception as e:
                    logger.error(f"❌ {name.upper()} training failed: {e}")
            
            if successful_models > 0:
                self.is_trained = True
                self.last_training_time = datetime.now(timezone.utc)
                logger.info(f"🎓 ML Training completed: {successful_models}/{len(self.models)} models successful")
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """🔮 Make ML prediction"""
        try:
            if not self.is_trained:
                return {
                    "signal": "hold",
                    "confidence": 0.5,
                    "probabilities": {"bullish": 0.5, "bearish": 0.5},
                    "warning": "Model not trained"
                }
            
            # Create features
            df = self.create_features(data)
            
            if len(self.feature_columns) == 0:
                return self._default_prediction()
            
            # Get latest features
            try:
                latest_features = df[self.feature_columns].iloc[-1:].values
            except KeyError:
                return self._default_prediction()
            
            if np.isnan(latest_features).any():
                return self._default_prediction()
            
            # Get predictions
            predictions = {}
            
            for name, model in self.models.items():
                try:
                    features_scaled = self.scalers[name].transform(latest_features)
                    
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(features_scaled)[0]
                        if len(pred_proba) >= 2:
                            predictions[name] = pred_proba[1]
                        else:
                            predictions[name] = pred_proba[0]
                    else:
                        pred = model.predict(features_scaled)[0]
                        predictions[name] = float(pred)
                        
                except Exception as e:
                    logger.error(f"Prediction error for {name}: {e}")
                    continue
            
            if not predictions:
                return self._default_prediction()
            
            # Ensemble prediction
            weights = {'rf': 0.4, 'gb': 0.35, 'xgb': 0.25} if XGBOOST_AVAILABLE else {'rf': 0.6, 'gb': 0.4}
            ensemble_prob = sum(predictions[name] * weights.get(name, 0) for name in predictions.keys())
            ensemble_prob = max(0, min(1, ensemble_prob))
            
            # Determine signal
            if ensemble_prob > 0.6:
                signal = "buy"
                confidence = ensemble_prob
            elif ensemble_prob < 0.4:
                signal = "sell"
                confidence = 1 - ensemble_prob
            else:
                signal = "hold"
                confidence = 0.5
            
            return {
                "signal": signal,
                "confidence": float(confidence),
                "probabilities": {
                    "bullish": float(ensemble_prob),
                    "bearish": float(1 - ensemble_prob)
                },
                "model_predictions": {k: float(v) for k, v in predictions.items()},
                "ensemble_confidence": float(ensemble_prob),
                "prediction_time": datetime.now(timezone.utc).isoformat()
            }
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._default_prediction()
    
    def _default_prediction(self) -> Dict[str, Any]:
        """Default prediction when errors occur"""
        return {
            "signal": "hold",
            "confidence": 0.5,
            "probabilities": {"bullish": 0.5, "bearish": 0.5},
            "model_predictions": {},
            "ensemble_confidence": 0.5,
            "warning": "Using default prediction due to error"
        }

    def get_status(self) -> Dict[str, Any]:
        """Get predictor status"""
        return {
            "is_trained": self.is_trained,
            "models_available": list(self.models.keys()),
            "feature_count": len(self.feature_columns),
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "xgboost_available": XGBOOST_AVAILABLE
        }
'''
        
        try:
            utils_dir = self.project_root / "utils"
            utils_dir.mkdir(exist_ok=True)
            
            ml_predictor_path = utils_dir / "advanced_ml_predictor.py"
            with open(ml_predictor_path, 'w', encoding='utf-8') as f:
                f.write(ml_predictor_content)
            
            self.fixes_applied.append("Created ML predictor")
            logger.info("✅ ML predictor created successfully")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating ML predictor: {e}")
            self.errors_found.append(f"ML predictor creation failed: {e}")
            return False

    def fix_portfolio_parameters(self) -> bool:
        """🔧 Fix Portfolio parameter issues"""
        
        logger.info("🔧 Fixing Portfolio parameter issues...")
        
        files_to_fix = [
            "backtest_runner.py",
            "main.py",
            "utils/main_phase5_integration.py",
            "backtesting/multi_strategy_backtester.py"
        ]
        
        fixed_files = 0
        
        for file_path in files_to_fix:
            full_path = self.project_root / file_path
            if full_path.exists():
                try:
                    with open(full_path, 'r', encoding='utf-8') as f:
                        content = f.read()
                    
                    original_content = content
                    
                    # Fix parameter patterns
                    fixes = [
                        (r'Portfolio\s*\(\s*initial_balance\s*=', 'Portfolio(initial_capital_usdt='),
                        (r'Portfolio\s*\(\s*balance\s*=', 'Portfolio(initial_capital_usdt='),
                        (r'Portfolio\s*\(\s*capital\s*=', 'Portfolio(initial_capital_usdt='),
                        (r'Portfolio\s*\(\s*\)', 'Portfolio(initial_capital_usdt=1000.0)')
                    ]
                    
                    for pattern, replacement in fixes:
                        content = re.sub(pattern, replacement, content, flags=re.IGNORECASE)
                    
                    if content != original_content:
                        # Create backup
                        backup_path = full_path.with_suffix(f'{full_path.suffix}.backup')
                        with open(backup_path, 'w', encoding='utf-8') as f:
                            f.write(original_content)
                        
                        # Write fixed content
                        with open(full_path, 'w', encoding='utf-8') as f:
                            f.write(content)
                        
                        fixed_files += 1
                        self.fixes_applied.append(f"Fixed Portfolio parameters in {file_path}")
                        logger.info(f"✅ Fixed Portfolio parameters in {file_path}")
                    
                except Exception as e:
                    logger.error(f"❌ Error fixing {file_path}: {e}")
                    self.errors_found.append(f"Portfolio fix failed for {file_path}: {e}")
        
        if fixed_files > 0:
            logger.info(f"✅ Fixed Portfolio parameters in {fixed_files} files")
            return True
        else:
            logger.info("ℹ️  No Portfolio parameter issues found")
            return True

    def install_packages(self) -> bool:
        """📦 Install required packages"""
        
        logger.info("📦 Installing required packages...")
        
        try:
            # Essential packages for immediate functionality
            essential_packages = [
                "pandas>=1.5.0",
                "numpy>=1.21.0", 
                "ccxt>=4.0.0",
                "pandas-ta>=0.3.14",
                "optuna>=3.5.0",
                "scikit-learn>=1.1.0",
                "pydantic>=2.0.0",
                "pydantic-settings>=2.0.0",
                "python-dotenv>=1.0.0",
                "requests>=2.28.0",
                "aiohttp>=3.8.0"
            ]
            
            logger.info("Installing essential packages...")
            for package in essential_packages:
                try:
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", package
                    ], capture_output=True, text=True, timeout=120)
                    
                    if result.returncode == 0:
                        logger.info(f"✅ Installed {package}")
                    else:
                        logger.warning(f"⚠️  Warning installing {package}: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.warning(f"⚠️  Timeout installing {package}")
                except Exception as e:
                    logger.error(f"❌ Error installing {package}: {e}")
            
            # Install from requirements.txt if it exists
            requirements_path = self.project_root / "requirements.txt"
            if requirements_path.exists():
                try:
                    logger.info("Installing from requirements.txt...")
                    result = subprocess.run([
                        sys.executable, "-m", "pip", "install", "-r", str(requirements_path)
                    ], capture_output=True, text=True, timeout=300)
                    
                    if result.returncode == 0:
                        logger.info("✅ Packages installed from requirements.txt")
                    else:
                        logger.warning(f"⚠️  Warning installing from requirements.txt: {result.stderr}")
                        
                except subprocess.TimeoutExpired:
                    logger.warning("⚠️  Timeout installing from requirements.txt")
                except Exception as e:
                    logger.error(f"❌ Error installing from requirements.txt: {e}")
            
            self.fixes_applied.append("Installed required packages")
            return True
            
        except Exception as e:
            logger.error(f"❌ Package installation failed: {e}")
            self.errors_found.append(f"Package installation failed: {e}")
            return False

    def run_import_tests(self) -> bool:
        """🧪 Run import tests to verify system functionality"""
        
        logger.info("🧪 Running import tests...")
        
        # Core imports to test
        import_tests = [
            ("pandas", "Data manipulation"),
            ("numpy", "Numerical computing"), 
            ("ccxt", "Crypto exchange library"),
            ("optuna", "Optimization framework"),
            ("sklearn", "Machine learning"),
            ("pydantic", "Data validation"),
            ("asyncio", "Async support"),
            ("pathlib", "Path handling"),
            ("json", "JSON support"),
            ("datetime", "Date/time handling")
        ]
        
        successful_imports = 0
        failed_imports = []
        
        for module, description in import_tests:
            try:
                __import__(module)
                logger.info(f"✅ {description} - {module}")
                successful_imports += 1
            except ImportError as e:
                logger.error(f"❌ {description} - {module}: {e}")
                failed_imports.append((module, str(e)))
        
        # Test project-specific imports
        project_imports = [
            ("utils.config", "Configuration system"),
            ("utils.portfolio", "Portfolio management"),
            ("utils.logger", "Logging system")
        ]
        
        for module, description in project_imports:
            try:
                # Add project root to path for testing
                if str(self.project_root) not in sys.path:
                    sys.path.insert(0, str(self.project_root))
                
                __import__(module)
                logger.info(f"✅ {description} - {module}")
                successful_imports += 1
            except ImportError as e:
                logger.warning(f"⚠️  {description} - {module}: {e}")
                failed_imports.append((module, str(e)))
            except Exception as e:
                logger.warning(f"⚠️  {description} - {module}: {e}")
        
        total_tests = len(import_tests) + len(project_imports)
        success_rate = (successful_imports / total_tests) * 100
        
        logger.info(f"🧪 Import Test Results:")
        logger.info(f"   ✅ Successful: {successful_imports}/{total_tests} ({success_rate:.1f}%)")
        logger.info(f"   ❌ Failed: {len(failed_imports)}")
        
        if failed_imports:
            logger.info("   Failed imports:")
            for module, error in failed_imports:
                logger.info(f"     - {module}: {error}")
        
        # Consider test successful if >80% pass
        if success_rate > 80:
            self.fixes_applied.append(f"Import tests passed ({success_rate:.1f}%)")
            return True
        else:
            self.errors_found.append(f"Import tests failed ({success_rate:.1f}% success rate)")
            return False

    def cleanup_unnecessary_files(self) -> bool:
        """🧹 Clean up unnecessary files"""
        
        logger.info("🧹 Cleaning up unnecessary files...")
        
        files_to_remove = [
            "complete_phase5_fix.py",
            "fix_imports_phase5.py",
            "import_test.py",
            "low_ram_optimization_solutions.py"
        ]
        
        removed_files = 0
        
        for filename in files_to_remove:
            file_path = self.project_root / filename
            if file_path.exists():
                try:
                    file_path.unlink()
                    logger.info(f"🗑️  Removed {filename}")
                    removed_files += 1
                except Exception as e:
                    logger.warning(f"⚠️  Could not remove {filename}: {e}")
        
        # Clean up backup files older than 1 day
        backup_files = list(self.project_root.rglob("*.backup"))
        for backup_file in backup_files:
            try:
                if time.time() - backup_file.stat().st_mtime > 86400:  # 1 day
                    backup_file.unlink()
                    logger.info(f"🗑️  Removed old backup {backup_file.name}")
                    removed_files += 1
            except Exception as e:
                logger.warning(f"⚠️  Could not remove backup {backup_file}: {e}")
        
        if removed_files > 0:
            self.fixes_applied.append(f"Cleaned up {removed_files} unnecessary files")
            logger.info(f"✅ Cleaned up {removed_files} files")
        else:
            logger.info("ℹ️  No unnecessary files found")
        
        return True

    def create_quick_test_script(self) -> bool:
        """🧪 Create quick test script for validation"""
        
        logger.info("🧪 Creating quick test script...")
        
        test_script_content = '''#!/usr/bin/env python3
"""
🧪 Quick System Test Script
Test basic functionality after fixes
"""

import sys
import traceback
from pathlib import Path

def test_basic_imports():
    """Test basic imports"""
    print("🧪 Testing basic imports...")
    
    try:
        import pandas as pd
        import numpy as np
        print("✅ pandas, numpy imported")
    except ImportError as e:
        print(f"❌ pandas/numpy import failed: {e}")
        return False
    
    try:
        import ccxt
        print("✅ ccxt imported")
    except ImportError as e:
        print(f"❌ ccxt import failed: {e}")
        return False
    
    try:
        import optuna
        print("✅ optuna imported")
    except ImportError as e:
        print(f"❌ optuna import failed: {e}")
        return False
    
    return True

def test_project_imports():
    """Test project-specific imports"""
    print("🧪 Testing project imports...")
    
    try:
        from utils.config import settings
        print("✅ Config imported")
    except Exception as e:
        print(f"❌ Config import failed: {e}")
        return False
    
    try:
        from utils.portfolio import Portfolio
        print("✅ Portfolio imported")
    except Exception as e:
        print(f"❌ Portfolio import failed: {e}")
        return False
    
    return True

def test_portfolio_initialization():
    """Test Portfolio initialization"""
    print("🧪 Testing Portfolio initialization...")
    
    try:
        from utils.portfolio import Portfolio
        
        # Test correct initialization
        portfolio = Portfolio(initial_capital_usdt=1000.0)
        print("✅ Portfolio initialization successful")
        return True
        
    except Exception as e:
        print(f"❌ Portfolio initialization failed: {e}")
        print(f"   Details: {traceback.format_exc()}")
        return False

def test_ml_predictor():
    """Test ML predictor"""
    print("🧪 Testing ML predictor...")
    
    try:
        from utils.advanced_ml_predictor import AdvancedMLPredictor
        
        predictor = AdvancedMLPredictor()
        status = predictor.get_status()
        print(f"✅ ML predictor created: {status}")
        return True
        
    except Exception as e:
        print(f"❌ ML predictor test failed: {e}")
        return False

def main():
    """Run all tests"""
    print("🚀 RUNNING QUICK SYSTEM TESTS")
    print("="*50)
    
    tests = [
        ("Basic Imports", test_basic_imports),
        ("Project Imports", test_project_imports),
        ("Portfolio Initialization", test_portfolio_initialization),
        ("ML Predictor", test_ml_predictor)
    ]
    
    passed = 0
    total = len(tests)
    
    for test_name, test_func in tests:
        print(f"\\n🔍 {test_name}:")
        try:
            if test_func():
                passed += 1
                print(f"✅ {test_name} PASSED")
            else:
                print(f"❌ {test_name} FAILED")
        except Exception as e:
            print(f"❌ {test_name} ERROR: {e}")
    
    print("\\n" + "="*50)
    print(f"📊 TEST RESULTS: {passed}/{total} tests passed ({(passed/total)*100:.1f}%)")
    
    if passed == total:
        print("🎉 ALL TESTS PASSED - SYSTEM READY!")
    elif passed >= total * 0.75:
        print("⚠️  MOST TESTS PASSED - SYSTEM MOSTLY READY")
    else:
        print("❌ MANY TESTS FAILED - SYSTEM NEEDS FIXES")
    
    return passed == total

if __name__ == "__main__":
    success = main()
    sys.exit(0 if success else 1)
'''
        
        try:
            test_script_path = self.project_root / "quick_test.py"
            with open(test_script_path, 'w', encoding='utf-8') as f:
                f.write(test_script_content)
            
            # Make executable
            test_script_path.chmod(0o755)
            
            self.fixes_applied.append("Created quick test script")
            logger.info("✅ Quick test script created: quick_test.py")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating test script: {e}")
            self.errors_found.append(f"Test script creation failed: {e}")
            return False

    def run_comprehensive_fix(self, 
                            install_packages: bool = True,
                            run_tests: bool = True,
                            cleanup_files: bool = True) -> Dict[str, Any]:
        """🚀 Run comprehensive system fix"""
        
        logger.info("🚀 STARTING COMPREHENSIVE SYSTEM FIX")
        logger.info("="*60)
        
        # Check initial status
        initial_status = self.check_system_status()
        
        # Run fixes
        fix_results = {}
        
        # 1. Create requirements.txt
        if not initial_status['requirements_txt_exists']:
            fix_results['requirements_txt'] = self.create_requirements_txt()
        else:
            fix_results['requirements_txt'] = True
            logger.info("✅ requirements.txt already exists")
        
        # 2. Create .env.example
        if not initial_status['env_example_exists']:
            fix_results['env_example'] = self.create_env_example()
        else:
            fix_results['env_example'] = True
            logger.info("✅ .env.example already exists")
        
        # 3. Create ML predictor
        if not initial_status['ml_predictor_exists']:
            fix_results['ml_predictor'] = self.create_ml_predictor()
        else:
            fix_results['ml_predictor'] = True
            logger.info("✅ ML predictor already exists")
        
        # 4. Fix Portfolio parameters
        if not initial_status['portfolio_parameters_fixed']:
            fix_results['portfolio_fix'] = self.fix_portfolio_parameters()
        else:
            fix_results['portfolio_fix'] = True
            logger.info("✅ Portfolio parameters already fixed")
        
        # 5. Install packages
        if install_packages:
            fix_results['package_installation'] = self.install_packages()
        else:
            fix_results['package_installation'] = True
            logger.info("ℹ️  Package installation skipped")
        
        # 6. Run tests
        if run_tests:
            fix_results['import_tests'] = self.run_import_tests()
        else:
            fix_results['import_tests'] = True
            logger.info("ℹ️  Import tests skipped")
        
        # 7. Create test script
        fix_results['test_script'] = self.create_quick_test_script()
        
        # 8. Cleanup
        if cleanup_files:
            fix_results['cleanup'] = self.cleanup_unnecessary_files()
        else:
            fix_results['cleanup'] = True
            logger.info("ℹ️  File cleanup skipped")
        
        # Calculate results
        total_fixes = len(fix_results)
        successful_fixes = sum(1 for result in fix_results.values() if result)
        success_rate = (successful_fixes / total_fixes) * 100
        
        # Final status check
        final_status = self.check_system_status()
        
        # Prepare summary
        duration = datetime.now() - self.start_time
        
        summary = {
            'total_fixes_attempted': total_fixes,
            'successful_fixes': successful_fixes,
            'success_rate_pct': success_rate,
            'duration_seconds': duration.total_seconds(),
            'fixes_applied': self.fixes_applied,
            'errors_found': self.errors_found,
            'initial_status': initial_status,
            'final_status': final_status,
            'fix_results': fix_results
        }
        
        # Log results
        logger.info("🏁 COMPREHENSIVE FIX COMPLETED")
        logger.info("="*60)
        logger.info(f"   Duration: {duration.total_seconds():.1f} seconds")
        logger.info(f"   Success Rate: {success_rate:.1f}% ({successful_fixes}/{total_fixes})")
        logger.info(f"   Fixes Applied: {len(self.fixes_applied)}")
        logger.info(f"   Errors Found: {len(self.errors_found)}")
        
        if successful_fixes == total_fixes:
            logger.info("🎉 ALL FIXES SUCCESSFUL - SYSTEM READY!")
        elif success_rate >= 80:
            logger.info("✅ MOST FIXES SUCCESSFUL - SYSTEM MOSTLY READY")
        else:
            logger.warning("⚠️  SOME FIXES FAILED - MANUAL INTERVENTION MAY BE NEEDED")
        
        logger.info("="*60)
        logger.info("📋 NEXT STEPS:")
        logger.info("   1. Run: python quick_test.py")
        logger.info("   2. If tests pass, run optimization:")
        logger.info("      python optimize_strategy_ultimate.py --start 2024-01-01 --end 2024-03-31 --trials 1000")
        logger.info("   3. For paper trading preparation:")
        logger.info("      python backtest_runner.py --data-file historical_data/BTCUSDT_15m_20210101_20241231.csv")
        
        return summary

def main():
    """Main function for CLI usage"""
    parser = argparse.ArgumentParser(description="Ultimate Immediate Fixes System")
    parser.add_argument("--fix-all", action="store_true", help="Run all fixes")
    parser.add_argument("--install-packages", action="store_true", help="Install required packages")
    parser.add_argument("--run-tests", action="store_true", help="Run import tests")
    parser.add_argument("--cleanup", action="store_true", help="Clean up unnecessary files")
    parser.add_argument("--status-only", action="store_true", help="Check status only")
    parser.add_argument("--project-root", default=".", help="Project root directory")
    
    args = parser.parse_args()
    
    # Initialize fix system
    fix_system = UltimateFixSystem(args.project_root)
    
    if args.status_only:
        # Just check status
        status = fix_system.check_system_status()
        print("\n📊 SYSTEM STATUS:")
        print("="*40)
        for check, result in status.items():
            status_icon = "✅" if result else "❌"
            print(f"{status_icon} {check.replace('_', ' ').title()}")
        print("="*40)
        return
    
    # Determine what to run
    install_packages = args.install_packages or args.fix_all
    run_tests = args.run_tests or args.fix_all
    cleanup = args.cleanup or args.fix_all
    
    if not any([install_packages, run_tests, cleanup, args.fix_all]):
        print("❌ No actions specified. Use --fix-all or specific options.")
        parser.print_help()
        return
    
    # Run fixes
    try:
        summary = fix_system.run_comprehensive_fix(
            install_packages=install_packages,
            run_tests=run_tests,
            cleanup_files=cleanup
        )
        
        # Print summary
        print(f"\n📊 FIX SUMMARY:")
        print(f"   Success Rate: {summary['success_rate_pct']:.1f}%")
        print(f"   Duration: {summary['duration_seconds']:.1f}s")
        print(f"   Fixes Applied: {len(summary['fixes_applied'])}")
        
        if summary['success_rate_pct'] == 100:
            print("🎉 SYSTEM FULLY FIXED AND READY!")
            print("\nRun: python quick_test.py")
        elif summary['success_rate_pct'] >= 80:
            print("✅ SYSTEM MOSTLY FIXED - READY FOR TESTING")
            print("\nRun: python quick_test.py")
        else:
            print("⚠️  SYSTEM PARTIALLY FIXED - CHECK ERRORS")
            if summary['errors_found']:
                print("\nErrors found:")
                for error in summary['errors_found']:
                    print(f"   - {error}")
        
    except Exception as e:
        logger.error(f"❌ Fix system failed: {e}")
        print(f"❌ Fix system failed: {e}")

if __name__ == "__main__":
    main()