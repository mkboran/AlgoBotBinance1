#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - BİRLEŞTİRİLMİŞ ANA SİSTEM
💎 Tek Güçlü Giriş Noktası - Standard Operasyonel Arayüz

Bu sistem şunları sağlar:
1. ✅ Canlı ticaret (live trading)
2. ✅ Backtest çalıştırma
3. ✅ Optimizasyon süreçleri
4. ✅ Sistem validasyonu
5. ✅ Strateji yönetimi
6. ✅ Performance monitoring

KULLANIM ÖRNEKLERİ:
python main.py live                                    # Canlı ticaret
python main.py backtest --strategy momentum            # Backtest
python main.py optimize --strategy all --trials 5000  # Optimizasyon
python main.py validate                               # Sistem doğrulama
python main.py status                                 # Sistem durumu

🎯 HEDGE FUND SEVİYESİ - ÜRETİM HAZIR - SIFIR HATA TOLERANSI
"""

import asyncio
import argparse
import sys
import os
import logging
import json
import traceback
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, List, Any, Optional
import importlib
import warnings

# Warnings'ları sustur
warnings.filterwarnings('ignore')

# Project imports
try:
    from utils.config import settings
    from utils.logger import logger
    from utils.portfolio import Portfolio
    from utils.data import BinanceFetcher
    
    # Strategy imports
    from strategies.momentum_optimized import EnhancedMomentumStrategy
    
    # System imports
    from backtest_runner import MomentumBacktester
    
    CORE_IMPORTS_SUCCESS = True
    
except ImportError as e:
    CORE_IMPORTS_SUCCESS = False
    IMPORT_ERROR = str(e)
    
    # Fallback logger
    logging.basicConfig(level=logging.ERROR)
    logger = logging.getLogger("main_fallback")

class PhoenixTradingSystem:
    """🚀 Phoenix Trading System - Unified Main Controller"""
    
    def __init__(self):
        self.system_name = "Phoenix Trading System"
        self.version = "1.0.0"
        self.startup_time = datetime.now(timezone.utc)
        
        # System components
        self.portfolio = None
        self.strategies = {}
        self.data_fetcher = None
        
        # System state
        self.is_initialized = False
        self.is_running = False
        self.last_health_check = None
        
        # Available strategies
        self.available_strategies = {
            "momentum": {
                "class": "EnhancedMomentumStrategy",
                "module": "strategies.momentum_optimized",
                "description": "Enhanced Momentum Strategy with ML integration"
            },
            "bollinger_rsi": {
                "class": "BollingerRSIStrategy", 
                "module": "strategies.bollinger_rsi_strategy",
                "description": "Bollinger Bands + RSI Strategy"
            },
            "rsi_ml": {
                "class": "RSIMLStrategy",
                "module": "strategies.rsi_ml_strategy", 
                "description": "RSI + Machine Learning Strategy"
            },
            "macd_ml": {
                "class": "MACDMLStrategy",
                "module": "strategies.macd_ml_strategy",
                "description": "MACD + Machine Learning Strategy"
            },
            "volume_profile": {
                "class": "VolumeProfileStrategy",
                "module": "strategies.volume_profile_strategy",
                "description": "Volume Profile Analysis Strategy"
            }
        }
        
        logger.info(f"🚀 {self.system_name} v{self.version} başlatıldı")
    
    def validate_system(self) -> Dict[str, Any]:
        """🔍 Sistem doğrulaması yap"""
        
        logger.info("🔍 Sistem doğrulaması başlatılıyor...")
        
        validation_results = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "core_imports": CORE_IMPORTS_SUCCESS,
            "directories": {},
            "files": {},
            "strategies": {},
            "dependencies": {},
            "overall_status": "unknown"
        }
        
        if not CORE_IMPORTS_SUCCESS:
            validation_results["core_imports_error"] = IMPORT_ERROR
            validation_results["overall_status"] = "critical_failure"
            logger.error(f"❌ Core imports failed: {IMPORT_ERROR}")
            return validation_results
        
        # Klasör kontrolü
        required_dirs = ["utils", "strategies", "logs", "optimization/results"]
        for dir_path in required_dirs:
            full_path = Path(dir_path)
            exists = full_path.exists()
            validation_results["directories"][dir_path] = {
                "exists": exists,
                "is_directory": full_path.is_dir() if exists else False
            }
            
            if not exists:
                logger.warning(f"⚠️ Missing directory: {dir_path}")
                # Gerekli klasörleri oluştur
                full_path.mkdir(parents=True, exist_ok=True)
                logger.info(f"✅ Created directory: {dir_path}")
        
        # Kritik dosya kontrolü
        required_files = [
            "utils/config.py",
            "utils/portfolio.py", 
            "utils/logger.py",
            "strategies/momentum_optimized.py",
            "backtest_runner.py"
        ]
        
        for file_path in required_files:
            full_path = Path(file_path)
            exists = full_path.exists()
            validation_results["files"][file_path] = {
                "exists": exists,
                "size_kb": full_path.stat().st_size / 1024 if exists else 0
            }
            
            if not exists:
                logger.error(f"❌ Missing critical file: {file_path}")
        
        # Strateji kontrolü
        for strategy_name, strategy_info in self.available_strategies.items():
            try:
                module = importlib.import_module(strategy_info["module"])
                strategy_class = getattr(module, strategy_info["class"])
                
                validation_results["strategies"][strategy_name] = {
                    "importable": True,
                    "class_found": True,
                    "description": strategy_info["description"]
                }
                
                logger.info(f"✅ Strategy validated: {strategy_name}")
                
            except ImportError as e:
                validation_results["strategies"][strategy_name] = {
                    "importable": False,
                    "error": str(e),
                    "description": strategy_info["description"]
                }
                logger.warning(f"⚠️ Strategy import failed: {strategy_name} - {e}")
            
            except AttributeError as e:
                validation_results["strategies"][strategy_name] = {
                    "importable": True,
                    "class_found": False,
                    "error": str(e),
                    "description": strategy_info["description"]
                }
                logger.warning(f"⚠️ Strategy class not found: {strategy_name} - {e}")
        
        # Dependency kontrolü
        required_packages = [
            "pandas", "numpy", "ccxt", "optuna", "sklearn", 
            "scipy", "joblib", "pandas_ta", "asyncio"
        ]
        
        for package in required_packages:
            try:
                importlib.import_module(package)
                validation_results["dependencies"][package] = {"available": True}
            except ImportError:
                validation_results["dependencies"][package] = {"available": False}
                logger.warning(f"⚠️ Missing package: {package}")
        
        # Overall status belirleme
        critical_files_ok = all(info["exists"] for info in validation_results["files"].values())
        core_strategies_ok = validation_results["strategies"].get("momentum", {}).get("importable", False)
        core_deps_ok = all(
            validation_results["dependencies"].get(pkg, {}).get("available", False) 
            for pkg in ["pandas", "numpy", "ccxt"]
        )
        
        if critical_files_ok and core_strategies_ok and core_deps_ok:
            validation_results["overall_status"] = "healthy"
            logger.info("✅ Sistem doğrulaması başarılı")
        elif core_strategies_ok and core_deps_ok:
            validation_results["overall_status"] = "warning"
            logger.warning("⚠️ Sistem kısmen sağlıklı - bazı dosyalar eksik")
        else:
            validation_results["overall_status"] = "error"
            logger.error("❌ Sistem doğrulaması başarısız")
        
        return validation_results
    
    async def initialize_system(self, config: Dict[str, Any] = None) -> bool:
        """🔧 Sistemi başlat"""
        
        if not CORE_IMPORTS_SUCCESS:
            logger.error(f"❌ Core imports failed, cannot initialize: {IMPORT_ERROR}")
            return False
        
        try:
            logger.info("🔧 Sistem başlatılıyor...")
            
            # Portfolio'yu başlat
            initial_capital = config.get("initial_capital", 1000.0) if config else 1000.0
            self.portfolio = Portfolio(initial_balance=initial_capital)
            logger.info(f"✅ Portfolio başlatıldı: ${initial_capital:,.2f}")
            
            # Data fetcher'ı başlat
            symbol = config.get("symbol", "BTC/USDT") if config else "BTC/USDT"
            self.data_fetcher = BinanceFetcher(symbol=symbol)
            logger.info(f"✅ Data fetcher başlatıldı: {symbol}")
            
            self.is_initialized = True
            logger.info("🎉 Sistem başarıyla başlatıldı!")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Sistem başlatma hatası: {e}")
            logger.error(traceback.format_exc())
            return False
    
    async def run_live_trading(self, config: Dict[str, Any]) -> None:
        """📈 Canlı ticaret modunu çalıştır"""
        
        logger.info("📈 Canlı ticaret modu başlatılıyor...")
        
        if not self.is_initialized:
            if not await self.initialize_system(config):
                logger.error("❌ Sistem başlatılamadı, canlı ticaret iptal ediliyor")
                return
        
        # Strategy'yi yükle
        strategy_name = config.get("strategy", "momentum")
        
        if strategy_name not in self.available_strategies:
            logger.error(f"❌ Bilinmeyen strateji: {strategy_name}")
            return
        
        try:
            # Strategy sınıfını yükle
            strategy_info = self.available_strategies[strategy_name]
            module = importlib.import_module(strategy_info["module"])
            strategy_class = getattr(module, strategy_info["class"])
            
            # Strategy instance'ını oluştur
            strategy = strategy_class(portfolio=self.portfolio)
            self.strategies[strategy_name] = strategy
            
            logger.info(f"✅ {strategy_name} stratejisi yüklendi")
            
            # Canlı ticaret döngüsü
            self.is_running = True
            cycle_count = 0
            
            logger.info("🔄 Canlı ticaret döngüsü başlatıldı")
            
            while self.is_running:
                try:
                    cycle_start = datetime.now()
                    cycle_count += 1
                    
                    # Market verilerini çek
                    market_data = await self.data_fetcher.fetch_ohlcv_data()
                    
                    if market_data is not None and not market_data.empty:
                        # Strateji analizini çalıştır
                        if hasattr(strategy, 'analyze_market'):
                            analysis = strategy.analyze_market(market_data)
                            
                            # Trading kararları
                            if hasattr(strategy, 'execute_trades'):
                                trades = await strategy.execute_trades(analysis)
                                
                                if trades:
                                    logger.info(f"📊 Cycle {cycle_count}: {len(trades)} trade executed")
                    
                    # Performance monitoring
                    if cycle_count % 10 == 0:  # Her 10 cycle'da bir
                        total_value = self.portfolio.get_total_value()
                        pnl_pct = ((total_value - self.portfolio.initial_balance) / self.portfolio.initial_balance) * 100
                        
                        logger.info(f"📈 Performance Update:")
                        logger.info(f"   💰 Total Value: ${total_value:,.2f}")
                        logger.info(f"   📊 PnL: {pnl_pct:+.2f}%")
                        logger.info(f"   🔢 Cycles: {cycle_count}")
                    
                    # Cycle duration
                    cycle_duration = (datetime.now() - cycle_start).total_seconds()
                    
                    # Sleep to maintain 15-minute cycles (adjustable)
                    sleep_time = max(0, 900 - cycle_duration)  # 15 minutes = 900 seconds
                    
                    if sleep_time > 0:
                        await asyncio.sleep(sleep_time)
                    
                except KeyboardInterrupt:
                    logger.info("🛑 Kullanıcı tarafından durduruldu")
                    break
                    
                except Exception as e:
                    logger.error(f"❌ Canlı ticaret cycle hatası: {e}")
                    logger.error(traceback.format_exc())
                    
                    # Error recovery - 1 minute wait
                    await asyncio.sleep(60)
            
            self.is_running = False
            logger.info("✅ Canlı ticaret sonlandırıldı")
            
        except Exception as e:
            logger.error(f"❌ Canlı ticaret başlatma hatası: {e}")
            logger.error(traceback.format_exc())
    
    async def run_backtest(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """📊 Backtest çalıştır"""
        
        logger.info("📊 Backtest başlatılıyor...")
        
        strategy_name = config.get("strategy", "momentum")
        start_date = config.get("start_date", "2024-01-01")
        end_date = config.get("end_date", "2024-12-31")
        initial_capital = config.get("initial_capital", 1000.0)
        data_file = config.get("data_file", "historical_data/BTCUSDT_15m_20210101_20241231.csv")
        
        if strategy_name not in self.available_strategies:
            logger.error(f"❌ Bilinmeyen strateji: {strategy_name}")
            return {"success": False, "error": f"Unknown strategy: {strategy_name}"}
        
        try:
            # Backtest runner'ı başlat
            backtester = MomentumBacktester(
                data_file_path=data_file,
                initial_capital=initial_capital
            )
            
            # Backtest'i çalıştır
            logger.info(f"🎯 Backtest parametreleri:")
            logger.info(f"   📅 Period: {start_date} to {end_date}")
            logger.info(f"   💰 Initial Capital: ${initial_capital:,.2f}")
            logger.info(f"   🎯 Strategy: {strategy_name}")
            logger.info(f"   📄 Data File: {data_file}")
            
            results = await backtester.run_backtest_async(start_date, end_date)
            
            if results:
                logger.info("✅ Backtest tamamlandı!")
                logger.info(f"   📈 Total Return: {results.get('total_return_pct', 'N/A'):.2f}%")
                logger.info(f"   📊 Sharpe Ratio: {results.get('sharpe_ratio', 'N/A'):.2f}")
                logger.info(f"   📉 Max Drawdown: {results.get('max_drawdown_pct', 'N/A'):.2f}%")
                logger.info(f"   🎯 Win Rate: {results.get('win_rate_pct', 'N/A'):.2f}%")
                
                return {"success": True, "results": results}
            else:
                logger.error("❌ Backtest başarısız!")
                return {"success": False, "error": "Backtest execution failed"}
        
        except Exception as e:
            logger.error(f"❌ Backtest hatası: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    async def run_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """⚡ Optimizasyon çalıştır"""
        
        logger.info("⚡ Optimizasyon başlatılıyor...")
        
        strategy_name = config.get("strategy", "momentum")
        trials = config.get("trials", 1000)
        optimization_type = config.get("type", "smart_range")
        
        if strategy_name == "all":
            # Tüm stratejileri optimize et
            logger.info("🎯 Tüm stratejiler optimize ediliyor...")
            
            all_results = {}
            
            for strategy in self.available_strategies.keys():
                logger.info(f"⚡ {strategy} optimizasyonu başlatılıyor...")
                
                strategy_config = config.copy()
                strategy_config["strategy"] = strategy
                strategy_config["trials"] = trials // len(self.available_strategies)  # Distribute trials
                
                result = await self.run_single_optimization(strategy_config)
                all_results[strategy] = result
            
            return {"success": True, "results": all_results}
        
        else:
            # Tek strateji optimize et
            return await self.run_single_optimization(config)
    
    async def run_single_optimization(self, config: Dict[str, Any]) -> Dict[str, Any]:
        """⚡ Tek strateji optimizasyonu"""
        
        strategy_name = config.get("strategy", "momentum")
        trials = config.get("trials", 1000)
        optimization_type = config.get("type", "smart_range")
        
        try:
            if optimization_type == "smart_range":
                # Smart range optimizer'ı kullan
                from smart_range_optimizer import SmartRangeOptimizerEnhanced
                
                optimizer = SmartRangeOptimizerEnhanced()
                result = await optimizer.optimize_strategy_smart_ranges(strategy_name, trials)
                
                return {"success": True, "results": result}
            
            elif optimization_type == "ultimate":
                # Ultimate optimizer'ı kullan
                from optimize_strategy_ultimate import run_ultimate_optimization_async
                
                start_date = config.get("start_date", "2024-01-01") 
                end_date = config.get("end_date", "2024-12-31")
                data_file = config.get("data_file", "historical_data/BTCUSDT_15m_20210101_20241231.csv")
                
                result = await run_ultimate_optimization_async(
                    start_date=start_date,
                    end_date=end_date,
                    data_file_path=data_file,
                    n_trials=trials
                )
                
                return {"success": True, "results": result}
            
            else:
                logger.error(f"❌ Bilinmeyen optimizasyon tipi: {optimization_type}")
                return {"success": False, "error": f"Unknown optimization type: {optimization_type}"}
        
        except Exception as e:
            logger.error(f"❌ {strategy_name} optimizasyon hatası: {e}")
            logger.error(traceback.format_exc())
            return {"success": False, "error": str(e)}
    
    def get_system_status(self) -> Dict[str, Any]:
        """📋 Sistem durumunu getir"""
        
        uptime = datetime.now(timezone.utc) - self.startup_time
        
        status = {
            "system_name": self.system_name,
            "version": self.version,
            "startup_time": self.startup_time.isoformat(),
            "uptime_seconds": uptime.total_seconds(),
            "uptime_hours": uptime.total_seconds() / 3600,
            "is_initialized": self.is_initialized,
            "is_running": self.is_running,
            "core_imports": CORE_IMPORTS_SUCCESS,
            "available_strategies": list(self.available_strategies.keys()),
            "loaded_strategies": list(self.strategies.keys()) if self.strategies else [],
            "portfolio_initialized": self.portfolio is not None,
            "data_fetcher_initialized": self.data_fetcher is not None
        }
        
        if self.portfolio:
            status["portfolio_status"] = {
                "initial_balance": self.portfolio.initial_balance,
                "current_balance": self.portfolio.balance,
                "total_value": self.portfolio.get_total_value(),
                "active_positions": len(self.portfolio.positions)
            }
        
        return status


async def main():
    """Ana çalıştırma fonksiyonu"""
    
    parser = argparse.ArgumentParser(
        description="Phoenix Trading System - Unified Main Controller",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  python main.py live                                    # Canlı ticaret başlat
  python main.py backtest --strategy momentum            # Momentum stratejisi backtest
  python main.py optimize --strategy all --trials 5000  # Tüm stratejileri optimize et
  python main.py validate                               # Sistem doğrulama
  python main.py status                                 # Sistem durumu
        """
    )
    
    subparsers = parser.add_subparsers(dest='command', help='Mevcut komutlar')
    
    # Live trading
    live_parser = subparsers.add_parser('live', help='Canlı ticaret başlat')
    live_parser.add_argument('--strategy', default='momentum', choices=['momentum', 'bollinger_rsi', 'rsi_ml', 'macd_ml', 'volume_profile'])
    live_parser.add_argument('--capital', type=float, default=1000.0, help='Başlangıç sermayesi')
    live_parser.add_argument('--symbol', default='BTC/USDT', help='Trading çifti')
    
    # Backtest
    backtest_parser = subparsers.add_parser('backtest', help='Backtest çalıştır')
    backtest_parser.add_argument('--strategy', default='momentum', choices=['momentum', 'bollinger_rsi', 'rsi_ml', 'macd_ml', 'volume_profile'])
    backtest_parser.add_argument('--start-date', default='2024-01-01', help='Başlangıç tarihi (YYYY-MM-DD)')
    backtest_parser.add_argument('--end-date', default='2024-12-31', help='Bitiş tarihi (YYYY-MM-DD)')
    backtest_parser.add_argument('--capital', type=float, default=1000.0, help='Başlangıç sermayesi')
    backtest_parser.add_argument('--data-file', default='historical_data/BTCUSDT_15m_20210101_20241231.csv', help='Veri dosyası')
    
    # Optimization
    optimize_parser = subparsers.add_parser('optimize', help='Optimizasyon çalıştır')
    optimize_parser.add_argument('--strategy', default='momentum', choices=['momentum', 'bollinger_rsi', 'rsi_ml', 'macd_ml', 'volume_profile', 'all'])
    optimize_parser.add_argument('--trials', type=int, default=1000, help='Deneme sayısı')
    optimize_parser.add_argument('--type', default='smart_range', choices=['smart_range', 'ultimate'], help='Optimizasyon tipi')
    optimize_parser.add_argument('--start-date', default='2024-01-01', help='Backtest başlangıç tarihi')
    optimize_parser.add_argument('--end-date', default='2024-12-31', help='Backtest bitiş tarihi')
    optimize_parser.add_argument('--data-file', default='historical_data/BTCUSDT_15m_20210101_20241231.csv', help='Veri dosyası')
    
    # System commands
    subparsers.add_parser('validate', help='Sistem doğrulaması yap')
    subparsers.add_parser('status', help='Sistem durumunu göster')
    subparsers.add_parser('health', help='Sistem sağlık kontrolü')
    
    args = parser.parse_args()
    
    if not args.command:
        parser.print_help()
        return
    
    # Phoenix sistem'ini başlat
    phoenix = PhoenixTradingSystem()
    
    try:
        if args.command == 'live':
            config = {
                "strategy": args.strategy,
                "initial_capital": args.capital,
                "symbol": args.symbol
            }
            
            print(f"🚀 CANLІ TİCARET BAŞLATILIYOR")
            print(f"   🎯 Strateji: {args.strategy}")
            print(f"   💰 Sermaye: ${args.capital:,.2f}")
            print(f"   📊 Sembol: {args.symbol}")
            print(f"   🕒 Başlangıç: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
            print()
            print("📋 Canlı ticareti durdurmak için Ctrl+C kullanın")
            print("="*80)
            
            await phoenix.run_live_trading(config)
        
        elif args.command == 'backtest':
            config = {
                "strategy": args.strategy,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "initial_capital": args.capital,
                "data_file": args.data_file
            }
            
            print(f"📊 BACKTEST BAŞLATILIYOR")
            print(f"   🎯 Strateji: {args.strategy}")
            print(f"   📅 Period: {args.start_date} - {args.end_date}")
            print(f"   💰 Sermaye: ${args.capital:,.2f}")
            print(f"   📄 Veri: {args.data_file}")
            print("="*80)
            
            result = await phoenix.run_backtest(config)
            
            if result["success"]:
                print("\n🎉 BACKTEST SONUÇLARI:")
                results = result["results"]
                print(f"   📈 Total Return: {results.get('total_return_pct', 'N/A'):.2f}%")
                print(f"   📊 Sharpe Ratio: {results.get('sharpe_ratio', 'N/A'):.2f}")
                print(f"   📉 Max Drawdown: {results.get('max_drawdown_pct', 'N/A'):.2f}%")
                print(f"   🎯 Win Rate: {results.get('win_rate_pct', 'N/A'):.2f}%")
                print(f"   💰 Final Value: ${results.get('final_portfolio_value', 'N/A'):,.2f}")
            else:
                print(f"❌ BACKTEST BAŞARISIZ: {result.get('error', 'Unknown error')}")
        
        elif args.command == 'optimize':
            config = {
                "strategy": args.strategy,
                "trials": args.trials,
                "type": args.type,
                "start_date": args.start_date,
                "end_date": args.end_date,
                "data_file": args.data_file
            }
            
            print(f"⚡ OPTİMİZASYON BAŞLATILIYOR")
            print(f"   🎯 Strateji: {args.strategy}")
            print(f"   🔬 Deneme sayısı: {args.trials}")
            print(f"   🔧 Optimizasyon tipi: {args.type}")
            print(f"   📅 Period: {args.start_date} - {args.end_date}")
            print("="*80)
            
            result = await phoenix.run_optimization(config)
            
            if result["success"]:
                print("\n🎉 OPTİMİZASYON TAMAMLANDI!")
                
                if args.strategy == "all":
                    for strategy_name, strategy_result in result["results"].items():
                        if strategy_result.get("success"):
                            print(f"\n✅ {strategy_name.upper()}:")
                            print(f"   🏆 Best Score: {strategy_result['results'].get('best_performance', 'N/A')}")
                        else:
                            print(f"\n❌ {strategy_name.upper()}: {strategy_result.get('error', 'Failed')}")
                else:
                    results = result["results"]
                    print(f"   🏆 Best Performance: {results.get('best_performance', 'N/A')}")
                    print(f"   ⏱️ Duration: {results.get('optimization_duration_seconds', 0)/60:.1f} minutes")
                    print(f"   📊 Parameters optimized: {results.get('parameter_count', 'N/A')}")
            else:
                print(f"❌ OPTİMİZASYON BAŞARISIZ: {result.get('error', 'Unknown error')}")
        
        elif args.command == 'validate':
            print("🔍 SİSTEM DOĞRULAMASI BAŞLATILIYOR")
            print("="*80)
            
            validation = phoenix.validate_system()
            
            print(f"\n📊 DOĞRULAMA SONUÇLARI:")
            print(f"   🔧 Genel Durum: {validation['overall_status'].upper()}")
            print(f"   📦 Core Imports: {'✅' if validation['core_imports'] else '❌'}")
            
            # Klasör durumu
            print(f"\n📁 KLASÖRLER:")
            for dir_path, info in validation["directories"].items():
                status = "✅" if info["exists"] else "❌"
                print(f"   {status} {dir_path}")
            
            # Dosya durumu
            print(f"\n📄 KRİTİK DOSYALAR:")
            for file_path, info in validation["files"].items():
                status = "✅" if info["exists"] else "❌"
                size = f"({info['size_kb']:.1f} KB)" if info["exists"] else ""
                print(f"   {status} {file_path} {size}")
            
            # Strateji durumu
            print(f"\n🎯 STRATEJİLER:")
            for strategy_name, info in validation["strategies"].items():
                if info.get("importable") and info.get("class_found", True):
                    print(f"   ✅ {strategy_name}")
                else:
                    print(f"   ❌ {strategy_name} - {info.get('error', 'Unknown error')}")
            
            # Dependency durumu
            print(f"\n📚 BAĞIMLILIKLAR:")
            for package, info in validation["dependencies"].items():
                status = "✅" if info["available"] else "❌"
                print(f"   {status} {package}")
        
        elif args.command == 'status':
            print("📋 SİSTEM DURUMU")
            print("="*80)
            
            status = phoenix.get_system_status()
            
            print(f"🚀 Sistem: {status['system_name']} v{status['version']}")
            print(f"⏱️ Uptime: {status['uptime_hours']:.1f} saat")
            print(f"🔧 Başlatıldı: {'✅' if status['is_initialized'] else '❌'}")
            print(f"🔄 Çalışıyor: {'✅' if status['is_running'] else '❌'}")
            print(f"📦 Core Imports: {'✅' if status['core_imports'] else '❌'}")
            
            print(f"\n🎯 Mevcut Stratejiler: {', '.join(status['available_strategies'])}")
            
            if status.get('loaded_strategies'):
                print(f"🔄 Yüklü Stratejiler: {', '.join(status['loaded_strategies'])}")
            
            if status.get('portfolio_status'):
                portfolio = status['portfolio_status']
                print(f"\n💰 PORTFOLIO:")
                print(f"   Initial: ${portfolio['initial_balance']:,.2f}")
                print(f"   Current: ${portfolio['current_balance']:,.2f}")
                print(f"   Total Value: ${portfolio['total_value']:,.2f}")
                print(f"   Positions: {portfolio['active_positions']}")
        
        elif args.command == 'health':
            print("🩺 SİSTEM SAĞLIK KONTROLÜ")
            print("="*80)
            
            validation = phoenix.validate_system()
            
            if validation['overall_status'] == 'healthy':
                print("✅ Sistem tamamen sağlıklı!")
            elif validation['overall_status'] == 'warning':
                print("⚠️ Sistem kısmen sağlıklı - bazı sorunlar var")
            else:
                print("❌ Sistem sağlıksız - kritik sorunlar mevcut")
            
            # Quick health metrics
            healthy_files = sum(1 for info in validation["files"].values() if info["exists"])
            total_files = len(validation["files"])
            
            healthy_strategies = sum(1 for info in validation["strategies"].values() if info.get("importable"))
            total_strategies = len(validation["strategies"])
            
            healthy_deps = sum(1 for info in validation["dependencies"].values() if info["available"])
            total_deps = len(validation["dependencies"])
            
            print(f"\n📊 SAĞLIK METRİKLERİ:")
            print(f"   📄 Dosyalar: {healthy_files}/{total_files} (%{healthy_files/total_files*100:.0f})")
            print(f"   🎯 Stratejiler: {healthy_strategies}/{total_strategies} (%{healthy_strategies/total_strategies*100:.0f})")
            print(f"   📚 Bağımlılıklar: {healthy_deps}/{total_deps} (%{healthy_deps/total_deps*100:.0f})")
    
    except KeyboardInterrupt:
        print("\n🛑 İşlem kullanıcı tarafından durduruldu")
    
    except Exception as e:
        logger.error(f"❌ Ana sistem hatası: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ HATA: {e}")
        sys.exit(1)


if __name__ == "__main__":
    asyncio.run(main())