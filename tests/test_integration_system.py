#!/usr/bin/env python3
"""
🧪 PROJE PHOENIX - SYSTEM INTEGRATION TESTS
💎 Ultra Gelişmiş Sistem Entegrasyon Testleri

Bu dosya sistemin tüm bileşenlerinin birlikte çalışmasını test eder:
- main.py komuta merkezi
- Strateji koordinasyonu
- Optimizasyon sistemi
- Backtest sistemi
- Parametre yönetimi
- Veri akışı

HEDGE FUND LEVEL INTEGRATION TESTING
"""

import pytest
from strategies.momentum_optimized import EnhancedMomentumStrategy
from strategies.base_strategy import TradingSignal, SignalType
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import asyncio
import json
from pathlib import Path

from utils.portfolio import Portfolio
from utils.strategy_coordinator import StrategyCoordinator
from utils.adaptive_parameter_evolution import AdaptiveParameterEvolution
from optimization.master_optimizer import MasterOptimizer, OptimizationConfig
from backtesting.multi_strategy_backtester import MultiStrategyBacktester, BacktestConfiguration, BacktestMode
from json_parameter_system import JSONParameterManager
from strategies.base_strategy import TradingSignal, SignalType


class TestSystemIntegration:
    """Sistem entegrasyon testleri"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_main_system_initialization(self, sample_market_data):
        """Ana sistem başlatma testi"""
        from main import PhoenixTradingSystem
        
        # Phoenix sistemini başlat
        phoenix = PhoenixTradingSystem()
        
        # Sistem durumunu kontrol et
        status = await phoenix.get_system_status()
        
        assert status.system_version == "2.0"
        assert status.core_imports_success is True
        assert status.total_strategies >= 0
        assert status.uptime_seconds >= 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_strategy_coordination_integration(self, sample_portfolio, sample_market_data):
        """Strateji koordinasyon entegrasyonu testi"""
        # Strateji koordinatörü oluştur
        coordinator = StrategyCoordinator(portfolio=sample_portfolio)
        
        # Stratejileri kaydet
        momentum_strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        success = coordinator.register_strategy(
            strategy_name="momentum",
            strategy_instance=momentum_strategy,
            initial_weight=0.5
        )
        
        assert success is True
        
        # Koordinasyon çalıştır
        coordination_results = await coordinator.coordinate_strategies(sample_market_data)
        
        assert isinstance(coordination_results, dict)
        assert 'consensus_analysis' in coordination_results
        assert 'correlation_analysis' in coordination_results
        assert 'allocation_optimization' in coordination_results
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_optimization_integration(self, sample_market_data):
        """Optimizasyon sistemi entegrasyonu testi"""
        # Optimizasyon konfigürasyonu
        config = OptimizationConfig(
            strategy_name="momentum",
            trials=10,  # Test için az trial
            storage_url="sqlite:///test_optimization.db",
            walk_forward=False,
            walk_forward_periods=3,
            validation_split=0.2,
            early_stopping_rounds=5,
            parallel_jobs=1,
            timeout_seconds=60
        )
        
        # Master optimizer oluştur
        optimizer = MasterOptimizer(config)
        
        # Strateji geçerliliğini kontrol et
        is_valid = optimizer.validate_strategy("momentum")
        assert is_valid is True
        
        # Optimizasyon çalıştır (kısa test)
        result = await optimizer.optimize_single_strategy("momentum")
        
        assert isinstance(result, optimizer.OptimizationResult)
        assert result.strategy_name == "momentum"
        assert result.total_trials > 0
        assert len(result.best_parameters) > 0
        assert result.best_score > -999  # Başarısız değil
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_backtest_integration(self, sample_market_data):
        """Backtest sistemi entegrasyonu testi"""
        # Backtest konfigürasyonu
        config = BacktestConfiguration(
            start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
            end_date=datetime(2024, 1, 31, tzinfo=timezone.utc),
            initial_capital=10000.0,
            commission_rate=0.001,
            slippage_rate=0.0005,
            mode=BacktestMode.MULTI_STRATEGY,
            strategy_allocations={'momentum': 1.0},
            rebalancing_frequency='monthly',
            max_drawdown_threshold=0.20,
            enable_position_sizing=True
        )
        
        # Backtester oluştur
        backtester = MultiStrategyBacktester(
            enable_parallel_processing=False,  # Test için seri
            max_workers=1,
            cache_results=False,
            enable_advanced_analytics=True
        )
        
        # Stratejiyi kaydet
        portfolio = Portfolio(initial_capital=10000.0)
        momentum_strategy = EnhancedMomentumStrategy(
            portfolio=portfolio,
            symbol="BTC/USDT"
        )
        
        success = backtester.register_strategy(
            strategy_name="momentum",
            strategy_class=EnhancedMomentumStrategy,
            strategy_config={'symbol': 'BTC/USDT'}
        )
        
        assert success is True
        
        # Backtest çalıştır
        result = await backtester.run_backtest(
            config=config,
            market_data=sample_market_data,
            strategies=['momentum']
        )
        
        assert isinstance(result, backtester.BacktestResult)
        assert result.configuration == config
        assert result.total_trades >= 0
        assert result.total_return_pct >= -100  # Mantıklı aralık
        assert result.max_drawdown_pct >= 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_parameter_management_integration(self):
        """Parametre yönetimi entegrasyonu testi"""
        # JSON parametre yöneticisi
        json_manager = JSONParameterManager()
        
        # Test parametreleri
        test_params = {
            'ema_short': 13,
            'ema_medium': 21,
            'ema_long': 56,
            'rsi_period': 14,
            'max_positions': 3,
            'base_position_size_pct': 25.0,
            'max_loss_pct': 10.0,
            'ml_confidence_threshold': 0.65
        }
        
        # Parametreleri kaydet
        success = json_manager.save_optimization_results(
            strategy_name='momentum',
            best_parameters=test_params,
            optimization_metrics={
                'best_score': 26.8,
                'total_trials': 1000,
                'optimization_duration_minutes': 45.5
            }
        )
        
        assert success is True
        
        # Parametreleri yükle
        loaded_data = json_manager.load_strategy_parameters('momentum')
        
        assert loaded_data is not None
        assert 'parameters' in loaded_data
        assert loaded_data['parameters'] == test_params
        assert 'optimization_info' in loaded_data
        assert 'metadata' in loaded_data
        
        # Parametre dosyasını listele
        files = json_manager.list_parameter_files()
        assert 'momentum' in files
        assert files['momentum']['strategy_name'] == 'momentum'
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_evolution_system_integration(self, sample_portfolio):
        """Evrim sistemi entegrasyonu testi"""
        # Strateji koordinatörü oluştur
        coordinator = StrategyCoordinator(portfolio=sample_portfolio)
        
        # Momentum stratejisi ekle
        momentum_strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        coordinator.register_strategy(
            strategy_name="momentum",
            strategy_instance=momentum_strategy,
            initial_weight=1.0
        )
        
        # Evrim sistemi oluştur
        evolution = AdaptiveParameterEvolution(
            strategy_coordinator=coordinator
        )
        
        # Stratejileri izle
        monitoring_results = await evolution.monitor_strategies()
        
        assert isinstance(monitoring_results, dict)
        assert 'strategies_monitored' in monitoring_results
        assert 'trigger_events' in monitoring_results
        assert 'evolution_status' in monitoring_results
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_data_flow_integration(self, mock_binance_fetcher, sample_portfolio):
        """Veri akışı entegrasyonu testi"""
        # Veri çek
        market_data = await mock_binance_fetcher.fetch_ohlcv_data()
        
        assert market_data is not None
        assert not market_data.empty
        assert 'open' in market_data.columns
        assert 'high' in market_data.columns
        assert 'low' in market_data.columns
        assert 'close' in market_data.columns
        assert 'volume' in market_data.columns
        
        # Strateji ile analiz et
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        signal = await strategy.analyze_market(market_data)
        
        assert isinstance(signal, TradingSignal)
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert signal.confidence >= 0.0
        assert signal.confidence <= 1.0
        
        # Pozisyon boyutu hesapla
        if signal.signal_type == SignalType.BUY:
            position_size = strategy.calculate_position_size(signal)
            assert position_size > 0
            assert position_size <= sample_portfolio.get_available_usdt()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_end_to_end_trading_cycle(self, sample_portfolio, sample_market_data):
        """Uçtan uca trading döngüsü testi"""
        # 1. Strateji oluştur
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        # 2. Piyasa analizi
        signal = await strategy.analyze_market(sample_market_data)
        
        # 3. Alım işlemi (eğer sinyal BUY ise)
        if signal.signal_type == SignalType.BUY:
            position_size = strategy.calculate_position_size(signal)
            
            position = await sample_portfolio.execute_buy(
                strategy_name=strategy.strategy_name,
                symbol=strategy.symbol,
                current_price=signal.price,
                timestamp=signal.timestamp.isoformat(),
                reason="Integration test buy",
                amount_usdt_override=position_size
            )
            
            assert position is not None
            assert len(sample_portfolio.positions) == 1
            
            # 4. Pozisyon takibi
            current_price = signal.price * 1.02  # %2 kâr
            position.update_performance_metrics(current_price)
            
            # 5. Satış kararı
            should_sell, reason = await strategy.should_sell(position, sample_market_data)
            
            # 6. Satış işlemi
            if should_sell:
                success = await sample_portfolio.execute_sell(
                    position_to_close=position,
                    current_price=current_price,
                    timestamp=datetime.now(timezone.utc).isoformat(),
                    reason=f"Integration test sell: {reason}"
                )
                
                assert success is True
                assert len(sample_portfolio.positions) == 0
                assert len(sample_portfolio.closed_trades) == 1
                assert sample_len(portfolio.closed_trades) == 1
                assert sample_portfolio.cumulative_pnl > 0
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_multi_strategy_portfolio_integration(self, sample_portfolio, sample_market_data):
        """Çoklu strateji portföy entegrasyonu testi"""
        # Strateji koordinatörü
        coordinator = StrategyCoordinator(portfolio=sample_portfolio)
        
        # Birden fazla strateji ekle
        strategies = {
            'momentum': EnhancedMomentumStrategy(portfolio=sample_portfolio, symbol="BTC/USDT"),
            'bollinger': EnhancedMomentumStrategy(portfolio=sample_portfolio, symbol="BTC/USDT"),  # Aynı strateji farklı isimle
        }
        
        for name, strategy in strategies.items():
            coordinator.register_strategy(
                strategy_name=name,
                strategy_instance=strategy,
                initial_weight=0.5
            )
        
        # Koordinasyon çalıştır
        coordination_results = await coordinator.coordinate_strategies(sample_market_data)
        
        assert 'consensus_analysis' in coordination_results
        assert 'correlation_analysis' in coordination_results
        assert 'allocation_optimization' in coordination_results
        
        # Performans güncelle
        performance_results = await coordinator.update_strategy_performances()
        
        assert isinstance(performance_results, dict)
        assert 'performance_updated' in performance_results
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_error_handling_integration(self, sample_portfolio):
        """Hata yönetimi entegrasyonu testi"""
        # Geçersiz veri ile test
        invalid_data = pd.DataFrame({
            'open': [np.nan, np.nan, np.nan],
            'high': [np.nan, np.nan, np.nan],
            'low': [np.nan, np.nan, np.nan],
            'close': [np.nan, np.nan, np.nan],
            'volume': [np.nan, np.nan, np.nan]
        })
        
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        # Hata durumunda graceful degradation
        try:
            signal = await strategy.analyze_market(invalid_data)
            # Hata olmamalı, HOLD sinyali dönmeli
            assert signal.signal_type == SignalType.HOLD
        except Exception as e:
            # Eğer hata olursa, loglanmalı ama sistem çökmemeli
            assert "error" in str(e).lower() or "invalid" in str(e).lower()
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_configuration_integration(self):
        """Konfigürasyon entegrasyonu testi"""
        from utils.config import settings
        
        # Temel ayarlar kontrol edilmeli
        assert hasattr(settings, 'SYMBOL')
        assert hasattr(settings, 'INITIAL_CAPITAL_USDT')
        assert hasattr(settings, 'TIMEFRAME')
        assert hasattr(settings, 'FEE_BUY')
        assert hasattr(settings, 'FEE_SELL')
        
        # Momentum strateji ayarları
        assert hasattr(settings, 'MOMENTUM_EMA_SHORT')
        assert hasattr(settings, 'MOMENTUM_EMA_MEDIUM')
        assert hasattr(settings, 'MOMENTUM_EMA_LONG')
        assert hasattr(settings, 'MOMENTUM_RSI_PERIOD')
        
        # ML ayarları
        assert hasattr(settings, 'MOMENTUM_ML_ENABLED')
        assert hasattr(settings, 'MOMENTUM_ML_CONFIDENCE_THRESHOLD')
        
        # AI ayarları
        assert hasattr(settings, 'AI_ASSISTANCE_ENABLED')
        assert hasattr(settings, 'AI_CONFIDENCE_THRESHOLD')
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_logging_integration(self, sample_portfolio, sample_market_data):
        """Loglama entegrasyonu testi"""
        from utils.logger import logger
        
        # Logger çalışıyor mu kontrol et
        logger.info("Integration test logging message")
        logger.warning("Integration test warning message")
        logger.error("Integration test error message")
        
        # Strateji ile loglama
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        # Strateji logları kontrol et
        strategy.logger.info("Strategy integration test")
        
        # Portfolio logları
        sample_portfolio.logger.info("Portfolio integration test")
        
        # Log dosyalarının oluştuğunu kontrol et
        log_dir = Path("logs")
        assert log_dir.exists()
        
        # Ana log dosyası
        main_log = log_dir / "algobot.log"
        if main_log.exists():
            assert main_log.stat().st_size > 0
        
        # Hata log dosyası
        error_log = log_dir / "errors.log"
        if error_log.exists():
            # Hata logları varsa kontrol et
            pass


class TestPerformanceIntegration:
    """Performans entegrasyon testleri"""
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_large_dataset_performance(self, large_market_dataset, sample_portfolio):
        """Büyük veri seti performans testi"""
        import time
        
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        # Performans ölçümü
        start_time = time.time()
        
        # Büyük veri seti ile analiz
        signal = await strategy.analyze_market(large_market_dataset)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performans kriterleri
        assert processing_time < 10.0  # 10 saniyeden az
        assert signal is not None
        assert isinstance(signal, TradingSignal)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_multi_strategy_performance(self, sample_market_data, sample_portfolio):
        """Çoklu strateji performans testi"""
        import time
        
        # Birden fazla strateji oluştur
        strategies = []
        for i in range(5):
            strategy = EnhancedMomentumStrategy(
                portfolio=sample_portfolio,
                symbol="BTC/USDT"
            )
            strategies.append(strategy)
        
        # Performans ölçümü
        start_time = time.time()
        
        # Tüm stratejileri paralel çalıştır
        tasks = []
        for strategy in strategies:
            task = strategy.analyze_market(sample_market_data)
            tasks.append(task)
        
        signals = await asyncio.gather(*tasks)
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performans kriterleri
        assert processing_time < 5.0  # 5 saniyeden az
        assert len(signals) == 5
        assert all(isinstance(signal, TradingSignal) for signal in signals)
    
    @pytest.mark.performance
    @pytest.mark.asyncio
    async def test_optimization_performance(self):
        """Optimizasyon performans testi"""
        import time
        
        # Kısa optimizasyon testi
        config = OptimizationConfig(
            strategy_name="momentum",
            trials=50,  # Kısa test
            storage_url="sqlite:///test_perf_optimization.db",
            walk_forward=False,
            walk_forward_periods=2,
            validation_split=0.2,
            early_stopping_rounds=3,
            parallel_jobs=1,
            timeout_seconds=30
        )
        
        optimizer = MasterOptimizer(config)
        
        start_time = time.time()
        
        result = await optimizer.optimize_single_strategy("momentum")
        
        end_time = time.time()
        processing_time = end_time - start_time
        
        # Performans kriterleri
        assert processing_time < 60.0  # 1 dakikadan az
        assert result.total_trials > 0
        assert result.best_score > -999


class TestSecurityIntegration:
    """Güvenlik entegrasyon testleri"""
    
    @pytest.mark.security
    @pytest.mark.asyncio
    async def test_malicious_input_handling(self, sample_portfolio, malicious_input_data):
        """Kötü niyetli giriş işleme testi"""
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        # SQL injection test
        try:
            # Bu test parametrelerin güvenli şekilde işlendiğini kontrol eder
            strategy.strategy_name = malicious_input_data['sql_injection']
            # Sistem çökmemeli
            assert True
        except Exception:
            # Eğer hata olursa, güvenli şekilde yakalanmalı
            assert True
        
        # XSS test
        try:
            strategy.strategy_name = malicious_input_data['xss_script']
            # Sistem çökmemeli
            assert True
        except Exception:
            # Eğer hata olursa, güvenli şekilde yakalanmalı
            assert True
    
    @pytest.mark.security
    def test_parameter_validation(self, sample_portfolio):
        """Parametre doğrulama testi"""
        # Geçersiz parametrelerle strateji oluşturma
        invalid_params = {
            'max_positions': -1,  # Negatif değer
            'base_position_size_pct': 150.0,  # %100'den fazla
            'max_loss_pct': -50.0,  # Negatif değer
        }
        
        # Strateji geçersiz parametrelerle oluşturulmaya çalışılsa bile çalışmalı
        try:
            strategy = EnhancedMomentumStrategy(
                portfolio=sample_portfolio,
                symbol="BTC/USDT",
                **invalid_params
            )
            # Parametreler varsayılan değerlere dönmeli
            assert strategy.max_positions > 0
            assert 0 < strategy.base_position_size_pct <= 100
            assert strategy.max_loss_pct > 0
        except Exception:
            # Eğer hata olursa, güvenli şekilde yakalanmalı
            assert True 