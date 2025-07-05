#!/usr/bin/env python3
"""
🧪 PROJE PHOENIX - STRATEGIES UNIT TESTS
💎 Ultra Gelişmiş Strateji Test Sistemi

Bu dosya strateji sınıflarının tüm fonksiyonlarını test eder:
- BaseStrategy temel fonksiyonları
- Momentum stratejisi
- Diğer stratejiler
- Sinyal üretimi
- Pozisyon boyutlandırma
- Çıkış kararları

HEDGE FUND LEVEL STRATEGY TESTING
"""

import pytest
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import asyncio

from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, VolatilityRegime, GlobalMarketRegime
from strategies.momentum_optimized import EnhancedMomentumStrategy
from utils.portfolio import Portfolio
from json_parameter_system import JSONParameterManager


class TestBaseStrategy:
    """BaseStrategy sınıfı unit testleri"""
    
    @pytest.fixture
    def sample_strategy(self, sample_portfolio):
        """Örnek strateji oluştur"""
        class TestStrategy(BaseStrategy):
            async def analyze_market(self, data: pd.DataFrame) -> TradingSignal:
                # Basit test sinyali
                return TradingSignal(
                    signal_type=SignalType.BUY,
                    confidence=0.7,
                    price=data['close'].iloc[-1],
                    timestamp=datetime.now(timezone.utc),
                    reasons=['Test signal']
                )
            
            def calculate_position_size(self, signal: TradingSignal) -> float:
                return 100.0  # Sabit pozisyon boyutu
        
        return TestStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT",
            strategy_name="TestStrategy"
        )
    
    @pytest.mark.unit
    def test_base_strategy_initialization(self, sample_portfolio):
        """BaseStrategy başlatma testi"""
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        assert strategy.portfolio == sample_portfolio
        assert strategy.symbol == "BTC/USDT"
        assert strategy.strategy_name == "EnhancedMomentum"
        assert strategy.state.value == "initializing"
        assert strategy.ml_enabled is True
        assert strategy.dynamic_exit_enabled is True
        assert strategy.kelly_enabled is True
        assert strategy.global_intelligence_enabled is True
    
    @pytest.mark.unit
    def test_volatility_regime_detection(self, sample_strategy, sample_market_data):
        """Volatilite rejimi tespiti testi"""
        # Düşük volatilite verisi oluştur
        low_vol_data = sample_market_data.copy()
        low_vol_data['close'] = low_vol_data['close'] * (1 + np.random.normal(0, 0.005, len(low_vol_data)))
        
        regime = sample_strategy._detect_volatility_regime(low_vol_data)
        assert isinstance(regime, VolatilityRegime)
        assert regime.regime_name in ['ultra_low', 'low', 'normal']
        
        # Yüksek volatilite verisi oluştur
        high_vol_data = sample_market_data.copy()
        high_vol_data['close'] = high_vol_data['close'] * (1 + np.random.normal(0, 0.05, len(high_vol_data)))
        
        regime = sample_strategy._detect_volatility_regime(high_vol_data)
        assert regime.regime_name in ['high', 'extreme']
    
    @pytest.mark.unit
    def test_dynamic_exit_timing(self, sample_strategy, sample_market_data, sample_position):
        """Dinamik çıkış zamanlaması testi"""
        # Pozisyon yaşını ayarla
        sample_position.timestamp = (datetime.now(timezone.utc) - timedelta(minutes=30)).isoformat()
        
        exit_decision = sample_strategy.calculate_dynamic_exit_timing(
            sample_market_data, sample_position
        )
        
        assert isinstance(exit_decision, sample_strategy.DynamicExitDecision)
        assert exit_decision.total_planned_time > 0
        assert exit_decision.volatility_regime in ['ultra_low', 'low', 'normal', 'high', 'extreme']
        assert 0.0 <= exit_decision.decision_confidence <= 1.0
    
    @pytest.mark.unit
    def test_kelly_position_sizing(self, sample_strategy, sample_trading_signal):
        """Kelly pozisyon boyutlandırma testi"""
        kelly_result = sample_strategy.calculate_kelly_position_size(
            sample_trading_signal
        )
        
        assert isinstance(kelly_result, sample_strategy.KellyPositionResult)
        assert kelly_result.position_size_usdt > 0
        assert 0.0 <= kelly_result.kelly_percentage <= 1.0
        assert 0.0 <= kelly_result.sizing_confidence <= 1.0
        assert len(kelly_result.recommendations) >= 0
    
    @pytest.mark.unit
    def test_global_market_analysis(self, sample_strategy):
        """Global piyasa analizi testi"""
        # Mock global market data
        global_data = {
            'BTC': pd.DataFrame({
                'close': [50000] * 100,
                'volume': [1000] * 100
            }),
            'SPY': pd.DataFrame({
                'close': [400] * 100,
                'volume': [1000000] * 100
            }),
            'DXY': pd.DataFrame({
                'close': [100] * 100,
                'volume': [100000] * 100
            })
        }
        
        analysis = sample_strategy._analyze_global_market_risk(global_data)
        
        assert isinstance(analysis, sample_strategy.GlobalMarketAnalysis)
        assert isinstance(analysis.market_regime, GlobalMarketRegime)
        assert 0.0 <= analysis.regime_confidence <= 1.0
        assert 0.0 <= analysis.risk_score <= 1.0
        assert -1.0 <= analysis.btc_spy_correlation <= 1.0
        assert -1.0 <= analysis.btc_dxy_correlation <= 1.0
        assert -1.0 <= analysis.btc_vix_correlation <= 1.0
        assert analysis.position_size_adjustment > 0
    
    @pytest.mark.unit
    def test_technical_indicators(self, sample_market_data):
        """Teknik indikatör hesaplama testi"""
        indicators = BaseStrategy.calculate_technical_indicators(sample_market_data)
        
        required_indicators = ['rsi', 'ema_short', 'ema_medium', 'ema_long', 'atr']
        
        for indicator in required_indicators:
            assert indicator in indicators
            assert isinstance(indicators[indicator], pd.Series)
            assert len(indicators[indicator]) == len(sample_market_data)
            assert not indicators[indicator].isna().all()  # Tamamen NaN olmamalı
    
    @pytest.mark.unit
    def test_signal_creation(self):
        """Sinyal oluşturma testi"""
        signal = BaseStrategy.create_signal(
            signal_type=SignalType.BUY,
            confidence=0.8,
            price=50000.0,
            reasons=['Strong momentum', 'RSI oversold'],
            metadata={'quality_score': 15}
        )
        
        assert isinstance(signal, TradingSignal)
        assert signal.signal_type == SignalType.BUY
        assert signal.confidence == 0.8
        assert signal.price == 50000.0
        assert len(signal.reasons) == 2
        assert signal.metadata['quality_score'] == 15
    
    @pytest.mark.unit
    def test_strategy_analytics(self, sample_strategy):
        """Strateji analitikleri testi"""
        analytics = sample_strategy.get_strategy_analytics()
        
        required_fields = [
            'total_trades', 'winning_trades', 'losing_trades',
            'total_profit_usdt', 'total_return_pct', 'win_rate_pct',
            'avg_profit_per_trade', 'max_drawdown_pct', 'sharpe_ratio'
        ]
        
        for field in required_fields:
            assert field in analytics
            assert isinstance(analytics[field], (int, float))
    
    @pytest.mark.unit
    def test_strategy_state_management(self, sample_strategy):
        """Strateji durum yönetimi testi"""
        # Başlangıç durumu
        assert sample_strategy.state.value == "initializing"
        assert not sample_strategy.is_active()
        
        # Aktif duruma geç
        sample_strategy.state = sample_strategy.StrategyState.ACTIVE
        assert sample_strategy.is_active()
        
        # Duraklat
        sample_strategy.pause_strategy()
        assert sample_strategy.state.value == "paused"
        assert not sample_strategy.is_active()
        
        # Devam et
        sample_strategy.resume_strategy()
        assert sample_strategy.state.value == "active"
        assert sample_strategy.is_active()
        
        # Durdur
        sample_strategy.stop_strategy()
        assert sample_strategy.state.value == "stopped"
        assert not sample_strategy.is_active()


class TestEnhancedMomentumStrategy:
    """EnhancedMomentumStrategy sınıfı unit testleri"""
    
    @pytest.fixture
    def momentum_strategy(self, sample_portfolio):
        """Momentum stratejisi oluştur"""
        return EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
    
    @pytest.mark.unit
    def test_momentum_initialization(self, momentum_strategy):
        """Momentum stratejisi başlatma testi"""
        # Optimized parametreler kontrol edilmeli
        assert momentum_strategy.ema_short in [13, 14, 15]  # Optimized range
        assert momentum_strategy.ema_medium in [21, 22, 23]  # Optimized range
        assert momentum_strategy.ema_long in [56, 57, 58]  # Optimized range
        assert momentum_strategy.rsi_period in [13, 14, 15]  # Optimized range
        
        # Performance based sizing parametreleri
        assert momentum_strategy.size_high_profit_pct > 0
        assert momentum_strategy.size_good_profit_pct > 0
        assert momentum_strategy.size_normal_profit_pct > 0
        
        # Risk yönetimi parametreleri
        assert momentum_strategy.max_loss_pct > 0
        assert momentum_strategy.min_profit_target_usdt > 0
    
    @pytest.mark.unit
    def test_momentum_indicators(self, momentum_strategy, sample_market_data):
        """Momentum indikatörleri testi"""
        indicators = momentum_strategy._calculate_momentum_indicators(sample_market_data)
        
        required_indicators = [
            'ema_short', 'ema_medium', 'ema_long', 'rsi', 'adx',
            'atr', 'volume_sma', 'price_momentum', 'volume_ratio'
        ]
        
        for indicator in required_indicators:
            assert indicator in indicators
            assert isinstance(indicators[indicator], (float, pd.Series))
    
    @pytest.mark.unit
    def test_momentum_signal_analysis(self, momentum_strategy, sample_market_data):
        """Momentum sinyal analizi testi"""
        signals = momentum_strategy._analyze_momentum_signals(sample_market_data)
        
        required_fields = [
            'signal_strength', 'quality_score', 'momentum_score',
            'trend_alignment', 'volume_confirmation', 'risk_assessment'
        ]
        
        for field in required_fields:
            assert field in signals
            assert isinstance(signals[field], (int, float, bool))
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_momentum_market_analysis(self, momentum_strategy, sample_market_data):
        """Momentum piyasa analizi testi"""
        signal = await momentum_strategy.analyze_market(sample_market_data)
        
        assert isinstance(signal, TradingSignal)
        assert signal.signal_type in [SignalType.BUY, SignalType.SELL, SignalType.HOLD]
        assert 0.0 <= signal.confidence <= 1.0
        assert signal.price > 0
        assert len(signal.reasons) > 0
        assert 'metadata' in signal.__dict__ or hasattr(signal, 'metadata')
    
    @pytest.mark.unit
    def test_momentum_position_sizing(self, momentum_strategy, sample_trading_signal):
        """Momentum pozisyon boyutlandırma testi"""
        position_size = momentum_strategy.calculate_position_size(sample_trading_signal)
        
        assert isinstance(position_size, float)
        assert position_size > 0
        assert position_size <= momentum_strategy.portfolio.get_available_usdt()
    
    @pytest.mark.unit
    def test_performance_based_sizing(self, momentum_strategy, sample_trading_signal):
        """Performans bazlı boyutlandırma testi"""
        size = momentum_strategy._calculate_performance_based_size(sample_trading_signal)
        
        assert isinstance(size, float)
        assert size > 0
        
        # Performance multiplier hesapla
        multiplier = momentum_strategy._calculate_performance_multiplier()
        assert isinstance(multiplier, float)
        assert multiplier > 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_momentum_sell_decision(self, momentum_strategy, sample_market_data, sample_position):
        """Momentum satış kararı testi"""
        should_sell, reason = await momentum_strategy.should_sell(sample_position, sample_market_data)
        
        assert isinstance(should_sell, bool)
        assert isinstance(reason, str)
        assert len(reason) > 0
    
    @pytest.mark.unit
    def test_position_age_calculation(self, momentum_strategy, sample_position):
        """Pozisyon yaşı hesaplama testi"""
        age_minutes = momentum_strategy._get_position_age_minutes(sample_position)
        
        assert isinstance(age_minutes, int)
        assert age_minutes >= 0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_ml_prediction_integration(self, momentum_strategy, sample_market_data):
        """ML tahmin entegrasyonu testi"""
        features = momentum_strategy._prepare_ml_features(sample_market_data)
        
        assert isinstance(features, dict)
        assert len(features) > 0
        
        # ML tahmin al (eğer ML predictor mevcutsa)
        if momentum_strategy.ml_predictor is not None:
            prediction = await momentum_strategy._get_ml_prediction(features)
            if prediction is not None:
                assert isinstance(prediction, dict)
                assert 'prediction' in prediction
                assert 'confidence' in prediction
    
    @pytest.mark.unit
    def test_momentum_analytics(self, momentum_strategy):
        """Momentum analitikleri testi"""
        analytics = momentum_strategy.get_strategy_analytics()
        
        # Base analytics
        base_fields = [
            'total_trades', 'winning_trades', 'losing_trades',
            'total_profit_usdt', 'total_return_pct', 'win_rate_pct'
        ]
        
        for field in base_fields:
            assert field in analytics
        
        # Momentum specific analytics
        momentum_fields = [
            'momentum_performance', 'quality_score_distribution',
            'regime_performance', 'ml_prediction_accuracy'
        ]
        
        for field in momentum_fields:
            if field in analytics:  # Bazı alanlar opsiyonel olabilir
                assert analytics[field] is not None


class TestStrategyIntegration:
    """Strateji entegrasyon testleri"""
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_strategy_portfolio_integration(self, sample_portfolio, sample_market_data):
        """Strateji-portföy entegrasyonu testi"""
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        # Piyasa analizi yap
        signal = await strategy.analyze_market(sample_market_data)
        
        if signal.signal_type == SignalType.BUY:
            # Pozisyon boyutu hesapla
            position_size = strategy.calculate_position_size(signal)
            
            # Alım işlemi gerçekleştir
            position = await sample_portfolio.execute_buy(
                strategy_name=strategy.strategy_name,
                symbol=strategy.symbol,
                current_price=signal.price,
                timestamp=signal.timestamp.isoformat(),
                reason="Test integration",
                amount_usdt_override=position_size
            )
            
            assert position is not None
            assert position.strategy_name == strategy.strategy_name
            assert len(sample_portfolio.positions) == 1
    
    @pytest.mark.integration
    @pytest.mark.asyncio
    async def test_strategy_evolution_integration(self, sample_portfolio, sample_market_data):
        """Strateji evrim entegrasyonu testi"""
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT"
        )
        
        # Birkaç trade simüle et
        for i in range(5):
            signal = await strategy.analyze_market(sample_market_data)
            
            if signal.signal_type == SignalType.BUY:
                position_size = strategy.calculate_position_size(signal)
                position = await sample_portfolio.execute_buy(
                    strategy_name=strategy.strategy_name,
                    symbol=strategy.symbol,
                    current_price=signal.price,
                    timestamp=signal.timestamp.isoformat(),
                    reason=f"Test trade {i}",
                    amount_usdt_override=position_size
                )
                
                # Pozisyonu kapat
                await sample_portfolio.execute_sell(
                    position_to_close=position,
                    current_price=signal.price * 1.02,  # %2 kâr
                    timestamp=signal.timestamp.isoformat(),
                    reason=f"Test close {i}"
                )
        
        # Strateji analitikleri kontrol et
        analytics = strategy.get_strategy_analytics()
        assert analytics['total_trades'] > 0
        assert analytics['total_profit_usdt'] > 0
    
    @pytest.mark.integration
    def test_strategy_parameter_loading(self, sample_portfolio):
        """Strateji parametre yükleme testi"""
        # JSON parametre yükleme testi
        json_manager = JSONParameterManager()
        
        # Test parametreleri oluştur
        test_params = {
            'ema_short': 13,
            'ema_medium': 21,
            'ema_long': 56,
            'rsi_period': 14,
            'max_positions': 3,
            'base_position_size_pct': 25.0
        }
        
        # Parametreleri kaydet
        success = json_manager.save_optimization_results(
            strategy_name='momentum',
            best_parameters=test_params,
            optimization_metrics={'best_score': 26.8}
        )
        
        assert success is True
        
        # Parametreleri yükle
        loaded_data = json_manager.load_strategy_parameters('momentum')
        assert loaded_data is not None
        assert 'parameters' in loaded_data
        
        # Stratejiyi parametrelerle başlat
        strategy = EnhancedMomentumStrategy(
            portfolio=sample_portfolio,
            symbol="BTC/USDT",
            **loaded_data['parameters']
        )
        
        # Parametrelerin doğru yüklendiğini kontrol et
        assert strategy.ema_short == test_params['ema_short']
        assert strategy.ema_medium == test_params['ema_medium']
        assert strategy.ema_long == test_params['ema_long'] 