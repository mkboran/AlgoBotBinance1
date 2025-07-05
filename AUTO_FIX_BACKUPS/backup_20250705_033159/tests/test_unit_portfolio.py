#!/usr/bin/env python3
"""
🧪 PROJE PHOENIX - PORTFOLIO UNIT TESTS
💎 Ultra Gelişmiş Portfolio Test Sistemi

Bu dosya Portfolio sınıfının tüm fonksiyonlarını test eder:
- Position yönetimi
- Risk hesaplamaları
- Performans metrikleri
- Trade execution
- Portfolio analytics

HEDGE FUND LEVEL UNIT TESTING
"""

import pytest
from unittest.mock import MagicMock, patch, AsyncMock
import pandas as pd
import numpy as np
from datetime import datetime, timezone, timedelta
from typing import Dict, Any
import asyncio

from utils.portfolio import Portfolio, Position


class TestPortfolio:
    """Portfolio sınıfı unit testleri"""
    
    @pytest.mark.unit
    def test_portfolio_initialization(self):
        """Portfolio başlatma testi"""
        # Test farklı başlangıç sermayeleri
        test_capitals = [1000.0, 10000.0, 100000.0, 0.0]
        
        for capital in test_capitals:
            portfolio = Portfolio(initial_capital_usdt=capital)
            
            assert portfolio.initial_capital_usdt == capital
            assert portfolio.available_usdt == capital
            assert len(portfolio.positions) == 0
            assert len(portfolio.closed_trades) == 0
            assert len(portfolio.closed_trades) == 0
            assert portfolio.cumulative_pnl == 0.0
    
    @pytest.mark.unit
    def test_portfolio_available_balance(self):
        """Mevcut bakiye hesaplama testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        
        # Başlangıçta tüm sermaye kullanılabilir
        assert portfolio.get_available_usdt() == 10000.0
        
        # Pozisyon açtıktan sonra bakiye azalmalı
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        portfolio.positions.append(position)
        
        # Kullanılabilir bakiye azalmalı
        assert portfolio.get_available_usdt() == 9500.0
    
    @pytest.mark.unit
    def test_portfolio_total_value_calculation(self):
        """Toplam portföy değeri hesaplama testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        
        # Başlangıçta sadece nakit
        current_price = 50000.0
        total_value = portfolio.get_total_portfolio_value_usdt(current_price)
        assert total_value == 10000.0
        
        # Pozisyon ekle
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        portfolio.positions.append(position)
        
        # Farklı fiyatlarda toplam değer hesapla
        test_prices = [45000.0, 50000.0, 55000.0]
        expected_values = [9500.0, 10000.0, 10500.0]  # 9500 + 0.01 * price
        
        for price, expected in zip(test_prices, expected_values):
            total_value = portfolio.get_total_portfolio_value_usdt(price)
            assert abs(total_value - expected) < 0.01
    
    @pytest.mark.unit
    def test_position_size_calculation(self):
        """Pozisyon boyutu hesaplama testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        current_price = 50000.0
        
        # Farklı quality score'lar için pozisyon boyutu
        test_cases = [
            (5, 0.05),   # Düşük quality
            (10, 0.10),  # Orta quality
            (15, 0.15),  # Yüksek quality
            (20, 0.20),  # Çok yüksek quality
        ]
        
        for quality_score, expected_size_pct in test_cases:
            position_size = portfolio.calculate_optimal_position_size(
                current_price, quality_score, "BULL"
            )
            
            # Pozisyon boyutu quality score'a göre artmalı
            assert position_size > 0
            assert position_size <= portfolio.available_usdt * 0.25  # Max %25
    
    @pytest.mark.unit
    def test_portfolio_performance_multiplier(self):
        """Portföy performans çarpanı testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        
        # Başlangıçta çarpan 1.0 olmalı
        multiplier = portfolio.get_portfolio_performance_multiplier()
        assert multiplier == 1.0
        
        # Kârlı trade ekle
        # Mock trade'ler oluşturarak kapalı işlemleri simüle et
        # MagicMock, test sırasında gerçek bir nesne gibi davranabilen sahte bir nesnedir.
        portfolio.closed_trades = [MagicMock()] * 10
        portfolio.winning_trades = 7
        portfolio.cumulative_pnl = 500.0
        
        # Performans çarpanı artmalı
        multiplier = portfolio.get_portfolio_performance_multiplier()
        assert multiplier > 1.0
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_buy_execution(self):
        """Alım işlemi testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        current_price = 50000.0
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Alım işlemi gerçekleştir
        position = await portfolio.execute_buy(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            current_price=current_price,
            timestamp=timestamp,
            reason="Test buy order",
            amount_usdt_override=500.0
        )
        
        # Pozisyon oluşturulmalı
        assert position is not None
        assert position.strategy_name == "test_strategy"
        assert position.symbol == "BTC/USDT"
        assert position.entry_price == current_price
        assert position.entry_cost_usdt_total == 500.0
        assert position.quantity_btc == 0.01  # 500 / 50000
        
        # Portföy güncellenmeli
        assert len(portfolio.positions) == 1
        assert portfolio.get_available_usdt() == 9500.0  # 10000 - 500
        assert len(portfolio.closed_trades) == 1
    
    @pytest.mark.unit
    @pytest.mark.asyncio
    async def test_sell_execution(self):
        """Satış işlemi testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        current_price = 50000.0
        timestamp = datetime.now(timezone.utc).isoformat()
        
        # Önce alım yap
        position = await portfolio.execute_buy(
            strategy_name="test_strategy",
            symbol="BTC/USDT",
            current_price=current_price,
            timestamp=timestamp,
            reason="Test buy order",
            amount_usdt_override=500.0
        )
        
        # Sonra satış yap
        sell_price = 55000.0  # %10 kâr
        success = await portfolio.execute_sell(
            position_to_close=position,
            current_price=sell_price,
            timestamp=timestamp,
            reason="Test sell order"
        )
        
        # Satış başarılı olmalı
        assert success is True
        
        # Pozisyon kapatılmalı
        assert len(portfolio.positions) == 0
        assert len(portfolio.closed_trades) == 1
        
        # Kâr hesaplanmalı
        expected_profit = (sell_price - current_price) * position.quantity_btc
        assert abs(portfolio.cumulative_pnl - expected_profit) < 0.01
    
    @pytest.mark.unit
    def test_position_performance_metrics(self):
        """Pozisyon performans metrikleri testi"""
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Farklı fiyatlarda performans metrikleri
        test_cases = [
            (45000.0, -50.0, -10.0),   # Zarar
            (50000.0, 0.0, 0.0),       # Breakeven
            (55000.0, 50.0, 10.0),     # Kâr
        ]
        
        for current_price, expected_profit, expected_pct in test_cases:
            metrics = position.update_performance_metrics(current_price)
            
            assert abs(metrics['current_profit'] - expected_profit) < 0.01
            assert abs(metrics['current_profit_pct'] - expected_pct) < 0.01
    
    @pytest.mark.unit
    def test_position_trailing_stop(self):
        """Pozisyon trailing stop testi"""
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Kâr yokken trailing stop aktif olmamalı
        should_trail, new_stop = position.should_trail_stop(50000.0)
        assert should_trail is False
        assert new_stop is None
        
        # %5 kârda trailing stop aktif olmalı
        should_trail, new_stop = position.should_trail_stop(52500.0)  # %5 kâr
        assert should_trail is True
        assert new_stop is not None
        assert new_stop < 52500.0  # Stop fiyatı mevcut fiyattan düşük olmalı
    
    @pytest.mark.unit
    def test_portfolio_risk_metrics(self):
        """Portföy risk metrikleri testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        
        # Başlangıçta risk metrikleri sıfır olmalı
        assert portfolio.max_drawdown_pct == 0.0
        assert portfolio.current_drawdown_pct == 0.0
        
        # Pozisyon ekle ve fiyat düşür
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        portfolio.positions.append(position)
        
        # Fiyat düşür ve drawdown hesapla
        lower_price = 45000.0  # %10 düşüş
        portfolio.track_portfolio_value(lower_price)
        
        # Drawdown hesaplanmalı
        assert portfolio.current_drawdown_pct > 0.0
        assert portfolio.max_drawdown_pct > 0.0
    
    @pytest.mark.unit
    def test_portfolio_performance_summary(self):
        """Portföy performans özeti testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        current_price = 50000.0
        
        # Performans özeti al
        summary = portfolio.get_performance_summary(current_price)
        
        # Gerekli alanlar olmalı
        required_fields = [
            'total_trades', 'winning_trades', 'losing_trades',
            'cumulative_pnl', 'total_return_pct', 'win_rate_pct',
            'sharpe_ratio', 'max_drawdown_pct', 'available_usdt',
            'total_portfolio_value_usdt'
        ]
        
        for field in required_fields:
            assert field in summary
            assert isinstance(summary[field], (int, float))
    
    @pytest.mark.unit
    def test_position_filtering(self):
        """Pozisyon filtreleme testi"""
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        
        # Farklı stratejilerden pozisyonlar ekle
        positions = [
            Position("pos1", "momentum", "BTC/USDT", 0.01, 50000.0, 500.0, datetime.now(timezone.utc).isoformat()),
            Position("pos2", "bollinger", "BTC/USDT", 0.01, 50000.0, 500.0, datetime.now(timezone.utc).isoformat()),
            Position("pos3", "momentum", "ETH/USDT", 0.1, 3000.0, 300.0, datetime.now(timezone.utc).isoformat()),
        ]
        
        for pos in positions:
            portfolio.positions.append(pos)
        
        # Stratejiye göre filtrele
        momentum_positions = portfolio.get_open_positions(strategy_name="momentum")
        assert len(momentum_positions) == 2
        
        # Sembole göre filtrele
        btc_positions = portfolio.get_open_positions(symbol="BTC/USDT")
        assert len(btc_positions) == 2
        
        # Hem strateji hem sembol
        momentum_btc = portfolio.get_open_positions(strategy_name="momentum", symbol="BTC/USDT")
        assert len(momentum_btc) == 1
    
    @pytest.mark.unit
    def test_portfolio_edge_cases(self):
        """Portföy edge case testleri"""
        # Sıfır sermaye ile başlat
        portfolio = Portfolio(initial_capital_usdt=0.0)
        assert portfolio.get_available_usdt() == 0.0
        
        # Çok büyük sermaye
        portfolio = Portfolio(initial_capital_usdt=1000000.0)
        assert portfolio.get_available_usdt() == 1000000.0
        
        # Negatif fiyat (hata durumu)
        portfolio = Portfolio(initial_capital_usdt=10000.0)
        with pytest.raises(Exception):
            portfolio.get_total_portfolio_value_usdt(-100.0)
    
    @pytest.mark.unit
    def test_position_serialization(self):
        """Pozisyon serileştirme testi"""
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # String representation
        pos_str = str(position)
        assert "test_001" in pos_str
        assert "BTC/USDT" in pos_str
        
        # Repr representation
        pos_repr = repr(position)
        assert "Position" in pos_repr
        assert "test_001" in pos_repr


class TestPosition:
    """Position sınıfı unit testleri"""
    
    @pytest.mark.unit
    def test_position_initialization(self):
        """Position başlatma testi"""
        position = Position(
            position_id="test_001",
            strategy_name="momentum",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat(),
            stop_loss_price=49500.0,
            entry_context={'quality_score': 15}
        )
        
        assert position.position_id == "test_001"
        assert position.strategy_name == "momentum"
        assert position.symbol == "BTC/USDT"
        assert position.quantity_btc == 0.01
        assert position.entry_price == 50000.0
        assert position.entry_cost_usdt_total == 500.0
        assert position.stop_loss_price == 49500.0
        assert position.quality_score == 15
        assert position.highest_price_seen == 50000.0
        assert position.lowest_price_seen == 50000.0
    
    @pytest.mark.unit
    def test_position_performance_tracking(self):
        """Pozisyon performans takibi testi"""
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Fiyat yüksel
        metrics = position.update_performance_metrics(55000.0)
        assert metrics['current_profit'] > 0
        assert metrics['current_profit_pct'] > 0
        assert position.highest_price_seen == 55000.0
        
        # Fiyat düş
        metrics = position.update_performance_metrics(45000.0)
        assert metrics['current_profit'] < 0
        assert metrics['current_profit_pct'] < 0
        assert position.lowest_price_seen == 45000.0
        assert position.highest_price_seen == 55000.0  # Değişmemeli
    
    @pytest.mark.unit
    def test_position_risk_metrics(self):
        """Pozisyon risk metrikleri testi"""
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        # Fiyat güncellemeleri yap
        for price in [51000, 52000, 53000, 52000, 51000]:
            position.update_performance_metrics(price)
        
        # Risk metrikleri hesaplanmalı
        assert 'volatility' in position.risk_metrics
        assert 'downside_volatility' in position.risk_metrics
        assert 'value_at_risk_5' in position.risk_metrics
        assert 'sharpe_estimate' in position.risk_metrics
    
    @pytest.mark.unit
    def test_position_summary(self):
        """Pozisyon özeti testi"""
        position = Position(
            position_id="test_001",
            strategy_name="test",
            symbol="BTC/USDT",
            quantity_btc=0.01,
            entry_price=50000.0,
            entry_cost_usdt_total=500.0,
            timestamp=datetime.now(timezone.utc).isoformat()
        )
        
        current_price = 55000.0
        summary = position.get_performance_summary(current_price)
        
        required_fields = [
            'position_id', 'strategy', 'entry_price', 'current_price',
            'quantity', 'entry_cost', 'current_value', 'unrealized_pnl',
            'unrealized_pnl_pct', 'max_profit', 'max_drawdown',
            'hold_time_minutes', 'quality_score', 'ai_approved'
        ]
        
        for field in required_fields:
            assert field in summary 