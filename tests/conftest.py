#!/usr/bin/env python3
"""
🧪 PROJE PHOENIX - PYTEST KONFİGÜRASYONU VE FIXTURES
💎 Ultra Gelişmiş Test Altyapısı

Bu dosya şunları sağlar:
- Pytest konfigürasyonu
- Test fixtures (test verileri, mock objeler)
- Test ortamı ayarları
- Coverage konfigürasyonu
- Performance test ayarları

HEDGE FUND LEVEL TESTING CONFIGURATION
"""

import pytest
import pandas as pd
import numpy as np
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, List
import sys
import os

# Proje kökünü Python path'e ekle
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Test verileri için klasör oluştur
TEST_DATA_DIR = PROJECT_ROOT / "tests" / "test_data"
TEST_DATA_DIR.mkdir(parents=True, exist_ok=True)

# Test logları için klasör oluştur
TEST_LOGS_DIR = PROJECT_ROOT / "tests" / "logs"
TEST_LOGS_DIR.mkdir(parents=True, exist_ok=True)

# Test sonuçları için klasör oluştur
TEST_RESULTS_DIR = PROJECT_ROOT / "tests" / "results"
TEST_RESULTS_DIR.mkdir(parents=True, exist_ok=True)

# Pytest konfigürasyonu
def pytest_configure(config):
    """Pytest konfigürasyonu"""
    config.addinivalue_line(
        "markers", "unit: Unit testleri"
    )
    config.addinivalue_line(
        "markers", "integration: Entegrasyon testleri"
    )
    config.addinivalue_line(
        "markers", "performance: Performans testleri"
    )
    config.addinivalue_line(
        "markers", "security: Güvenlik testleri"
    )
    config.addinivalue_line(
        "markers", "slow: Yavaş çalışan testler"
    )

def pytest_collection_modifyitems(config, items):
    """Test koleksiyonunu modifiye et"""
    for item in items:
        # Unit testleri için marker ekle
        if "test_unit" in item.nodeid:
            item.add_marker(pytest.mark.unit)
        # Integration testleri için marker ekle
        elif "test_integration" in item.nodeid:
            item.add_marker(pytest.mark.integration)
        # Performance testleri için marker ekle
        elif "test_performance" in item.nodeid:
            item.add_marker(pytest.mark.performance)
        # Security testleri için marker ekle
        elif "test_security" in item.nodeid:
            item.add_marker(pytest.mark.security)

# ==================================================================================
# TEST FIXTURES
# ==================================================================================

@pytest.fixture(scope="session")
def test_data_dir():
    """Test verileri klasörü"""
    return TEST_DATA_DIR

@pytest.fixture(scope="session")
def test_logs_dir():
    """Test logları klasörü"""
    return TEST_LOGS_DIR

@pytest.fixture(scope="session")
def test_results_dir():
    """Test sonuçları klasörü"""
    return TEST_RESULTS_DIR

@pytest.fixture(scope="session")
def sample_market_data():
    """Örnek piyasa verisi oluştur"""
    # 100 günlük örnek OHLCV verisi
    dates = pd.date_range(
        start=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end=datetime(2024, 4, 10, tzinfo=timezone.utc),
        freq='15min'
    )
    
    # Gerçekçi fiyat verisi oluştur
    np.random.seed(42)  # Tekrarlanabilirlik için
    base_price = 50000.0
    returns = np.random.normal(0, 0.02, len(dates))  # %2 günlük volatilite
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    # OHLCV verisi oluştur
    data = []
    for i, (date, price) in enumerate(zip(dates, prices)):
        # High, Low, Open, Close varyasyonları
        variation = price * 0.01  # %1 varyasyon
        high = price + np.random.uniform(0, variation)
        low = price - np.random.uniform(0, variation)
        open_price = price + np.random.uniform(-variation/2, variation/2)
        close_price = price + np.random.uniform(-variation/2, variation/2)
        volume = np.random.uniform(100, 1000)
        
        data.append({
            'timestamp': date,
            'open': open_price,
            'high': high,
            'low': low,
            'close': close_price,
            'volume': volume
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

@pytest.fixture(scope="session")
def sample_portfolio():
    """Örnek portföy oluştur"""
    from utils.portfolio import Portfolio
    
    portfolio = Portfolio(initial_capital_usdt=10000.0)
    return portfolio

@pytest.fixture(scope="session")
def sample_strategy_config():
    """Örnek strateji konfigürasyonu"""
    return {
        'symbol': 'BTC/USDT',
        'timeframe': '15m',
        'max_positions': 3,
        'base_position_size_pct': 25.0,
        'max_loss_pct': 10.0,
        'min_profit_target_usdt': 5.0,
        'ml_enabled': True,
        'ml_confidence_threshold': 0.65,
        'dynamic_exit_enabled': True,
        'kelly_enabled': True,
        'global_intelligence_enabled': True
    }

@pytest.fixture(scope="session")
def mock_binance_fetcher():
    """Mock Binance veri çekici"""
    class MockBinanceFetcher:
        def __init__(self, symbol="BTC/USDT", timeframe="15m"):
            self.symbol = symbol
            self.timeframe = timeframe
            self.data = None
        
        async def fetch_ohlcv_data(self, limit_override=None, since_timestamp_ms=None):
            """Mock OHLCV veri çekme"""
            if self.data is None:
                # Örnek veri oluştur
                dates = pd.date_range(
                    start=datetime.now(timezone.utc) - timedelta(days=7),
                    end=datetime.now(timezone.utc),
                    freq='15min'
                )
                
                np.random.seed(42)
                base_price = 50000.0
                prices = [base_price]
                for _ in range(len(dates) - 1):
                    ret = np.random.normal(0, 0.01)
                    prices.append(prices[-1] * (1 + ret))
                
                data = []
                for date, price in zip(dates, prices):
                    variation = price * 0.005
                    data.append({
                        'timestamp': date,
                        'open': price + np.random.uniform(-variation, variation),
                        'high': price + np.random.uniform(0, variation),
                        'low': price - np.random.uniform(0, variation),
                        'close': price + np.random.uniform(-variation, variation),
                        'volume': np.random.uniform(100, 1000)
                    })
                
                df = pd.DataFrame(data)
                df.set_index('timestamp', inplace=True)
                self.data = df
            
            return self.data
        
        async def close_connection(self):
            """Mock bağlantı kapatma"""
            pass
    
    return MockBinanceFetcher()

@pytest.fixture(scope="session")
def sample_optimization_result():
    """Örnek optimizasyon sonucu"""
    return {
        'strategy_name': 'momentum',
        'best_parameters': {
            'ema_short': 13,
            'ema_medium': 21,
            'ema_long': 56,
            'rsi_period': 14,
            'max_positions': 3,
            'base_position_size_pct': 25.0,
            'max_loss_pct': 10.0,
            'ml_confidence_threshold': 0.65
        },
        'best_score': 26.8,
        'total_trials': 1000,
        'successful_trials': 950,
        'failed_trials': 50,
        'optimization_duration_minutes': 45.5,
        'robustness_score': 0.85,
        'final_validation_score': 0.82
    }

@pytest.fixture(scope="session")
def sample_backtest_config():
    """Örnek backtest konfigürasyonu"""
    from backtesting.multi_strategy_backtester import BacktestConfiguration, BacktestMode
    
    return BacktestConfiguration(
        start_date=datetime(2024, 1, 1, tzinfo=timezone.utc),
        end_date=datetime(2024, 3, 31, tzinfo=timezone.utc),
        initial_capital=10000.0,
        commission_rate=0.001,
        slippage_rate=0.0005,
        mode=BacktestMode.MULTI_STRATEGY,
        strategy_allocations={'momentum': 0.5, 'bollinger_rsi': 0.3, 'rsi_ml': 0.2},
        rebalancing_frequency='monthly',
        max_drawdown_threshold=0.20,
        enable_position_sizing=True
    )

@pytest.fixture(scope="session")
def sample_trading_signal():
    """Örnek trading sinyali"""
    from strategies.base_strategy import TradingSignal, SignalType
    
    return TradingSignal(
        signal_type=SignalType.BUY,
        confidence=0.75,
        price=50000.0,
        timestamp=datetime.now(timezone.utc),
        reasons=['Strong momentum', 'RSI oversold', 'Volume confirmation'],
        metadata={
            'quality_score': 15,
            'momentum_strength': 0.8,
            'volume_ratio': 2.5
        }
    )

@pytest.fixture(scope="session")
def sample_position():
    """Örnek pozisyon"""
    from utils.portfolio import Position
    
    return Position(
        position_id="test_pos_001",
        strategy_name="momentum",
        symbol="BTC/USDT",
        quantity_btc=0.01,
        entry_price=50000.0,
        entry_cost_usdt_total=500.0,
        timestamp=datetime.now(timezone.utc).isoformat(),
        stop_loss_price=49500.0,
        entry_context={
            'quality_score': 15,
            'momentum_score': 0.8,
            'market_regime': 'BULL',
            'volatility': 0.02
        }
    )

# ==================================================================================
# PERFORMANCE TEST FIXTURES
# ==================================================================================

@pytest.fixture(scope="session")
def large_market_dataset():
    """Büyük piyasa veri seti (performans testleri için)"""
    # 1 yıllık 15 dakikalık veri (yaklaşık 35,040 kayıt)
    dates = pd.date_range(
        start=datetime(2023, 1, 1, tzinfo=timezone.utc),
        end=datetime(2023, 12, 31, tzinfo=timezone.utc),
        freq='15min'
    )
    
    np.random.seed(42)
    base_price = 50000.0
    returns = np.random.normal(0, 0.015, len(dates))
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = []
    for date, price in zip(dates, prices):
        variation = price * 0.008
        data.append({
            'timestamp': date,
            'open': price + np.random.uniform(-variation, variation),
            'high': price + np.random.uniform(0, variation),
            'low': price - np.random.uniform(0, variation),
            'close': price + np.random.uniform(-variation, variation),
            'volume': np.random.uniform(50, 2000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

# ==================================================================================
# SECURITY TEST FIXTURES
# ==================================================================================

@pytest.fixture(scope="session")
def malicious_input_data():
    """Kötü niyetli giriş verileri (güvenlik testleri için)"""
    return {
        'sql_injection': "'; DROP TABLE users; --",
        'xss_script': "<script>alert('XSS')</script>",
        'path_traversal': "../../../etc/passwd",
        'large_number': "9" * 1000,
        'negative_values': [-999999, -0.0001, -float('inf')],
        'invalid_json': "{'invalid': json}",
        'unicode_injection': "🚀💣💥",
        'null_bytes': "test\x00string",
        'very_long_string': "A" * 10000
    }

# ==================================================================================
# TEST UTILITIES
# ==================================================================================

def create_test_market_data(start_date: datetime, end_date: datetime, 
                          base_price: float = 50000.0, volatility: float = 0.02) -> pd.DataFrame:
    """Test için piyasa verisi oluştur"""
    dates = pd.date_range(start=start_date, end=end_date, freq='15min')
    
    np.random.seed(42)
    returns = np.random.normal(0, volatility, len(dates))
    
    prices = [base_price]
    for ret in returns[1:]:
        prices.append(prices[-1] * (1 + ret))
    
    data = []
    for date, price in zip(dates, prices):
        variation = price * volatility * 0.5
        data.append({
            'timestamp': date,
            'open': price + np.random.uniform(-variation, variation),
            'high': price + np.random.uniform(0, variation),
            'low': price - np.random.uniform(0, variation),
            'close': price + np.random.uniform(-variation, variation),
            'volume': np.random.uniform(100, 1000)
        })
    
    df = pd.DataFrame(data)
    df.set_index('timestamp', inplace=True)
    
    return df

def assert_dataframe_structure(df: pd.DataFrame, expected_columns: List[str]):
    """DataFrame yapısını doğrula"""
    assert isinstance(df, pd.DataFrame), "DataFrame olmalı"
    assert not df.empty, "DataFrame boş olmamalı"
    assert all(col in df.columns for col in expected_columns), f"Beklenen kolonlar: {expected_columns}"
    assert df.index.name == 'timestamp', "Index 'timestamp' olmalı"

def assert_performance_metrics(metrics: Dict[str, Any]):
    """Performans metriklerini doğrula"""
    required_metrics = ['total_return_pct', 'sharpe_ratio', 'max_drawdown_pct', 'win_rate_pct']
    
    for metric in required_metrics:
        assert metric in metrics, f"Metrik eksik: {metric}"
        assert isinstance(metrics[metric], (int, float)), f"Metrik sayısal olmalı: {metric}"
    
    # Mantıklı değer aralıkları
    assert -100 <= metrics['total_return_pct'] <= 1000, "Total return mantıklı aralıkta olmalı"
    assert 0 <= metrics['max_drawdown_pct'] <= 100, "Max drawdown 0-100 arasında olmalı"
    assert 0 <= metrics['win_rate_pct'] <= 100, "Win rate 0-100 arasında olmalı" 