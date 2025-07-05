#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - BACKTEST CONFIGURATION FIX
💎 FIXED: BacktestConfiguration'a eksik parametreler eklendi

ÇÖZÜMLER:
1. ✅ enable_position_sizing parametresi eklendi
2. ✅ Diğer gelişmiş backtest parametreleri eklendi
3. ✅ Type hints ve validation geliştirildi
"""

from dataclasses import dataclass, field
from datetime import datetime, timezone
from enum import Enum
from typing import Dict, Any, Optional, List
import pandas as pd

class BacktestMode(Enum):
    """Backtest modları"""
    SINGLE_STRATEGY = "single_strategy"
    MULTI_STRATEGY = "multi_strategy"
    PORTFOLIO = "portfolio"
    WALK_FORWARD = "walk_forward"

@dataclass
class BacktestConfiguration:
    """
    🎯 ULTRA ADVANCED BACKTEST CONFIGURATION
    💎 Hedge Fund Level Backtesting Parameters
    
    Tüm backtest parametrelerini içeren gelişmiş konfigürasyon
    """
    
    # TEMEL PARAMETRELER
    start_date: datetime
    end_date: datetime
    initial_capital: float = 10000.0
    
    # TRADING COSTS
    commission_rate: float = 0.001  # %0.1
    slippage_rate: float = 0.0005   # %0.05
    spread_pips: float = 0.0        # Opsiyonel spread
    
    # BACKTEST MODU
    mode: BacktestMode = BacktestMode.SINGLE_STRATEGY
    
    # ✅ POZİSYON YÖNETİMİ - FIXED
    enable_position_sizing: bool = True  # Dinamik pozisyon boyutlandırma
    max_positions: int = 5               # Maksimum açık pozisyon
    position_size_method: str = "kelly"  # kelly, fixed, volatility_adjusted
    
    # ✅ RİSK YÖNETİMİ
    max_drawdown_pct: float = 0.20      # %20 maksimum düşüş
    daily_loss_limit: float = 0.05      # Günlük %5 kayıp limiti
    position_risk_pct: float = 0.02     # Pozisyon başına %2 risk
    portfolio_heat: float = 0.06        # Toplam portföy riski %6
    
    # ✅ PERFORMANS ANALİZİ
    calculate_sharpe: bool = True
    calculate_sortino: bool = True
    calculate_calmar: bool = True
    calculate_max_drawdown: bool = True
    calculate_win_rate: bool = True
    calculate_profit_factor: bool = True
    calculate_expectancy: bool = True
    
    # ✅ GELİŞMİŞ ÖZELLİKLER
    enable_monte_carlo: bool = True          # Monte Carlo simülasyonu
    monte_carlo_iterations: int = 1000       # MC iterasyon sayısı
    enable_walk_forward: bool = False        # Walk-forward analizi
    walk_forward_periods: int = 4            # WF period sayısı
    enable_parameter_sensitivity: bool = True # Parametre hassasiyet analizi
    
    # ✅ DATA PROCESSING
    data_frequency: str = "15min"            # Veri frekansı
    enable_data_validation: bool = True      # Veri doğrulama
    handle_missing_data: str = "forward_fill" # Eksik veri yönetimi
    outlier_detection: bool = True           # Outlier tespiti
    outlier_threshold: float = 3.0           # Standart sapma eşiği
    
    # ✅ EXECUTION SIMULATION
    enable_partial_fills: bool = True        # Kısmi dolum simülasyonu
    enable_order_latency: bool = True        # Emir gecikmesi
    latency_ms: int = 50                     # Ortalama gecikme (ms)
    enable_market_impact: bool = True        # Piyasa etkisi
    market_impact_model: str = "linear"      # linear, sqrt, power
    
    # ✅ REPORTING
    generate_html_report: bool = True        # HTML rapor oluştur
    generate_pdf_report: bool = False        # PDF rapor oluştur
    save_trade_log: bool = True              # Trade log kaydet
    save_equity_curve: bool = True           # Equity curve kaydet
    save_drawdown_series: bool = True        # Drawdown serisi kaydet
    
    # ✅ OPTIMIZATION SUPPORT
    optimization_metric: str = "sharpe_ratio" # Optimizasyon metriği
    optimization_constraints: Dict[str, Any] = field(default_factory=lambda: {
        "min_trades": 30,
        "max_drawdown": 0.25,
        "min_sharpe": 1.0,
        "min_win_rate": 0.40
    })
    
    # ✅ STRATEGY SPECIFIC
    strategy_parameters: Dict[str, Any] = field(default_factory=dict)
    benchmark_symbol: str = "BTC/USDT"       # Karşılaştırma sembolü
    risk_free_rate: float = 0.02             # Risksiz getiri oranı
    
    def validate(self) -> bool:
        """Konfigürasyon doğrulama"""
        errors = []
        
        # Tarih kontrolü
        if self.start_date >= self.end_date:
            errors.append("Start date must be before end date")
        
        # Sermaye kontrolü
        if self.initial_capital <= 0:
            errors.append("Initial capital must be positive")
        
        # Komisyon kontrolü
        if not 0 <= self.commission_rate <= 0.01:
            errors.append("Commission rate must be between 0 and 1%")
        
        # Risk limitleri
        if not 0 < self.max_drawdown_pct <= 1.0:
            errors.append("Max drawdown must be between 0 and 100%")
        
        if errors:
            raise ValueError(f"Configuration errors: {', '.join(errors)}")
        
        return True
    
    def to_dict(self) -> Dict[str, Any]:
        """Dictionary'e dönüştür"""
        config_dict = {
            'start_date': self.start_date.isoformat(),
            'end_date': self.end_date.isoformat(),
            'initial_capital': self.initial_capital,
            'commission_rate': self.commission_rate,
            'slippage_rate': self.slippage_rate,
            'mode': self.mode.value,
            'enable_position_sizing': self.enable_position_sizing,
            'max_positions': self.max_positions,
            'position_size_method': self.position_size_method,
            'max_drawdown_pct': self.max_drawdown_pct,
            'optimization_metric': self.optimization_metric,
            'strategy_parameters': self.strategy_parameters
        }
        return config_dict
    
    @classmethod
    def from_dict(cls, config_dict: Dict[str, Any]) -> 'BacktestConfiguration':
        """Dictionary'den oluştur"""
        # Parse dates
        config_dict['start_date'] = datetime.fromisoformat(config_dict['start_date'])
        config_dict['end_date'] = datetime.fromisoformat(config_dict['end_date'])
        
        # Parse mode
        if 'mode' in config_dict:
            config_dict['mode'] = BacktestMode(config_dict['mode'])
        
        return cls(**config_dict)
    
    def get_risk_adjusted_position_size(self, 
                                      account_balance: float,
                                      volatility: float,
                                      confidence: float = 0.95) -> float:
        """
        Risk-adjusted pozisyon boyutu hesapla
        
        Kelly Criterion + Volatility Adjustment + Confidence Scaling
        """
        base_size = account_balance * self.position_risk_pct
        
        if self.position_size_method == "kelly":
            # Kelly fraction (simplified)
            kelly_fraction = min(confidence * 0.25, 0.25)  # Max 25%
            base_size *= kelly_fraction
        
        elif self.position_size_method == "volatility_adjusted":
            # Normalize volatility (assume 0.02 is normal)
            vol_adjustment = 0.02 / max(volatility, 0.001)
            base_size *= min(vol_adjustment, 2.0)  # Max 2x adjustment
        
        # Apply confidence scaling
        base_size *= confidence
        
        # Apply maximum position size limit
        max_size = account_balance * 0.5  # Max 50% per position
        
        return min(base_size, max_size)


# ==================================================================================
# BACKTEST RESULT DATACLASS
# ==================================================================================

@dataclass
class BacktestResult:
    """
    📊 COMPREHENSIVE BACKTEST RESULTS
    💎 Institutional Grade Performance Metrics
    """
    
    # TEMEL METRIKLER
    strategy_name: str
    total_return: float = 0.0
    total_trades: int = 0
    winning_trades: int = 0
    losing_trades: int = 0
    
    # PERFORMANS METRİKLERİ
    sharpe_ratio: float = 0.0
    sortino_ratio: float = 0.0
    calmar_ratio: float = 0.0
    profit_factor: float = 0.0
    win_rate: float = 0.0
    expectancy: float = 0.0
    
    # RISK METRİKLERİ
    max_drawdown: float = 0.0
    max_drawdown_duration: int = 0
    var_95: float = 0.0  # Value at Risk
    cvar_95: float = 0.0  # Conditional VaR
    
    # TRADE İSTATİSTİKLERİ
    avg_win: float = 0.0
    avg_loss: float = 0.0
    largest_win: float = 0.0
    largest_loss: float = 0.0
    avg_trade_duration: float = 0.0
    
    # TIME SERIES
    equity_curve: Optional[pd.Series] = None
    drawdown_series: Optional[pd.Series] = None
    returns_series: Optional[pd.Series] = None
    
    # ADVANCED ANALYTICS
    monte_carlo_results: Optional[Dict[str, Any]] = None
    parameter_sensitivity: Optional[Dict[str, Any]] = None
    regime_performance: Optional[Dict[str, Any]] = None
    
    # EXECUTION STATS
    total_commission: float = 0.0
    total_slippage: float = 0.0
    data_points_processed: int = 0
    
    def get_summary(self) -> Dict[str, Any]:
        """Özet sonuçları döndür"""
        return {
            'strategy': self.strategy_name,
            'total_return_pct': round(self.total_return * 100, 2),
            'sharpe_ratio': round(self.sharpe_ratio, 2),
            'max_drawdown_pct': round(self.max_drawdown * 100, 2),
            'win_rate_pct': round(self.win_rate * 100, 2),
            'profit_factor': round(self.profit_factor, 2),
            'total_trades': self.total_trades,
            'expectancy_usdt': round(self.expectancy, 2)
        }