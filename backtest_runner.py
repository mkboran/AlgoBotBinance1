#!/usr/bin/env python3
"""
🧪 BACKTEST RUNNER - Test Compatibility Module
Bu dosya test uyumluluğu için oluşturuldu.
"""

from backtesting.multi_strategy_backtester import MultiStrategyBacktester

class MomentumBacktester:
    """Compatibility wrapper for MultiStrategyBacktester"""
    
    def __init__(self, csv_path: str, initial_capital: float = 10000.0, 
                 start_date: str = "2024-01-01", end_date: str = "2024-12-31",
                 symbol: str = "BTC/USDT"):
        self.csv_path = csv_path
        self.initial_capital = initial_capital
        self.start_date = start_date
        self.end_date = end_date
        self.symbol = symbol
        
        print(f"🧪 MomentumBacktester compatibility wrapper initialized")
        print(f"📁 Data: {csv_path}")
        print(f"💰 Capital: ${initial_capital:,.2f}")
        print(f"📅 Period: {start_date} to {end_date}")
