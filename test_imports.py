# test_imports.py - Safe Version
import sys
from pathlib import Path

# Add project root
project_root = Path(__file__).parent.absolute()
if str(project_root) not in sys.path:
    sys.path.insert(0, str(project_root))

print("Testing critical imports...")

try:
    import pandas
    print("OK pandas")
except: print("FAIL pandas")

try:
    import numpy
    print("OK numpy")
except: print("FAIL numpy")

try:
    import ccxt
    print("OK ccxt")
except: print("FAIL ccxt")

try:
    from utils.portfolio import Portfolio
    print("OK utils.portfolio")
except: print("FAIL utils.portfolio")

try:
    from strategies.momentum_optimized import EnhancedMomentumStrategy
    print("OK strategies.momentum_optimized")
except: print("FAIL strategies.momentum_optimized")

try:
    from backtesting.multi_strategy_backtester import MultiStrategyBacktester
    print("OK backtesting.multi_strategy_backtester")
except: print("FAIL backtesting.multi_strategy_backtester")

try:
    from optimization.master_optimizer import MasterOptimizer
    print("OK optimization.master_optimizer")
except: print("FAIL optimization.master_optimizer")

print("\nImport test completed.")
