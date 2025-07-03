# test_imports.py
# Sistem doğrulama için temel import testi

try:
    import pandas
    import numpy
    import ccxt
    import pandas_ta
    import strategies.momentum_optimized
    import backtesting.backtest_runner
    import utils.portfolio
    import utils.strategy_coordinator
    import optimization.master_optimizer
    import scripts.validate_system
    from json_parameter_system import JSONParameterManager
    print("✅ All critical imports succeeded.")
except Exception as e:
    print(f"❌ Import failed: {e}")
    raise 