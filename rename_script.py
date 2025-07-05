import os

rename_map = [
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/1.md', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_summary.md'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/1.py', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_parameter_spaces.py'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/2.md', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_application_guide.md'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/2.py', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_main_imports.py'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/3.md', 'C:/Projects/AlgoBotBinance/Claude_cevabı/test_validation_guide.md'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/3.py', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_backtest_config.py'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/4.py', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_json_parameter_system.py'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/5.py', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_portfolio_logger.py'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/6.py', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_base_strategy_methods.py'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/7.py', 'C:/Projects/AlgoBotBinance/Claude_cevabı/fix_momentum_strategy_methods.py'),
    ('C:/Projects/AlgoBotBinance/Claude_cevabı/en_son_yazdı_bunu.py', 'C:/Projects/AlgoBotBinance/Claude_cevabı/auto_fix_script.py')
]

for old_name, new_name in rename_map:
    try:
        os.rename(old_name, new_name)
        print(f"Renamed: {old_name} -> {new_name}")
    except FileNotFoundError:
        print(f"File not found: {old_name}")
    except Exception as e:
        print(f"Error renaming {old_name}: {e}")
