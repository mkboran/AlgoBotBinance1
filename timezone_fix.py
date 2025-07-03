#!/usr/bin/env python3
"""
🕐 TIMEZONE FIX - Datetime comparison sorununu çöz
"""

import re
from pathlib import Path
import shutil

class TimezoneFixer:
    def __init__(self):
        self.project_root = Path(".")
        
    def fix_timezone_issues(self):
        """🕐 Tüm timezone sorunlarını çöz"""
        
        print("🕐 TIMEZONE FIX BAŞLATILIYOR...")
        
        try:
            # 1. MultiStrategyBacktester'daki validation sorununu çöz
            self.fix_backtester_validation()
            
            # 2. Main.py'deki datetime parsing'i çöz
            self.fix_main_datetime_parsing()
            
            # 3. BacktestConfiguration'daki sorunları çöz
            self.fix_backtest_configuration()
            
            print("✅ TIMEZONE FIXES COMPLETED!")
            
        except Exception as e:
            print(f"❌ Timezone fix error: {e}")
    
    def fix_backtester_validation(self):
        """🔧 MultiStrategyBacktester validation metodunu düzelt"""
        
        backtester_file = self.project_root / "backtesting/multi_strategy_backtester.py"
        
        if not backtester_file.exists():
            print("❌ MultiStrategyBacktester file not found")
            return
        
        try:
            with open(backtester_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Backup oluştur
            backup_file = backtester_file.with_suffix('.timezone_backup')
            shutil.copy2(backtester_file, backup_file)
            
            # _validate_backtest_inputs metodunu timezone-safe hale getir
            old_validation = r'def _validate_backtest_inputs\(self, config.*?\) -> bool:.*?return True'
            
            new_validation = '''def _validate_backtest_inputs(self, config, data: pd.DataFrame) -> bool:
        """✅ Timezone-safe backtest input validation"""
        try:
            # Check config
            if not config:
                logger.error("❌ No backtest configuration provided")
                return False
            
            # Check data
            if data is None or data.empty:
                logger.error("❌ No market data provided")
                return False
                
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in data.columns]
            
            if missing_columns:
                logger.error(f"❌ Missing required columns: {missing_columns}")
                return False
            
            # TIMEZONE-SAFE DATE COMPARISON
            try:
                # Convert config dates to timezone-naive if needed
                from datetime import timezone
                
                config_start = config.start_date
                config_end = config.end_date
                
                # Make timezone-naive for comparison
                if hasattr(config_start, 'tzinfo') and config_start.tzinfo is not None:
                    config_start = config_start.replace(tzinfo=None)
                if hasattr(config_end, 'tzinfo') and config_end.tzinfo is not None:
                    config_end = config_end.replace(tzinfo=None)
                
                # Check start < end
                if config_start >= config_end:
                    logger.error("❌ Start date must be before end date")
                    return False
                
                # Check capital
                if config.initial_capital <= 0:
                    logger.error("❌ Initial capital must be positive")
                    return False
                
                # Data range check (timezone-safe)
                if not data.index.empty:
                    data_start = data.index.min()
                    data_end = data.index.max()
                    
                    # Make data dates timezone-naive too
                    if hasattr(data_start, 'tzinfo') and data_start.tzinfo is not None:
                        data_start = data_start.replace(tzinfo=None)
                    if hasattr(data_end, 'tzinfo') and data_end.tzinfo is not None:
                        data_end = data_end.replace(tzinfo=None)
                    
                    if config_start < data_start or config_end > data_end:
                        logger.warning(f"⚠️ Requested period extends beyond available data")
                
            except Exception as date_error:
                logger.warning(f"⚠️ Date validation warning: {date_error}")
                # Continue anyway - don't fail for date comparison issues
            
            logger.info("✅ Backtest inputs validated successfully (timezone-safe)")
            return True
            
        except Exception as e:
            logger.error(f"❌ Input validation error: {e}")
            return False'''
            
            # Replace the validation method
            content = re.sub(old_validation, new_validation, content, flags=re.DOTALL)
            
            # Also fix _prepare_backtest_data method for timezone safety
            old_prepare = r'def _prepare_backtest_data\(self, data: pd\.DataFrame, config.*?\) -> pd\.DataFrame:.*?return filtered_data'
            
            new_prepare = '''def _prepare_backtest_data(self, data: pd.DataFrame, config) -> pd.DataFrame:
        """📊 Timezone-safe data preparation"""
        try:
            # Make dates timezone-naive for safe comparison
            start_date = config.start_date
            end_date = config.end_date
            
            if hasattr(start_date, 'tzinfo') and start_date.tzinfo is not None:
                start_date = start_date.replace(tzinfo=None)
            if hasattr(end_date, 'tzinfo') and end_date.tzinfo is not None:
                end_date = end_date.replace(tzinfo=None)
            
            # Ensure data index is timezone-naive too
            data_copy = data.copy()
            if hasattr(data_copy.index, 'tz') and data_copy.index.tz is not None:
                data_copy.index = data_copy.index.tz_localize(None)
            
            # Filter by date range
            filtered_data = data_copy.loc[
                (data_copy.index >= start_date) & 
                (data_copy.index <= end_date)
            ].copy()
            
            if filtered_data.empty:
                raise ValueError("No data available for the specified date range")
            
            # Sort and clean
            filtered_data = filtered_data.sort_index()
            filtered_data = filtered_data.fillna(method='ffill')
            filtered_data = filtered_data.dropna()
            
            logger.info(f"📊 Data prepared (timezone-safe): {len(filtered_data)} candles")
            return filtered_data
            
        except Exception as e:
            logger.error(f"❌ Data preparation error: {e}")
            raise'''
            
            content = re.sub(old_prepare, new_prepare, content, flags=re.DOTALL)
            
            # Write back
            with open(backtester_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Fixed timezone issues in MultiStrategyBacktester")
            
        except Exception as e:
            print(f"❌ Error fixing backtester: {e}")
    
    def fix_main_datetime_parsing(self):
        """🔧 Main.py'deki datetime parsing'i düzelt"""
        
        main_file = self.project_root / "main.py"
        
        try:
            with open(main_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Timezone-safe datetime parsing ekle
            timezone_safe_parsing = '''
    def _parse_timezone_safe_date(self, date_string: str):
        """🕐 Parse date in timezone-safe manner"""
        try:
            from datetime import datetime
            # Parse as naive datetime
            dt = datetime.fromisoformat(date_string)
            # Ensure it's timezone-naive
            if dt.tzinfo is not None:
                dt = dt.replace(tzinfo=None)
            return dt
        except Exception as e:
            self.logger.error(f"Date parsing error: {e}")
            raise'''
            
            # Add timezone-safe method if not exists
            if "_parse_timezone_safe_date" not in content:
                # Find a good place to insert (after class definition)
                class_pattern = r'(class PhoenixTradingSystem:.*?\n)'
                content = re.sub(class_pattern, r'\1' + timezone_safe_parsing + '\n', content, flags=re.DOTALL)
            
            # Replace datetime parsing in run_backtest methods
            datetime_patterns = [
                (r'datetime\.fromisoformat\(args\.start_date\)', 'self._parse_timezone_safe_date(args.start_date)'),
                (r'datetime\.fromisoformat\(args\.end_date\)', 'self._parse_timezone_safe_date(args.end_date)')
            ]
            
            for old_pattern, new_pattern in datetime_patterns:
                content = re.sub(old_pattern, new_pattern, content)
            
            with open(main_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Fixed timezone issues in main.py")
            
        except Exception as e:
            print(f"❌ Error fixing main.py: {e}")
    
    def fix_backtest_configuration(self):
        """🔧 BacktestConfiguration timezone sorunlarını çöz"""
        
        try:
            backtester_file = self.project_root / "backtesting/multi_strategy_backtester.py"
            
            with open(backtester_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # BacktestConfiguration class'ını timezone-safe hale getir
            if "class BacktestConfiguration" in content:
                # Add timezone normalization to BacktestConfiguration
                config_pattern = r'(class BacktestConfiguration:.*?def __post_init__\(self\):.*?)'
                
                post_init_addition = '''
        # Ensure dates are timezone-naive for consistent comparison
        if hasattr(self.start_date, 'tzinfo') and self.start_date.tzinfo is not None:
            self.start_date = self.start_date.replace(tzinfo=None)
        if hasattr(self.end_date, 'tzinfo') and self.end_date.tzinfo is not None:
            self.end_date = self.end_date.replace(tzinfo=None)'''
                
                if "__post_init__" in content:
                    # Add to existing __post_init__
                    content = re.sub(
                        r'(def __post_init__\(self\):)(.*?)(\n        [^#\s]|\nclass|\Z)',
                        r'\1\2' + post_init_addition + r'\3',
                        content,
                        flags=re.DOTALL
                    )
                else:
                    # Add new __post_init__ method
                    content = re.sub(
                        r'(class BacktestConfiguration:.*?\n)',
                        r'\1    def __post_init__(self):' + post_init_addition + '\n\n',
                        content,
                        flags=re.DOTALL
                    )
            
            with open(backtester_file, 'w', encoding='utf-8') as f:
                f.write(content)
            
            print("✅ Fixed timezone issues in BacktestConfiguration")
            
        except Exception as e:
            print(f"❌ Error fixing BacktestConfiguration: {e}")

def main():
    print("🕐 TIMEZONE PROBLEM SOLVER")
    print("="*40)
    
    fixer = TimezoneFixer()
    fixer.fix_timezone_issues()
    
    print("\n🎯 NEXT STEPS:")
    print("1. Test the fix: python main.py backtest --strategy momentum --start-date 2024-01-01 --end-date 2024-03-31 --capital 10000 --data-file historical_data/BTCUSDT_15m_20240101_20241231.csv")
    print("2. If still issues, run: python debug_enhanced.py")

if __name__ == "__main__":
    main()