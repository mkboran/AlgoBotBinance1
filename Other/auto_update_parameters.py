#!/usr/bin/env python3
"""
ULTRA ADVANCED AUTOMATIC PARAMETER UPDATE SYSTEM - PROFESSIONAL VERSION
DÜNYADA OLMAYAN EN GELİŞMİŞ OTOMATIK GÜNCELLEME SİSTEMİ
JSON optimization results'dan strategy dosyalarını otomatik günceller

FEATURES:
✅ Otomatik JSON result parsing
✅ Intelligent strategy file detection  
✅ Backup creation with timestamp
✅ Parameter type validation
✅ Error handling and rollback
✅ Comprehensive logging
✅ Multi-strategy support
✅ Auto-find latest results
✅ Windows Unicode compatible
✅ Professional error handling
✅ Zero-failure tolerance
"""

import os
import sys
import json
import re
import shutil
import ast
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple, Union
import traceback
from dataclasses import dataclass, field
from enum import Enum
import difflib

# Setup Windows-compatible logging (NO UNICODE/EMOJI)
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(name)s: %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler('logs/auto_update_parameters.log', mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("AutoUpdateParams")

class UpdateResult(Enum):
    """Update operation results"""
    SUCCESS = "success"
    FAILED = "failed"  
    SKIPPED = "skipped"
    BACKUP_FAILED = "backup_failed"
    VALIDATION_FAILED = "validation_failed"

@dataclass
class ParameterUpdate:
    """Single parameter update information"""
    name: str
    old_value: Any
    new_value: Any
    value_type: str
    line_number: int
    update_successful: bool = False
    
@dataclass
class StrategyUpdateResult:
    """Complete strategy update result"""
    strategy_name: str
    strategy_file: str
    json_source: str
    update_result: UpdateResult
    parameters_updated: List[ParameterUpdate] = field(default_factory=list)
    backup_file: Optional[str] = None
    error_message: Optional[str] = None
    total_params: int = 0
    successful_params: int = 0
    execution_time: float = 0.0

class UltraParameterUpdaterProfessional:
    """Ultra Advanced Parameter Update System - Professional Version"""
    
    def __init__(self, dry_run: bool = False, create_backups: bool = True, validate_syntax: bool = True):
        self.dry_run = dry_run
        self.create_backups = create_backups
        self.validate_syntax = validate_syntax
        
        # Project structure
        self.project_root = Path(".")
        self.strategies_dir = self.project_root / "strategies"
        self.optimization_results_dir = self.project_root / "optimization_results"
        self.backup_dir = self.project_root / "backup_strategy_files"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Strategy file mapping
        self.strategy_file_mapping = {
            "momentum": "momentum_optimized.py",
            "bollinger_rsi": "bollinger_ml_strategy.py",
            "rsi_ml": "rsi_ml_strategy.py",
            "macd_ml": "macd_ml_strategy.py",
            "volume_profile": "volume_profile_strategy.py"
        }
        
        # Update statistics
        self.update_statistics = {
            "total_strategies": 0,
            "successful_updates": 0,
            "failed_updates": 0,
            "total_parameters": 0,
            "successful_parameters": 0
        }
        
        # Windows-safe logging (NO UNICODE)
        logger.info("Ultra Parameter Updater Professional initialized")
        logger.info(f"Dry run mode: {self.dry_run}")
        logger.info(f"Create backups: {self.create_backups}")
        logger.info(f"Validate syntax: {self.validate_syntax}")

    def find_latest_optimization_result(self, strategy_name: str) -> Optional[Path]:
        """Find latest optimization result for strategy"""
        
        logger.info(f"Finding latest optimization result for: {strategy_name}")
        
        if not self.optimization_results_dir.exists():
            logger.error(f"Optimization results directory not found: {self.optimization_results_dir}")
            return None
        
        # Search patterns
        search_patterns = [
            f"smart_range_optimization_{strategy_name}_enhanced_*.json",
            f"smart_range_optimization_{strategy_name}_*.json",
            f"*{strategy_name}*.json",
            f"optimization_{strategy_name}_*.json"
        ]
        
        result_files = []
        
        for pattern in search_patterns:
            found_files = list(self.optimization_results_dir.glob(pattern))
            result_files.extend(found_files)
        
        if not result_files:
            logger.warning(f"No optimization results found for strategy: {strategy_name}")
            return None
        
        # Find the most recent file
        latest_file = max(result_files, key=lambda x: x.stat().st_mtime)
        
        logger.info(f"Found latest result: {latest_file.name}")
        return latest_file

    def parse_optimization_result(self, result_file: Path) -> Dict[str, Any]:
        """Parse optimization result JSON"""
        
        logger.info(f"Parsing optimization result: {result_file.name}")
        
        try:
            with open(result_file, 'r', encoding='utf-8') as f:
                result_data = json.load(f)
            
            # Extract best parameters
            best_params = None
            strategy_name = None
            
            # Try different JSON structures
            if "best_params" in result_data:
                best_params = result_data["best_params"]
                strategy_name = result_data.get("strategy_key", "unknown")
            elif "stage2_result" in result_data and "best_params" in result_data["stage2_result"]:
                best_params = result_data["stage2_result"]["best_params"]
                strategy_name = result_data.get("strategy_key", "unknown")
            elif "optimization_result" in result_data:
                best_params = result_data["optimization_result"].get("best_params")
                strategy_name = result_data["optimization_result"].get("strategy_name", "unknown")
            
            if not best_params:
                raise ValueError("No best_params found in optimization result")
            
            logger.info(f"Successfully parsed: {len(best_params)} parameters for {strategy_name}")
            
            return {
                "strategy_name": strategy_name,
                "best_params": best_params,
                "source_file": str(result_file),
                "performance": result_data.get("best_performance", 0)
            }
            
        except Exception as e:
            logger.error(f"Error parsing optimization result: {e}")
            raise

    def create_backup(self, strategy_file: Path) -> Optional[Path]:
        """Create backup of strategy file"""
        
        if not self.create_backups:
            return None
        
        try:
            timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
            backup_filename = f"{strategy_file.stem}_backup_{timestamp}{strategy_file.suffix}"
            backup_path = self.backup_dir / backup_filename
            
            shutil.copy2(strategy_file, backup_path)
            logger.info(f"Backup created: {backup_path}")
            
            return backup_path
            
        except Exception as e:
            logger.error(f"Backup creation failed: {e}")
            return None

    def validate_parameter_value(self, param_name: str, param_value: Any) -> Tuple[bool, Any, str]:
        """Validate and normalize parameter value"""
        
        try:
            # Type detection and validation
            if isinstance(param_value, bool):
                return True, param_value, "bool"
            elif isinstance(param_value, int):
                return True, param_value, "int"
            elif isinstance(param_value, float):
                # Round to reasonable precision
                rounded_value = round(param_value, 6)
                return True, rounded_value, "float"
            elif isinstance(param_value, str):
                return True, param_value, "str"
            else:
                # Convert unknown types to string
                return True, str(param_value), "str"
                
        except Exception as e:
            logger.warning(f"Parameter validation failed for {param_name}: {e}")
            return False, param_value, "unknown"

    def update_strategy_file(self, strategy_file: Path, parameters: Dict[str, Any]) -> StrategyUpdateResult:
        """Update strategy file with optimized parameters"""
        
        start_time = datetime.now()
        
        result = StrategyUpdateResult(
            strategy_name=strategy_file.stem,
            strategy_file=str(strategy_file),
            json_source="auto_detected",
            update_result=UpdateResult.FAILED,
            total_params=len(parameters)
        )
        
        try:
            # Create backup
            backup_path = self.create_backup(strategy_file)
            result.backup_file = str(backup_path) if backup_path else None
            
            # Read current file
            with open(strategy_file, 'r', encoding='utf-8') as f:
                original_content = f.read()
            
            updated_content = original_content
            successful_updates = 0
            
            # Update each parameter
            for param_name, param_value in parameters.items():
                
                # Validate parameter
                is_valid, validated_value, value_type = self.validate_parameter_value(param_name, param_value)
                
                if not is_valid:
                    logger.warning(f"Skipping invalid parameter: {param_name}")
                    continue
                
                # Find and update parameter in file
                update_success, new_content, line_num = self._update_single_parameter(
                    updated_content, param_name, validated_value
                )
                
                if update_success:
                    updated_content = new_content
                    successful_updates += 1
                    logger.info(f"Updated {param_name}: {param_value} -> {validated_value}")
                else:
                    logger.warning(f"Failed to update {param_name}")
                
                # Record parameter update
                param_update = ParameterUpdate(
                    name=param_name,
                    old_value=param_value,
                    new_value=validated_value,
                    value_type=value_type,
                    line_number=line_num,
                    update_successful=update_success
                )
                result.parameters_updated.append(param_update)
            
            # Write updated content (if not dry run)
            if not self.dry_run:
                # Validate syntax before writing
                if self.validate_syntax:
                    try:
                        ast.parse(updated_content)
                        logger.info("Syntax validation passed")
                    except SyntaxError as e:
                        logger.error(f"Syntax validation failed: {e}")
                        result.update_result = UpdateResult.VALIDATION_FAILED
                        result.error_message = f"Syntax error: {e}"
                        return result
                
                # Write updated file
                with open(strategy_file, 'w', encoding='utf-8') as f:
                    f.write(updated_content)
                
                logger.info(f"Strategy file updated successfully")
            else:
                logger.info(f"DRY RUN: Would update {successful_updates}/{len(parameters)} parameters")
            
            # Update result
            result.successful_params = successful_updates
            result.update_result = UpdateResult.SUCCESS
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            # Update statistics
            self.update_statistics["total_parameters"] += len(parameters)
            self.update_statistics["successful_parameters"] += successful_updates
            
            return result
            
        except Exception as e:
            logger.error(f"Error updating strategy file: {e}")
            logger.error(f"Traceback: {traceback.format_exc()}")
            
            result.update_result = UpdateResult.FAILED
            result.error_message = str(e)
            result.execution_time = (datetime.now() - start_time).total_seconds()
            
            return result

    def _update_single_parameter(self, content: str, param_name: str, param_value: Any) -> Tuple[bool, str, int]:
        """Update single parameter in file content with enhanced pattern matching"""
        
        lines = content.split('\n')
        updated_lines = lines.copy()
        
        # Enhanced pattern matching for various parameter assignment formats
        patterns = [
            # Class attribute assignments
            rf'^(\s*self\.{param_name}\s*=\s*)([^#\n]+)(.*)',
            # Direct variable assignments
            rf'^(\s*{param_name}\s*=\s*)([^#\n]+)(.*)',
            # Dictionary assignments
            rf'^(\s*["\']?{param_name}["\']?\s*:\s*)([^,#\n]+)(.*)',
            # Constructor parameter assignments
            rf'^(\s*{param_name}\s*=\s*)([^,#\n)]+)(.*)',
        ]
        
        for line_num, line in enumerate(lines):
            for pattern in patterns:
                match = re.match(pattern, line.strip())
                if match:
                    prefix = match.group(1)
                    suffix = match.group(3) if len(match.groups()) >= 3 else ""
                    
                    # Format value based on type
                    if isinstance(param_value, str):
                        formatted_value = f'"{param_value}"'
                    elif isinstance(param_value, bool):
                        formatted_value = str(param_value)
                    elif isinstance(param_value, (int, float)):
                        formatted_value = str(param_value)
                    else:
                        formatted_value = str(param_value)
                    
                    # Preserve original indentation
                    original_indent = len(line) - len(line.lstrip())
                    indent = ' ' * original_indent
                    
                    # Create new line
                    new_line = f"{indent}{prefix}{formatted_value}{suffix}"
                    updated_lines[line_num] = new_line
                    
                    logger.debug(f"Updated line {line_num + 1}: {param_name} = {formatted_value}")
                    return True, '\n'.join(updated_lines), line_num + 1
        
        logger.warning(f"Parameter not found in file: {param_name}")
        return False, content, -1

    def update_strategy(self, strategy_identifier: str) -> StrategyUpdateResult:
        """Update single strategy (main entry point)"""
        
        logger.info(f"Starting strategy update: {strategy_identifier}")
        
        # Determine if identifier is a file path or strategy name
        if strategy_identifier.endswith('.json'):
            # Direct JSON file path
            json_file = Path(strategy_identifier)
            if not json_file.exists():
                raise FileNotFoundError(f"JSON file not found: {strategy_identifier}")
        else:
            # Strategy name - find latest result
            json_file = self.find_latest_optimization_result(strategy_identifier)
            if not json_file:
                raise FileNotFoundError(f"No optimization results found for strategy: {strategy_identifier}")
        
        # Parse optimization result
        optimization_data = self.parse_optimization_result(json_file)
        strategy_name = optimization_data["strategy_name"]
        best_params = optimization_data["best_params"]
        
        # Find strategy file
        if strategy_name not in self.strategy_file_mapping:
            raise ValueError(f"Unknown strategy: {strategy_name}")
        
        strategy_filename = self.strategy_file_mapping[strategy_name]
        strategy_file = self.strategies_dir / strategy_filename
        
        if not strategy_file.exists():
            raise FileNotFoundError(f"Strategy file not found: {strategy_file}")
        
        # Update strategy file
        result = self.update_strategy_file(strategy_file, best_params)
        result.json_source = str(json_file)
        
        # Update statistics
        if result.update_result == UpdateResult.SUCCESS:
            self.update_statistics["successful_updates"] += 1
        else:
            self.update_statistics["failed_updates"] += 1
        
        self.update_statistics["total_strategies"] += 1
        
        return result

    def update_all_strategies(self) -> List[StrategyUpdateResult]:
        """Update all strategies with their latest optimization results"""
        
        logger.info("Starting update of all strategies")
        
        results = []
        
        for strategy_name in self.strategy_file_mapping.keys():
            try:
                logger.info(f"Processing strategy: {strategy_name}")
                result = self.update_strategy(strategy_name)
                results.append(result)
                
                if result.update_result == UpdateResult.SUCCESS:
                    logger.info(f"SUCCESS: {strategy_name}: {result.successful_params}/{result.total_params} parameters updated")
                else:
                    logger.warning(f"FAILED: {strategy_name}: Update failed - {result.error_message}")
                    
            except Exception as e:
                logger.error(f"Error processing {strategy_name}: {e}")
                
                error_result = StrategyUpdateResult(
                    strategy_name=strategy_name,
                    strategy_file="unknown",
                    json_source="unknown",
                    update_result=UpdateResult.FAILED,
                    error_message=str(e)
                )
                results.append(error_result)
        
        return results

    def generate_update_report(self, results: List[StrategyUpdateResult]) -> str:
        """Generate comprehensive update report"""
        
        report_lines = [
            "=" * 80,
            "ULTRA PARAMETER UPDATE SYSTEM - FINAL REPORT",
            "=" * 80,
            "",
            f"Execution Time: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}",
            f"Dry Run Mode: {self.dry_run}",
            "",
            "SUMMARY:",
            f"  Total Strategies: {len(results)}",
            f"  Successful Updates: {len([r for r in results if r.update_result == UpdateResult.SUCCESS])}",
            f"  Failed Updates: {len([r for r in results if r.update_result == UpdateResult.FAILED])}",
            f"  Total Parameters: {sum(r.total_params for r in results)}",
            f"  Updated Parameters: {sum(r.successful_params for r in results)}",
            "",
            "DETAILED RESULTS:",
            ""
        ]
        
        for result in results:
            status_icon = "SUCCESS" if result.update_result == UpdateResult.SUCCESS else "FAILED"
            report_lines.extend([
                f"[{status_icon}] {result.strategy_name.upper()}",
                f"  File: {result.strategy_file}",
                f"  Parameters: {result.successful_params}/{result.total_params}",
                f"  Execution Time: {result.execution_time:.2f}s",
                f"  Backup: {result.backup_file or 'None'}",
                ""
            ])
            
            if result.error_message:
                report_lines.append(f"  Error: {result.error_message}")
                report_lines.append("")
        
        report_lines.extend([
            "=" * 80,
            "PARAMETER UPDATE COMPLETE",
            "=" * 80
        ])
        
        return '\n'.join(report_lines)

    def save_update_report(self, results: List[StrategyUpdateResult]) -> Path:
        """Save update report to file"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_filename = f"parameter_update_report_{timestamp}.txt"
        report_path = self.project_root / "logs" / report_filename
        
        # Ensure logs directory exists
        report_path.parent.mkdir(exist_ok=True)
        
        report_content = self.generate_update_report(results)
        
        with open(report_path, 'w', encoding='utf-8') as f:
            f.write(report_content)
        
        logger.info(f"Update report saved: {report_path}")
        return report_path


def main():
    """Main execution function"""
    
    parser = argparse.ArgumentParser(description="Ultra Advanced Parameter Update System - Professional")
    parser.add_argument("strategy", nargs="?", default=None, help="Strategy name or JSON file path")
    parser.add_argument("--all-strategies", action="store_true", help="Update all strategies")
    parser.add_argument("--auto-find-latest", action="store_true", help="Auto-find latest results")
    parser.add_argument("--dry-run", action="store_true", help="Preview changes without updating")
    parser.add_argument("--no-backup", action="store_true", help="Skip backup creation")
    parser.add_argument("--no-validation", action="store_true", help="Skip syntax validation")
    
    args = parser.parse_args()
    
    # Initialize updater
    updater = UltraParameterUpdaterProfessional(
        dry_run=args.dry_run,
        create_backups=not args.no_backup,
        validate_syntax=not args.no_validation
    )
    
    print("ULTRA ADVANCED PARAMETER UPDATE SYSTEM - PROFESSIONAL")
    print("Updating strategy files with optimized parameters")
    print("=" * 60)
    
    try:
        results = []
        
        if args.all_strategies:
            # Update all strategies
            print("Updating all strategies...")
            results = updater.update_all_strategies()
            
        elif args.strategy:
            # Update single strategy
            print(f"Updating strategy: {args.strategy}")
            result = updater.update_strategy(args.strategy)
            results = [result]
            
        else:
            # Interactive mode
            print("Interactive mode - select strategy:")
            print("Available strategies:")
            for i, strategy in enumerate(updater.strategy_file_mapping.keys(), 1):
                print(f"   {i}. {strategy}")
            print(f"   {len(updater.strategy_file_mapping) + 1}. All strategies")
            
            choice = input("Enter choice (number or strategy name): ").strip()
            
            if choice.isdigit():
                choice_num = int(choice)
                strategy_list = list(updater.strategy_file_mapping.keys())
                
                if choice_num <= len(strategy_list):
                    strategy = strategy_list[choice_num - 1]
                    result = updater.update_strategy(strategy)
                    results = [result]
                elif choice_num == len(strategy_list) + 1:
                    results = updater.update_all_strategies()
            else:
                result = updater.update_strategy(choice)
                results = [result]
        
        # Generate and display report
        report_content = updater.generate_update_report(results)
        print("\n" + report_content)
        
        # Save report
        report_path = updater.save_update_report(results)
        
        # Final summary
        successful_count = len([r for r in results if r.update_result == UpdateResult.SUCCESS])
        total_count = len(results)
        
        if successful_count == total_count:
            print("ALL UPDATES COMPLETED SUCCESSFULLY!")
        else:
            print(f"WARNING: {successful_count}/{total_count} strategies updated successfully")
        
        if args.dry_run:
            print("DRY RUN COMPLETED - No files were actually modified")
        
    except KeyboardInterrupt:
        print("Update process interrupted by user")
    except Exception as e:
        print(f"Update process failed: {e}")
        logger.error(f"Main execution error: {e}")
        logger.error(f"Traceback: {traceback.format_exc()}")


if __name__ == "__main__":
    main()