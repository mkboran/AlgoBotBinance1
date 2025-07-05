#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - JSON PARAMETER SYSTEM FIX
💎 FIXED: strategy_name KeyError hatası giderildi

ÇÖZÜMLER:
1. ✅ save_optimization_results strategy_name'i doğru kaydediyor
2. ✅ load_strategy_parameters strategy_name'i doğru döndürüyor
3. ✅ Backward compatibility sağlandı
"""

import json
import os
import shutil
import argparse
import logging
from pathlib import Path
from datetime import datetime, timezone, timedelta
from typing import Dict, Any, Optional, List, Union, Tuple
import pandas as pd
import numpy as np

# Configure logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s'
)
logger = logging.getLogger("JSONParameterSystem")

class ParameterValidationError(Exception):
    """Parameter validation error"""
    pass

class JSONParameterManager:
    """
    💎 JSON Parameter Management System - HEDGE FUND LEVEL
    🚀 Zero Source Code Modification Strategy
    """
    
    def __init__(self, results_dir: str = "optimization/results"):
        self.results_dir = Path(results_dir)
        self.results_dir.mkdir(parents=True, exist_ok=True)
        
        # Backup directory
        self.backup_dir = self.results_dir / "backups"
        self.backup_dir.mkdir(exist_ok=True)
        
        # Optimization directory
        self.optimization_dir = Path("optimization")
        self.optimization_dir.mkdir(exist_ok=True)
        
        logger.info(f"💎 JSON Parameter Manager initialized")
        logger.info(f"📁 Results directory: {self.results_dir}")
        logger.info(f"💾 Backup directory: {self.backup_dir}")
    
    def save_optimization_results(self, 
                                strategy_name: str,
                                best_parameters: Dict[str, Any],
                                optimization_metrics: Dict[str, Any],
                                additional_metadata: Optional[Dict[str, Any]] = None) -> bool:
        """
        💾 Save optimization results with comprehensive metadata
        ✅ FIXED: strategy_name properly saved in root level
        """
        
        try:
            timestamp = datetime.now(timezone.utc)
            
            # Create comprehensive parameter file
            parameter_data = {
                "strategy_name": strategy_name,  # ✅ ROOT LEVEL
                "version": "2.0",
                "timestamp": timestamp.isoformat(),
                "parameters": best_parameters,    # Parameters nested here
                "metrics": {
                    **optimization_metrics,
                    "timestamp": timestamp.isoformat()
                },
                "metadata": {
                    "optimization_date": timestamp.strftime("%Y-%m-%d"),
                    "optimization_time": timestamp.strftime("%H:%M:%S UTC"),
                    "platform": "Phoenix Trading System v2.0",
                    "parameter_count": len(best_parameters),
                    **(additional_metadata or {})
                },
                "validation": {
                    "validated": False,
                    "validation_date": None,
                    "validation_metrics": {}
                }
            }
            
            # File paths
            filename = f"{strategy_name}_best_params.json"
            filepath = self.results_dir / filename
            backup_filepath = self.backup_dir / f"{strategy_name}_backup_{timestamp.strftime('%Y%m%d_%H%M%S')}.json"
            
            # Create backup if file exists
            if filepath.exists():
                shutil.copy2(filepath, backup_filepath)
                logger.info(f"💾 Created backup: {backup_filepath.name}")
            
            # Save parameter file
            with open(filepath, 'w', encoding='utf-8') as f:
                json.dump(parameter_data, f, indent=2, ensure_ascii=False)
            
            logger.info(f"✅ Saved optimization results for {strategy_name}")
            logger.info(f"📊 Best score: {optimization_metrics.get('best_score', 'N/A')}")
            logger.info(f"📁 File: {filepath}")
            
            # Also save a simplified version for quick access
            simple_filepath = self.results_dir / f"{strategy_name}_params_only.json"
            simple_data = {
                "strategy_name": strategy_name,
                "parameters": best_parameters,
                "last_updated": timestamp.isoformat()
            }
            
            with open(simple_filepath, 'w') as f:
                json.dump(simple_data, f, indent=2)
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error saving optimization results: {e}")
            return False
    
    def load_strategy_parameters(self, 
                               strategy_name: str,
                               use_backup: bool = False) -> Optional[Dict[str, Any]]:
        """
        📥 Load strategy parameters from JSON
        ✅ FIXED: Always returns data with strategy_name at root level
        """
        
        try:
            if use_backup:
                # Find latest backup
                backup_files = sorted(self.backup_dir.glob(f"{strategy_name}_backup_*.json"))
                if not backup_files:
                    logger.error(f"❌ No backup files found for {strategy_name}")
                    return None
                filepath = backup_files[-1]
            else:
                filepath = self.results_dir / f"{strategy_name}_best_params.json"
            
            if not filepath.exists():
                logger.warning(f"⚠️ Parameter file not found: {filepath}")
                return None
            
            with open(filepath, 'r', encoding='utf-8') as f:
                data = json.load(f)
            
            # ✅ ENSURE strategy_name EXISTS AT ROOT LEVEL
            if 'strategy_name' not in data:
                data['strategy_name'] = strategy_name
            
            # ✅ ENSURE parameters EXISTS
            if 'parameters' not in data:
                # Handle old format where parameters might be at root
                parameters = {}
                for key, value in data.items():
                    if key not in ['strategy_name', 'version', 'timestamp', 'metrics', 'metadata', 'validation']:
                        parameters[key] = value
                data['parameters'] = parameters
            
            logger.info(f"✅ Loaded parameters for {strategy_name}")
            logger.info(f"📊 Parameter count: {len(data.get('parameters', {}))}")
            logger.info(f"📅 Last updated: {data.get('timestamp', 'Unknown')}")
            
            return data
            
        except json.JSONDecodeError as e:
            logger.error(f"❌ JSON decode error: {e}")
            return None
        except Exception as e:
            logger.error(f"❌ Error loading parameters: {e}")
            return None
    
    def update_strategy_parameters(self,
                                 strategy_name: str,
                                 parameter_updates: Dict[str, Any],
                                 reason: str = "Manual update") -> bool:
        """📝 Update specific strategy parameters"""
        
        try:
            # Load current parameters
            current_data = self.load_strategy_parameters(strategy_name)
            if not current_data:
                logger.error(f"❌ Cannot update - no existing parameters for {strategy_name}")
                return False
            
            # Update parameters
            current_params = current_data.get('parameters', {})
            current_params.update(parameter_updates)
            
            # Update metadata
            update_metadata = {
                "last_update_reason": reason,
                "parameters_updated": list(parameter_updates.keys()),
                "update_count": current_data.get('metadata', {}).get('update_count', 0) + 1
            }
            
            # Save updated parameters
            optimization_metrics = current_data.get('metrics', {})
            return self.save_optimization_results(
                strategy_name=strategy_name,
                best_parameters=current_params,
                optimization_metrics=optimization_metrics,
                additional_metadata=update_metadata
            )
            
        except Exception as e:
            logger.error(f"❌ Error updating parameters: {e}")
            return False
    
    def validate_parameters(self, 
                          strategy_name: str,
                          parameters: Dict[str, Any]) -> Tuple[bool, List[str]]:
        """🔍 Validate strategy parameters"""
        
        errors = []
        
        # Strategy-specific validation rules
        validation_rules = {
            "momentum": {
                "ema_short": lambda x: 5 <= x <= 50,
                "ema_medium": lambda x: 10 <= x <= 100,
                "ema_long": lambda x: 20 <= x <= 200,
                "rsi_period": lambda x: 5 <= x <= 30,
                "position_size_pct": lambda x: 0.01 <= x <= 1.0,
            },
            "bollinger_rsi": {
                "bb_period": lambda x: 10 <= x <= 50,
                "bb_std_dev": lambda x: 1.0 <= x <= 3.0,
                "rsi_period": lambda x: 5 <= x <= 30,
            }
        }
        
        if strategy_name in validation_rules:
            rules = validation_rules[strategy_name]
            
            for param_name, validate_fn in rules.items():
                if param_name in parameters:
                    if not validate_fn(parameters[param_name]):
                        errors.append(f"{param_name} value {parameters[param_name]} is out of valid range")
                else:
                    errors.append(f"Missing required parameter: {param_name}")
        
        is_valid = len(errors) == 0
        return is_valid, errors
    
    def get_all_strategy_parameters(self) -> Dict[str, Dict[str, Any]]:
        """📚 Get all saved strategy parameters"""
        
        all_parameters = {}
        
        for json_file in self.results_dir.glob("*_best_params.json"):
            strategy_name = json_file.stem.replace("_best_params", "")
            
            if strategy_name.endswith("_backup"):
                continue
                
            params = self.load_strategy_parameters(strategy_name)
            if params:
                all_parameters[strategy_name] = params
        
        return all_parameters
    
    def export_parameters_to_code(self, 
                                strategy_name: str,
                                output_file: Optional[str] = None) -> bool:
        """
        📤 Export parameters to Python code format
        Useful for embedding optimized parameters directly
        """
        
        try:
            data = self.load_strategy_parameters(strategy_name)
            if not data:
                return False
            
            parameters = data.get('parameters', {})
            
            # Generate Python code
            code_lines = [
                f"# Optimized parameters for {strategy_name}",
                f"# Generated: {datetime.now(timezone.utc).isoformat()}",
                f"# Best score: {data.get('metrics', {}).get('best_score', 'N/A')}",
                "",
                f"{strategy_name.upper()}_PARAMETERS = {{"
            ]
            
            for param_name, param_value in sorted(parameters.items()):
                if isinstance(param_value, str):
                    code_lines.append(f'    "{param_name}": "{param_value}",')
                else:
                    code_lines.append(f'    "{param_name}": {param_value},')
            
            code_lines.append("}")
            
            # Output
            if output_file:
                with open(output_file, 'w') as f:
                    f.write('\n'.join(code_lines))
                logger.info(f"✅ Exported parameters to {output_file}")
            else:
                print('\n'.join(code_lines))
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Error exporting parameters: {e}")
            return False
    
    def create_parameter_report(self, output_dir: str = "reports") -> bool:
        """📊 Create comprehensive parameter report"""
        
        try:
            output_path = Path(output_dir)
            output_path.mkdir(exist_ok=True)
            
            all_params = self.get_all_strategy_parameters()
            
            # Create report
            report_lines = [
                "# Phoenix Trading System - Parameter Report",
                f"Generated: {datetime.now(timezone.utc).strftime('%Y-%m-%d %H:%M:%S UTC')}",
                f"Total Strategies: {len(all_params)}",
                "",
                "## Strategy Parameters Overview",
                ""
            ]
            
            for strategy_name, data in sorted(all_params.items()):
                params = data.get('parameters', {})
                metrics = data.get('metrics', {})
                
                report_lines.extend([
                    f"### {strategy_name.upper()}",
                    f"- Last Updated: {data.get('timestamp', 'Unknown')}",
                    f"- Best Score: {metrics.get('best_score', 'N/A')}",
                    f"- Parameter Count: {len(params)}",
                    "",
                    "#### Parameters:",
                    "```json",
                    json.dumps(params, indent=2),
                    "```",
                    ""
                ])
            
            # Save report
            report_file = output_path / f"parameter_report_{datetime.now().strftime('%Y%m%d_%H%M%S')}.md"
            with open(report_file, 'w') as f:
                f.write('\n'.join(report_lines))
            
            logger.info(f"✅ Created parameter report: {report_file}")
            return True
            
        except Exception as e:
            logger.error(f"❌ Error creating report: {e}")
            return False


def main():
    """CLI interface for parameter management"""
    parser = argparse.ArgumentParser(description="Phoenix JSON Parameter Manager")
    
    subparsers = parser.add_subparsers(dest='command', help='Commands')
    
    # Save command
    save_parser = subparsers.add_parser('save', help='Save optimization results')
    save_parser.add_argument('--strategy', required=True, help='Strategy name')
    save_parser.add_argument('--params', required=True, help='Parameters JSON file')
    save_parser.add_argument('--score', type=float, help='Optimization score')
    
    # Load command
    load_parser = subparsers.add_parser('load', help='Load strategy parameters')
    load_parser.add_argument('--strategy', required=True, help='Strategy name')
    load_parser.add_argument('--backup', action='store_true', help='Load from backup')
    
    # Update command
    update_parser = subparsers.add_parser('update', help='Update parameters')
    update_parser.add_argument('--strategy', required=True, help='Strategy name')
    update_parser.add_argument('--param', required=True, help='Parameter name')
    update_parser.add_argument('--value', required=True, help='New value')
    
    # List command
    subparsers.add_parser('list', help='List all saved parameters')
    
    # Export command
    export_parser = subparsers.add_parser('export', help='Export to Python code')
    export_parser.add_argument('--strategy', required=True, help='Strategy name')
    export_parser.add_argument('--output', help='Output file')
    
    # Report command
    subparsers.add_parser('report', help='Generate parameter report')
    
    args = parser.parse_args()
    
    # Initialize manager
    manager = JSONParameterManager()
    
    # Execute commands
    if args.command == 'save':
        with open(args.params, 'r') as f:
            params = json.load(f)
        
        metrics = {'best_score': args.score} if args.score else {}
        success = manager.save_optimization_results(
            strategy_name=args.strategy,
            best_parameters=params,
            optimization_metrics=metrics
        )
        
        if success:
            print(f"✅ Saved parameters for {args.strategy}")
        else:
            print(f"❌ Failed to save parameters")
    
    elif args.command == 'load':
        data = manager.load_strategy_parameters(
            strategy_name=args.strategy,
            use_backup=args.backup
        )
        
        if data:
            print(json.dumps(data, indent=2))
        else:
            print(f"❌ No parameters found for {args.strategy}")
    
    elif args.command == 'update':
        # Parse value
        try:
            value = json.loads(args.value)
        except:
            value = args.value
        
        success = manager.update_strategy_parameters(
            strategy_name=args.strategy,
            parameter_updates={args.param: value}
        )
        
        if success:
            print(f"✅ Updated {args.param} for {args.strategy}")
        else:
            print(f"❌ Failed to update parameter")
    
    elif args.command == 'list':
        all_params = manager.get_all_strategy_parameters()
        
        print("\n📊 Saved Strategy Parameters:")
        print("-" * 50)
        
        for strategy, data in sorted(all_params.items()):
            metrics = data.get('metrics', {})
            print(f"\n{strategy.upper()}:")
            print(f"  Last Updated: {data.get('timestamp', 'Unknown')}")
            print(f"  Best Score: {metrics.get('best_score', 'N/A')}")
            print(f"  Parameters: {len(data.get('parameters', {}))}")
    
    elif args.command == 'export':
        success = manager.export_parameters_to_code(
            strategy_name=args.strategy,
            output_file=args.output
        )
        
        if not success:
            print(f"❌ Failed to export parameters")
    
    elif args.command == 'report':
        success = manager.create_parameter_report()
        
        if success:
            print("✅ Parameter report created")
        else:
            print("❌ Failed to create report")


if __name__ == "__main__":
    main()
