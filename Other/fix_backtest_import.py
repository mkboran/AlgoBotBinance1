#!/usr/bin/env python3
"""
EMERGENCY IMPORT FIX - Backtest Runner FIXED VERSION
Fixes duplicate alias syntax error 
"""

import re
from pathlib import Path

def fix_backtest_imports():
    """Fix import mismatch in backtest_runner.py - CLEAN VERSION"""
    
    backtest_file = Path("backtest_runner.py")
    
    if not backtest_file.exists():
        print("❌ backtest_runner.py not found!")
        return False
    
    # Read current content
    with open(backtest_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix the duplicate syntax error first
    content = re.sub(
        r'from strategies\.momentum_optimized import EnhancedMomentumStrategy as MomentumStrategy as MomentumStrategy',
        'from strategies.momentum_optimized import EnhancedMomentumStrategy as MomentumStrategy',
        content
    )
    
    # Then handle any remaining incorrect imports
    content = re.sub(
        r'from strategies\.momentum_optimized import MomentumStrategy',
        'from strategies.momentum_optimized import EnhancedMomentumStrategy as MomentumStrategy', 
        content
    )
    
    # Check if any changes made
    if content != original_content:
        # Create backup  
        backup_file = f"backtest_runner_fixed_{int(__import__('time').time())}.py"
        with open(backup_file, 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"📁 Backup created: {backup_file}")
        
        # Write fixed content
        with open(backtest_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ Syntax error fixed in backtest_runner.py")
        print("🔧 Clean import: EnhancedMomentumStrategy as MomentumStrategy")
        return True
    else:
        print("ℹ️ File already correctly formatted")
        return False

if __name__ == "__main__":
    fix_backtest_imports()