#!/usr/bin/env python3
"""
EMERGENCY ML PREDICTOR INTERFACE FIX
Fixes parameter mismatch between AdvancedMLPredictor and momentum strategy
"""

import re
from pathlib import Path

def fix_ml_predictor_interface():
    """Fix ML predictor instantiation in momentum_optimized.py"""
    
    strategy_file = Path("strategies/momentum_optimized.py")
    
    if not strategy_file.exists():
        print("❌ momentum_optimized.py not found!")
        return False
    
    # Read current content
    with open(strategy_file, 'r', encoding='utf-8') as f:
        content = f.read()
    
    original_content = content
    
    # Fix ML predictor instantiation
    # Current: AdvancedMLPredictor(lookback_window=100, prediction_horizon=4)
    # Fixed: AdvancedMLPredictor(prediction_horizon=4, confidence_threshold=0.6)
    
    # Pattern to find and replace ML predictor instantiation
    ml_predictor_pattern = r'self\.ml_predictor = AdvancedMLPredictor\(\s*lookback_window=\d+,\s*prediction_horizon=(\d+)\s*\)'
    
    # Replacement with correct parameters
    ml_predictor_replacement = r'self.ml_predictor = AdvancedMLPredictor(\n            prediction_horizon=\1,\n            confidence_threshold=self.ml_confidence_threshold,\n            auto_retrain=True,\n            feature_importance_tracking=True\n        )'
    
    # Apply the fix
    content = re.sub(ml_predictor_pattern, ml_predictor_replacement, content)
    
    # Alternative pattern if the above doesn't match
    if content == original_content:
        # Try more flexible pattern
        alt_pattern = r'AdvancedMLPredictor\([^)]*lookback_window[^)]*\)'
        alt_replacement = 'AdvancedMLPredictor(\n            prediction_horizon=4,\n            confidence_threshold=self.ml_confidence_threshold,\n            auto_retrain=True\n        )'
        content = re.sub(alt_pattern, alt_replacement, content)
    
    # Check if any changes made
    if content != original_content:
        # Create backup
        backup_file = f"momentum_optimized_ml_fix_backup_{int(__import__('time').time())}.py"
        with open(f"strategies/{backup_file}", 'w', encoding='utf-8') as f:
            f.write(original_content)
        print(f"📁 Backup created: strategies/{backup_file}")
        
        # Write fixed content
        with open(strategy_file, 'w', encoding='utf-8') as f:
            f.write(content)
        
        print("✅ ML Predictor interface fixed")
        print("🔧 Fixed: lookback_window → prediction_horizon + confidence_threshold")
        return True
    else:
        print("ℹ️ No ML predictor fixes needed")
        return False

if __name__ == "__main__":
    fix_ml_predictor_interface()