#!/usr/bin/env python3
"""
🏗️ PROJE PHOENIX - PROJECT STRUCTURE SETUP
💎 Eksik klasörleri oluşturur ve temel yapıyı hazırlar
"""

import os
from pathlib import Path

def create_project_structure():
    """📁 Proje yapısını oluştur"""
    
    required_dirs = [
        "logs",
        "scripts", 
        "utils",
        "strategies",
        "optimization", 
        "optimization/results",
        "backtesting",
        "backups",
        "historical_data"
    ]
    
    print("🏗️ Proje yapısı oluşturuluyor...")
    
    for dir_path in required_dirs:
        full_path = Path(dir_path)
        if not full_path.exists():
            full_path.mkdir(parents=True, exist_ok=True)
            print(f"✅ Klasör oluşturuldu: {dir_path}")
        else:
            print(f"📁 Klasör zaten mevcut: {dir_path}")
    
    # Logs klasöründe temel dosyaları oluştur
    log_files = [
        "logs/system_validation.log",
        "logs/dependency_analysis.log", 
        "logs/trading.log",
        "logs/backtest.log"
    ]
    
    for log_file in log_files:
        log_path = Path(log_file)
        if not log_path.exists():
            log_path.touch()
            print(f"📝 Log dosyası oluşturuldu: {log_file}")
    
    print("🎉 Proje yapısı hazır!")

if __name__ == "__main__":
    create_project_structure()