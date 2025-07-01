#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
BASIT IMPORT TEST SCRIPT'I
Import sorunlarını hızlıca test etmek için

Bu script import'ları manuel olarak test eder.
"""

import sys
import os
from pathlib import Path

# Python path'e proje kökünü ekle
PROJECT_ROOT = Path(__file__).parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

print("IMPORT TEST BASLIYOR...")
print(f"Proje koku: {PROJECT_ROOT}")
print(f"Python path'ler:")
for i, path in enumerate(sys.path[:5]):
    print(f"   {i+1}. {path}")
print()

# Test edilecek modüller
test_modules = [
    "utils",
    "utils.config", 
    "utils.logger",
    "utils.portfolio",
    "strategies",
    "strategies.momentum_optimized",
    "backtest_runner",
    "optimization.objective_fixed"
]

successful = []
failed = []

for module_name in test_modules:
    try:
        import importlib
        if module_name in sys.modules:
            importlib.reload(sys.modules[module_name])
        else:
            importlib.import_module(module_name)
        
        successful.append(module_name)
        print(f"SUCCESS {module_name}")
        
    except Exception as e:
        failed.append((module_name, str(e)))
        print(f"FAILED {module_name} - {e}")

print()
print("="*60)
print(f"SONUC: {len(successful)}/{len(test_modules)} import basarili")
print(f"BASARILI: {', '.join(successful)}")
if failed:
    print(f"BASARISIZ: {', '.join([f[0] for f in failed])}")
print("="*60)

if len(successful) >= 6:
    print("Import'lar buyuk olcude basarili!")
    print("Validation script'i tekrar calistirabilirsiniz")
else:
    print("Import sorunlari devam ediyor")
    print("Detayli hata analizi:")
    for module, error in failed:
        print(f"   HATA {module}: {error}")