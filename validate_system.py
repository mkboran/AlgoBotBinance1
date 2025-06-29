#!/usr/bin/env python3
"""
🛡️ PROJE PHOENIX - FAZ 0: SİSTEM DOĞRULAMA VE CI/CD TEMELLERİ
💎 Otomatik sistem koruması ve pre-commit hook sistemi

Bu sistem şunları yapar:
1. ✅ Temel importların çalışıp çalışmadığını kontrol eder
2. ✅ Ana sınıfların (Portfolio, Strategy vb.) başlatılabilirliğini test eder
3. ✅ Kritik dosyaların varlığını doğrular
4. ✅ Kod kalitesi kontrolü yapar
5. ✅ Pre-commit hook olarak kullanılabilir
6. ✅ CI/CD pipeline için exit code döndürür

KULLANIM:
python validate_system.py --full-validation        # Tam doğrulama
python validate_system.py --pre-commit            # Pre-commit kontrolü
python validate_system.py --ci-cd                 # CI/CD pipeline kontrolü
python validate_system.py --install-hooks         # Git hooks kurulumu
"""

import sys
import os
import importlib
import subprocess
import traceback
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional, Tuple
import argparse
import json
import ast
import tempfile

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs") / "system_validation.log", mode='a', encoding='utf-8')
    ]
)
logger = logging.getLogger("SystemValidator")

class ValidationResult:
    """Doğrulama sonuç sınıfı"""
    
    def __init__(self, name: str, passed: bool, message: str = "", details: Any = None):
        self.name = name
        self.passed = passed
        self.message = message
        self.details = details or {}
        self.timestamp = datetime.now(timezone.utc)

class PhoenixSystemValidator:
    """🛡️ Phoenix Sistem Doğrulama Motoru"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
        # Logs klasörünü oluştur
        (self.project_root / "logs").mkdir(exist_ok=True)
        
        # Doğrulama sonuçları
        self.validation_results = []
        self.critical_failures = []
        self.warnings = []
        
        # Kritik dosyalar ve modüller
        self.critical_files = [
            "utils/config.py",
            "utils/portfolio.py",
            "utils/logger.py",
            "strategies/momentum_optimized.py",
            "main.py",
            "backtest_runner.py"
        ]
        
        self.critical_imports = [
            "pandas",
            "numpy", 
            "ccxt",
            "utils.config",
            "utils.portfolio",
            "utils.logger"
        ]
        
        self.critical_classes = [
            ("utils.portfolio", "Portfolio"),
            ("strategies.momentum_optimized", "EnhancedMomentumStrategy"),
            ("backtest_runner", "MomentumBacktester")
        ]
        
        logger.info("🛡️ Phoenix System Validator başlatıldı")
        logger.info(f"📁 Proje kökü: {self.project_root.absolute()}")
    
    def add_result(self, result: ValidationResult) -> None:
        """Doğrulama sonucu ekle"""
        self.validation_results.append(result)
        
        if not result.passed:
            if "critical" in result.name.lower():
                self.critical_failures.append(result)
            else:
                self.warnings.append(result)
    
    def validate_critical_files(self) -> bool:
        """📄 Kritik dosyaların varlığını kontrol et"""
        
        logger.info("📄 Kritik dosya kontrolü...")
        
        all_files_exist = True
        missing_files = []
        
        for file_path in self.critical_files:
            full_path = self.project_root / file_path
            
            if full_path.exists():
                # Dosya boyutu kontrolü
                size_kb = full_path.stat().st_size / 1024
                
                if size_kb < 0.1:  # 100 byte'dan küçükse
                    self.add_result(ValidationResult(
                        name=f"critical_file_size_{file_path}",
                        passed=False,
                        message=f"Kritik dosya çok küçük: {file_path} ({size_kb:.1f} KB)",
                        details={"file_path": file_path, "size_kb": size_kb}
                    ))
                    all_files_exist = False
                else:
                    self.add_result(ValidationResult(
                        name=f"critical_file_{file_path}",
                        passed=True,
                        message=f"Kritik dosya mevcut: {file_path} ({size_kb:.1f} KB)",
                        details={"file_path": file_path, "size_kb": size_kb}
                    ))
            else:
                missing_files.append(file_path)
                self.add_result(ValidationResult(
                    name=f"critical_file_{file_path}",
                    passed=False,
                    message=f"Kritik dosya eksik: {file_path}",
                    details={"file_path": file_path, "missing": True}
                ))
                all_files_exist = False
        
        if missing_files:
            logger.error(f"❌ {len(missing_files)} kritik dosya eksik: {', '.join(missing_files)}")
        else:
            logger.info("✅ Tüm kritik dosyalar mevcut")
        
        return all_files_exist
    
    def validate_critical_imports(self) -> bool:
        """📦 Kritik importların çalışıp çalışmadığını kontrol et"""
        
        logger.info("📦 Kritik import kontrolü...")
        
        all_imports_work = True
        failed_imports = []
        
        for module_name in self.critical_imports:
            try:
                importlib.import_module(module_name)
                
                self.add_result(ValidationResult(
                    name=f"critical_import_{module_name}",
                    passed=True,
                    message=f"Import başarılı: {module_name}",
                    details={"module": module_name}
                ))
                
            except ImportError as e:
                failed_imports.append((module_name, str(e)))
                self.add_result(ValidationResult(
                    name=f"critical_import_{module_name}",
                    passed=False,
                    message=f"Import başarısız: {module_name} - {e}",
                    details={"module": module_name, "error": str(e)}
                ))
                all_imports_work = False
                
            except Exception as e:
                self.add_result(ValidationResult(
                    name=f"critical_import_{module_name}",
                    passed=False,
                    message=f"Import hatası: {module_name} - {e}",
                    details={"module": module_name, "error": str(e), "error_type": type(e).__name__}
                ))
                all_imports_work = False
        
        if failed_imports:
            logger.error(f"❌ {len(failed_imports)} kritik import başarısız")
            for module, error in failed_imports:
                logger.error(f"   {module}: {error}")
        else:
            logger.info("✅ Tüm kritik importlar başarılı")
        
        return all_imports_work
    
    def validate_critical_classes(self) -> bool:
        """🏗️ Kritik sınıfların başlatılabilirliğini test et"""
        
        logger.info("🏗️ Kritik sınıf kontrolü...")
        
        all_classes_work = True
        failed_classes = []
        
        for module_name, class_name in self.critical_classes:
            try:
                # Modülü import et
                module = importlib.import_module(module_name)
                
                # Sınıfı al
                cls = getattr(module, class_name)
                
                # Temel initialization testi (parametresiz olabilirse)
                test_success = self._test_class_initialization(cls, class_name)
                
                if test_success:
                    self.add_result(ValidationResult(
                        name=f"critical_class_{module_name}.{class_name}",
                        passed=True,
                        message=f"Sınıf başlatılabilir: {module_name}.{class_name}",
                        details={"module": module_name, "class": class_name}
                    ))
                else:
                    self.add_result(ValidationResult(
                        name=f"critical_class_{module_name}.{class_name}",
                        passed=False,
                        message=f"Sınıf başlatılamıyor: {module_name}.{class_name}",
                        details={"module": module_name, "class": class_name}
                    ))
                    all_classes_work = False
                
            except ImportError as e:
                failed_classes.append((f"{module_name}.{class_name}", f"Import error: {e}"))
                self.add_result(ValidationResult(
                    name=f"critical_class_{module_name}.{class_name}",
                    passed=False,
                    message=f"Sınıf import hatası: {module_name}.{class_name} - {e}",
                    details={"module": module_name, "class": class_name, "error": str(e)}
                ))
                all_classes_work = False
                
            except AttributeError as e:
                failed_classes.append((f"{module_name}.{class_name}", f"Class not found: {e}"))
                self.add_result(ValidationResult(
                    name=f"critical_class_{module_name}.{class_name}",
                    passed=False,
                    message=f"Sınıf bulunamadı: {module_name}.{class_name} - {e}",
                    details={"module": module_name, "class": class_name, "error": str(e)}
                ))
                all_classes_work = False
                
            except Exception as e:
                failed_classes.append((f"{module_name}.{class_name}", f"Unexpected error: {e}"))
                self.add_result(ValidationResult(
                    name=f"critical_class_{module_name}.{class_name}",
                    passed=False,
                    message=f"Sınıf test hatası: {module_name}.{class_name} - {e}",
                    details={"module": module_name, "class": class_name, "error": str(e), "error_type": type(e).__name__}
                ))
                all_classes_work = False
        
        if failed_classes:
            logger.error(f"❌ {len(failed_classes)} kritik sınıf başarısız")
            for class_name, error in failed_classes:
                logger.error(f"   {class_name}: {error}")
        else:
            logger.info("✅ Tüm kritik sınıflar çalışıyor")
        
        return all_classes_work
    
    def _test_class_initialization(self, cls, class_name: str) -> bool:
        """🧪 Sınıf başlatılabilirlik testi"""
        
        try:
            # Portfolio sınıfı için özel test
            if class_name == "Portfolio":
                instance = cls(initial_balance=1000.0)
                return hasattr(instance, 'balance') and instance.balance == 1000.0
            
            # Strategy sınıfları için özel test
            elif "Strategy" in class_name:
                # Strategy'ler Portfolio instance'ı bekler, mock kullan
                mock_portfolio = type('MockPortfolio', (), {
                    'balance': 1000.0,
                    'initial_balance': 1000.0,
                    'positions': []
                })()
                
                instance = cls(portfolio=mock_portfolio)
                return hasattr(instance, 'portfolio')
            
            # Backtester için özel test
            elif class_name == "MomentumBacktester":
                # Geçici CSV dosyası oluştur
                with tempfile.NamedTemporaryFile(mode='w', suffix='.csv', delete=False) as f:
                    f.write("timestamp,open,high,low,close,volume\n")
                    f.write("2024-01-01 00:00:00,40000,41000,39000,40500,1000\n")
                    temp_file = f.name
                
                try:
                    instance = cls(data_file_path=temp_file, initial_capital=1000.0)
                    return hasattr(instance, 'initial_capital')
                finally:
                    os.unlink(temp_file)
            
            # Genel test - parametresiz initialization dene
            else:
                instance = cls()
                return instance is not None
                
        except Exception as e:
            logger.debug(f"Class initialization test failed for {class_name}: {e}")
            return False
    
    def validate_syntax(self) -> bool:
        """🐍 Python syntax kontrolü"""
        
        logger.info("🐍 Python syntax kontrolü...")
        
        python_files = []
        
        # Python dosyalarını bul
        for pattern in ["*.py", "*/*.py", "*/*/*.py"]:
            python_files.extend(self.project_root.glob(pattern))
        
        syntax_errors = []
        checked_files = 0
        
        for py_file in python_files:
            try:
                # __pycache__ klasörlerini atla
                if "__pycache__" in str(py_file):
                    continue
                
                with open(py_file, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # AST parse ile syntax kontrolü
                ast.parse(content, filename=str(py_file))
                checked_files += 1
                
            except SyntaxError as e:
                syntax_errors.append((str(py_file), str(e)))
                self.add_result(ValidationResult(
                    name=f"syntax_error_{py_file.name}",
                    passed=False,
                    message=f"Syntax hatası: {py_file} - {e}",
                    details={"file": str(py_file), "error": str(e), "line": getattr(e, 'lineno', None)}
                ))
                
            except Exception as e:
                # Encoding veya diğer hatalar
                logger.warning(f"⚠️ {py_file} okunamadı: {e}")
        
        if syntax_errors:
            logger.error(f"❌ {len(syntax_errors)} dosyada syntax hatası")
            for file_path, error in syntax_errors:
                logger.error(f"   {file_path}: {error}")
        else:
            logger.info(f"✅ {checked_files} Python dosyasında syntax hatası yok")
        
        self.add_result(ValidationResult(
            name="syntax_validation",
            passed=len(syntax_errors) == 0,
            message=f"Syntax kontrolü: {checked_files} dosya, {len(syntax_errors)} hata",
            details={"checked_files": checked_files, "syntax_errors": len(syntax_errors)}
        ))
        
        return len(syntax_errors) == 0
    
    def validate_directory_structure(self) -> bool:
        """📁 Klasör yapısı kontrolü"""
        
        logger.info("📁 Klasör yapısı kontrolü...")
        
        required_directories = [
            "utils",
            "strategies", 
            "optimization",
            "optimization/results",
            "scripts",
            "logs"
        ]
        
        missing_dirs = []
        
        for dir_path in required_directories:
            full_path = self.project_root / dir_path
            
            if full_path.exists() and full_path.is_dir():
                self.add_result(ValidationResult(
                    name=f"directory_{dir_path}",
                    passed=True,
                    message=f"Klasör mevcut: {dir_path}",
                    details={"directory": dir_path}
                ))
            else:
                missing_dirs.append(dir_path)
                self.add_result(ValidationResult(
                    name=f"directory_{dir_path}",
                    passed=False,
                    message=f"Klasör eksik: {dir_path}",
                    details={"directory": dir_path, "missing": True}
                ))
        
        if missing_dirs:
            logger.error(f"❌ {len(missing_dirs)} klasör eksik: {', '.join(missing_dirs)}")
            
            # Eksik klasörleri oluştur
            for dir_path in missing_dirs:
                try:
                    (self.project_root / dir_path).mkdir(parents=True, exist_ok=True)
                    logger.info(f"✅ Klasör oluşturuldu: {dir_path}")
                except Exception as e:
                    logger.error(f"❌ Klasör oluşturulamadı {dir_path}: {e}")
        else:
            logger.info("✅ Tüm gerekli klasörler mevcut")
        
        return len(missing_dirs) == 0
    
    def validate_requirements(self) -> bool:
        """📋 Requirements.txt kontrolü"""
        
        logger.info("📋 Requirements.txt kontrolü...")
        
        req_file = self.project_root / "requirements.txt"
        
        if not req_file.exists():
            self.add_result(ValidationResult(
                name="requirements_file",
                passed=False,
                message="requirements.txt dosyası bulunamadı",
                details={"missing": True}
            ))
            return False
        
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Temel paketlerin varlığını kontrol et
            required_packages = ["pandas", "numpy", "ccxt", "optuna"]
            missing_packages = []
            
            for package in required_packages:
                if package not in content:
                    missing_packages.append(package)
            
            if missing_packages:
                self.add_result(ValidationResult(
                    name="requirements_packages",
                    passed=False,
                    message=f"Requirements.txt'de eksik paketler: {', '.join(missing_packages)}",
                    details={"missing_packages": missing_packages}
                ))
                return False
            else:
                self.add_result(ValidationResult(
                    name="requirements_packages", 
                    passed=True,
                    message="Requirements.txt gerekli paketleri içeriyor",
                    details={"file_size": len(content)}
                ))
                return True
                
        except Exception as e:
            self.add_result(ValidationResult(
                name="requirements_file",
                passed=False,
                message=f"Requirements.txt okunamadı: {e}",
                details={"error": str(e)}
            ))
            return False
    
    def run_full_validation(self) -> Dict[str, Any]:
        """🔍 Tam sistem doğrulaması"""
        
        logger.info("🔍 Tam sistem doğrulaması başlatılıyor...")
        
        validation_start = datetime.now()
        
        # Tüm doğrulama testlerini çalıştır
        validation_tests = [
            ("directory_structure", self.validate_directory_structure),
            ("critical_files", self.validate_critical_files),
            ("syntax_check", self.validate_syntax),
            ("critical_imports", self.validate_critical_imports),
            ("critical_classes", self.validate_critical_classes),
            ("requirements", self.validate_requirements)
        ]
        
        passed_tests = 0
        failed_tests = 0
        
        for test_name, test_func in validation_tests:
            try:
                logger.info(f"🧪 {test_name} testi çalıştırılıyor...")
                result = test_func()
                
                if result:
                    passed_tests += 1
                    logger.info(f"✅ {test_name} başarılı")
                else:
                    failed_tests += 1
                    logger.error(f"❌ {test_name} başarısız")
                    
            except Exception as e:
                failed_tests += 1
                logger.error(f"❌ {test_name} testi hatası: {e}")
                self.add_result(ValidationResult(
                    name=f"{test_name}_exception",
                    passed=False,
                    message=f"{test_name} testi exception: {e}",
                    details={"error": str(e), "traceback": traceback.format_exc()}
                ))
        
        validation_duration = datetime.now() - validation_start
        
        # Sonuçları özetle
        total_validations = len(self.validation_results)
        successful_validations = sum(1 for r in self.validation_results if r.passed)
        
        overall_success = (
            len(self.critical_failures) == 0 and 
            failed_tests == 0 and
            successful_validations > 0
        )
        
        validation_summary = {
            "timestamp": validation_start.isoformat(),
            "duration_seconds": validation_duration.total_seconds(),
            "overall_success": overall_success,
            "total_tests": len(validation_tests),
            "passed_tests": passed_tests,
            "failed_tests": failed_tests,
            "total_validations": total_validations,
            "successful_validations": successful_validations,
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "validation_results": [
                {
                    "name": r.name,
                    "passed": r.passed,
                    "message": r.message,
                    "details": r.details,
                    "timestamp": r.timestamp.isoformat()
                }
                for r in self.validation_results
            ]
        }
        
        # Sonuçları logla
        if overall_success:
            logger.info("🎉 Sistem doğrulaması BAŞARILI!")
        else:
            logger.error("❌ Sistem doğrulaması BAŞARISIZ!")
        
        logger.info(f"📊 Test sonuçları: {passed_tests}/{len(validation_tests)} başarılı")
        logger.info(f"📋 Doğrulama sonuçları: {successful_validations}/{total_validations} başarılı")
        logger.info(f"⚠️ Kritik hatalar: {len(self.critical_failures)}")
        logger.info(f"💡 Uyarılar: {len(self.warnings)}")
        
        return validation_summary
    
    def save_validation_report(self, validation_summary: Dict[str, Any]) -> None:
        """💾 Doğrulama raporunu kaydet"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / "logs" / f"validation_report_{timestamp}.json"
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(validation_summary, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Doğrulama raporu kaydedildi: {report_file}")
    
    def install_git_hooks(self) -> bool:
        """🪝 Git hooks kurulumu"""
        
        logger.info("🪝 Git hooks kurulumu...")
        
        git_hooks_dir = self.project_root / ".git" / "hooks"
        
        if not git_hooks_dir.exists():
            logger.error("❌ Git repository bulunamadı (.git/hooks klasörü yok)")
            return False
        
        # Pre-commit hook script'i
        pre_commit_script = '''#!/bin/bash
# Phoenix System Validation Pre-commit Hook

echo "🛡️ Phoenix System Validation çalıştırılıyor..."

# Python validator'ı çalıştır
python validate_system.py --pre-commit

# Exit code'u kontrol et
if [ $? -ne 0 ]; then
    echo "❌ Sistem doğrulaması başarısız! Commit engelleniyor."
    echo "💡 Hataları düzeltip tekrar deneyin."
    exit 1
fi

echo "✅ Sistem doğrulaması başarılı! Commit devam ediyor..."
exit 0
'''
        
        # Pre-commit hook dosyasını oluştur
        pre_commit_file = git_hooks_dir / "pre-commit"
        
        try:
            with open(pre_commit_file, 'w', encoding='utf-8') as f:
                f.write(pre_commit_script)
            
            # Executable yap (Unix/Linux/macOS)
            if os.name != 'nt':  # Windows değilse
                os.chmod(pre_commit_file, 0o755)
            
            logger.info("✅ Pre-commit hook kuruldu")
            
            # Pre-push hook da kuralım
            pre_push_script = '''#!/bin/bash
# Phoenix System Validation Pre-push Hook

echo "🛡️ Phoenix System Full Validation çalıştırılıyor..."

# Full validation çalıştır
python validate_system.py --full-validation

if [ $? -ne 0 ]; then
    echo "❌ Tam sistem doğrulaması başarısız! Push engelleniyor."
    exit 1
fi

echo "✅ Tam sistem doğrulaması başarılı! Push devam ediyor..."
exit 0
'''
            
            pre_push_file = git_hooks_dir / "pre-push"
            
            with open(pre_push_file, 'w', encoding='utf-8') as f:
                f.write(pre_push_script)
            
            if os.name != 'nt':
                os.chmod(pre_push_file, 0o755)
            
            logger.info("✅ Pre-push hook kuruldu")
            return True
            
        except Exception as e:
            logger.error(f"❌ Git hooks kurulum hatası: {e}")
            return False
    
    def run_pre_commit_validation(self) -> bool:
        """🚀 Pre-commit doğrulaması (hızlı)"""
        
        logger.info("🚀 Pre-commit doğrulaması (hızlı mod)...")
        
        # Sadece kritik testleri çalıştır
        critical_tests = [
            ("syntax_check", self.validate_syntax),
            ("critical_imports", self.validate_critical_imports)
        ]
        
        all_passed = True
        
        for test_name, test_func in critical_tests:
            try:
                result = test_func()
                if not result:
                    all_passed = False
                    break
            except Exception as e:
                logger.error(f"❌ {test_name} testi hatası: {e}")
                all_passed = False
                break
        
        return all_passed
    
    def get_exit_code(self) -> int:
        """🚪 CI/CD için exit code hesapla"""
        
        if len(self.critical_failures) > 0:
            return 2  # Critical failure
        elif len(self.warnings) > 0:
            return 1  # Warnings
        else:
            return 0  # Success


def main():
    """Ana çalıştırma fonksiyonu"""
    
    parser = argparse.ArgumentParser(
        description="Phoenix System Validator - Automated System Protection",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  python validate_system.py --full-validation        # Tam doğrulama
  python validate_system.py --pre-commit            # Pre-commit kontrolü
  python validate_system.py --ci-cd                 # CI/CD pipeline kontrolü
  python validate_system.py --install-hooks         # Git hooks kurulumu
  python validate_system.py --syntax-only           # Sadece syntax kontrolü
        """
    )
    
    parser.add_argument('--full-validation', action='store_true', help='Tam sistem doğrulaması')
    parser.add_argument('--pre-commit', action='store_true', help='Pre-commit kontrolü (hızlı)')
    parser.add_argument('--ci-cd', action='store_true', help='CI/CD pipeline kontrolü')
    parser.add_argument('--install-hooks', action='store_true', help='Git hooks kurulumu')
    parser.add_argument('--syntax-only', action='store_true', help='Sadece syntax kontrolü')
    parser.add_argument('--project-root', default='.', help='Proje kök klasörü')
    parser.add_argument('--save-report', action='store_true', help='Doğrulama raporunu kaydet')
    
    args = parser.parse_args()
    
    if not any([args.full_validation, args.pre_commit, args.ci_cd, 
                args.install_hooks, args.syntax_only]):
        parser.print_help()
        return
    
    # Validator'ı başlat
    validator = PhoenixSystemValidator(project_root=args.project_root)
    
    try:
        if args.install_hooks:
            print("🪝 GIT HOOKS KURULUMU")
            print("="*50)
            
            success = validator.install_git_hooks()
            
            if success:
                print("✅ Git hooks başarıyla kuruldu!")
                print("💡 Artık her commit öncesi otomatik doğrulama yapılacak")
            else:
                print("❌ Git hooks kurulumu başarısız!")
                sys.exit(1)
        
        elif args.syntax_only:
            print("🐍 SYNTAX KONTROLÜ")
            print("="*50)
            
            success = validator.validate_syntax()
            
            if success:
                print("✅ Syntax kontrolü başarılı!")
            else:
                print("❌ Syntax hataları bulundu!")
                sys.exit(1)
        
        elif args.pre_commit:
            print("🚀 PRE-COMMIT DOĞRULAMA")
            print("="*30)
            
            success = validator.run_pre_commit_validation()
            
            exit_code = validator.get_exit_code()
            
            if success:
                print("✅ Pre-commit doğrulama başarılı!")
            else:
                print("❌ Pre-commit doğrulama başarısız!")
            
            sys.exit(exit_code)
        
        elif args.full_validation or args.ci_cd:
            mode_name = "FULL VALIDATION" if args.full_validation else "CI/CD VALIDATION"
            print(f"🛡️ {mode_name}")
            print("="*80)
            
            validation_summary = validator.run_full_validation()
            
            # Sonuçları göster
            print(f"\n📊 DOĞRULAMA SONUÇLARI:")
            print(f"   ⏱️ Süre: {validation_summary['duration_seconds']:.2f} saniye")
            print(f"   🧪 Testler: {validation_summary['passed_tests']}/{validation_summary['total_tests']} başarılı")
            print(f"   📋 Doğrulamalar: {validation_summary['successful_validations']}/{validation_summary['total_validations']} başarılı")
            print(f"   ❌ Kritik hatalar: {validation_summary['critical_failures']}")
            print(f"   ⚠️ Uyarılar: {validation_summary['warnings']}")
            
            # Başarısız testleri göster
            if validation_summary['critical_failures'] > 0:
                print(f"\n❌ KRİTİK HATALAR:")
                for failure in validator.critical_failures:
                    print(f"   • {failure.message}")
            
            if validation_summary['warnings'] > 0:
                print(f"\n⚠️ UYARILAR:")
                for warning in validator.warnings[:5]:  # İlk 5'ini göster
                    print(f"   • {warning.message}")
            
            # Raporu kaydet
            if args.save_report or args.ci_cd:
                validator.save_validation_report(validation_summary)
            
            # Exit code
            exit_code = validator.get_exit_code()
            
            if validation_summary['overall_success']:
                print("\n🎉 SİSTEM DOĞRULAMA BAŞARILI!")
            else:
                print("\n❌ SİSTEM DOĞRULAMA BAŞARISIZ!")
            
            sys.exit(exit_code)
    
    except KeyboardInterrupt:
        print("\n🛑 Doğrulama kullanıcı tarafından durduruldu")
        sys.exit(130)
    
    except Exception as e:
        logger.error(f"❌ Sistem doğrulama hatası: {e}")
        logger.error(traceback.format_exc())
        print(f"\n❌ HATA: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()