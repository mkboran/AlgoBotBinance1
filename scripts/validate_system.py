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

# 🔧 CRITICAL: Python path'e proje kökünü ekle
PROJECT_ROOT = Path(__file__).parent.parent.absolute()
if str(PROJECT_ROOT) not in sys.path:
    sys.path.insert(0, str(PROJECT_ROOT))

# Logs için logs klasörünü oluştur
Path("logs").mkdir(exist_ok=True)

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
        
        # Logs klasörünü oluştur (yoksa otomatik oluştur)
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Log dosyasını oluştur (yoksa)
        log_file = logs_dir / "system_validation.log"
        if not log_file.exists():
            log_file.touch()
        
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
        """📁 Kritik dosyaların varlığı kontrolü"""
        
        logger.info("📁 Kritik dosyalar kontrolü...")
        
        missing_files = []
        
        for file_path in self.critical_files:
            full_path = self.project_root / file_path
            
            if full_path.exists() and full_path.is_file():
                self.add_result(ValidationResult(
                    name=f"critical_file_{file_path}",
                    passed=True,
                    message=f"Kritik dosya mevcut: {file_path}",
                    details={"file_path": file_path, "size": full_path.stat().st_size}
                ))
            else:
                missing_files.append(file_path)
                self.add_result(ValidationResult(
                    name=f"critical_file_{file_path}",
                    passed=False,
                    message=f"Kritik dosya eksik: {file_path}",
                    details={"file_path": file_path, "missing": True}
                ))
        
        if missing_files:
            logger.error(f"❌ {len(missing_files)} kritik dosya eksik: {', '.join(missing_files)}")
        else:
            logger.info("✅ Tüm kritik dosyalar mevcut")
        
        return len(missing_files) == 0

    def validate_directory_structure(self) -> bool:
        """📁 Klasör yapısı kontrolü"""
        
        logger.info("📁 Klasör yapısı kontrolü...")
        
        required_directories = [
            "utils",
            "strategies",
            "optimization", 
            "optimization/results",
            "scripts",
            "logs",
            "backtesting"
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
            logger.error("❌ requirements.txt dosyası bulunamadı")
            return False
        
        try:
            with open(req_file, 'r', encoding='utf-8') as f:
                requirements_content = f.read()
            
            # Temel gereksinimleri kontrol et
            required_packages = ['pandas', 'numpy', 'ccxt', 'optuna', 'scikit-learn']
            missing_packages = []
            
            for package in required_packages:
                if package not in requirements_content:
                    missing_packages.append(package)
            
            if missing_packages:
                self.add_result(ValidationResult(
                    name="requirements_packages",
                    passed=False,
                    message=f"Eksik paketler: {', '.join(missing_packages)}",
                    details={"missing_packages": missing_packages}
                ))
                logger.error(f"❌ Requirements.txt'de eksik paketler: {', '.join(missing_packages)}")
                return False
            else:
                self.add_result(ValidationResult(
                    name="requirements_packages",
                    passed=True,
                    message="Tüm temel paketler mevcut",
                    details={"packages_found": required_packages}
                ))
                logger.info("✅ Requirements.txt doğrulandı")
                return True
                
        except Exception as e:
            self.add_result(ValidationResult(
                name="requirements_file",
                passed=False,
                message=f"Requirements.txt okuma hatası: {e}",
                details={"error": str(e)}
            ))
            logger.error(f"❌ Requirements.txt okuma hatası: {e}")
            return False

    def validate_imports(self) -> bool:
        """📦 Kritik importların test edilmesi"""
        
        logger.info("📦 Kritik importlar kontrolü...")
        
        # Önce test_imports.py'yi çalıştırarak import'ları test et
        try:
            import subprocess
            result = subprocess.run([
                sys.executable, "test_imports.py"
            ], capture_output=True, text=True, cwd=self.project_root, encoding='utf-8')
            
            if result.returncode == 0 and "8/8 import basarili" in result.stdout:
                logger.info("✅ Import test (subprocess): Tüm import'lar başarılı")
                
                # Manuel import testini de yap
                return self._manual_import_test()
            else:
                logger.error(f"❌ Import test (subprocess) başarısız: {result.stderr}")
                return False
                
        except Exception as e:
            logger.warning(f"⚠️ Subprocess import test başarısız, manuel test yapılıyor: {e}")
            return self._manual_import_test()
    
    def _manual_import_test(self) -> bool:
        """🔧 Manuel import test (fallback)"""
        
        failed_imports = []
        
        # Python path'i ekle
        original_path = sys.path.copy()
        try:
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
            
            for import_name in self.critical_imports:
                try:
                    # Modül zaten yüklüyse reload et
                    if import_name in sys.modules:
                        importlib.reload(sys.modules[import_name])
                    else:
                        importlib.import_module(import_name)
                    
                    self.add_result(ValidationResult(
                        name=f"import_{import_name}",
                        passed=True,
                        message=f"Import başarılı: {import_name}",
                        details={"import_name": import_name}
                    ))
                    logger.debug(f"✅ Import başarılı: {import_name}")
                    
                except ImportError as e:
                    failed_imports.append(import_name)
                    self.add_result(ValidationResult(
                        name=f"import_{import_name}",
                        passed=False,
                        message=f"Import başarısız: {import_name} - {e}",
                        details={"import_name": import_name, "error": str(e)}
                    ))
                    logger.error(f"❌ Import başarısız: {import_name} - {e}")
                    
                except Exception as e:
                    failed_imports.append(import_name)
                    self.add_result(ValidationResult(
                        name=f"import_{import_name}",
                        passed=False,
                        message=f"Import beklenmedik hata: {import_name} - {e}",
                        details={"import_name": import_name, "error": str(e)}
                    ))
                    logger.error(f"❌ Import beklenmedik hata: {import_name} - {e}")
        
        finally:
            # Python path'i geri yükle
            sys.path = original_path
        
        if failed_imports:
            logger.error(f"❌ {len(failed_imports)} import başarısız: {', '.join(failed_imports)}")
        else:
            logger.info("✅ Tüm kritik importlar başarılı")
        
        return len(failed_imports) == 0

    def validate_class_instantiation(self) -> bool:
        """🏗️ Kritik sınıfların test edilmesi"""
        
        logger.info("🏗️ Kritik sınıflar kontrolü...")
        
        failed_classes = []
        
        # Python path'i ekle
        original_path = sys.path.copy()
        try:
            if str(self.project_root) not in sys.path:
                sys.path.insert(0, str(self.project_root))
        
            for module_name, class_name in self.critical_classes:
                try:
                    module = importlib.import_module(module_name)
                    cls = getattr(module, class_name)
                    
                    # Basit instantiation testi
                    if class_name == "Portfolio":
                        instance = cls(initial_capital_usdt=1000.0)
                    elif class_name == "EnhancedMomentumStrategy":
                        # Gerçek Portfolio instance oluştur
                        portfolio_module = importlib.import_module('utils.portfolio')
                        portfolio_cls = getattr(portfolio_module, 'Portfolio')
                        portfolio = portfolio_cls(initial_capital_usdt=1000.0)
                        instance = cls(portfolio=portfolio)
                    elif class_name == "MomentumBacktester":
                        # MomentumBacktester required parameters ile test et
                        instance = cls(
                            csv_path="test.csv",
                            initial_capital=1000.0,
                            start_date="2024-01-01",
                            end_date="2024-12-31", 
                            symbol="BTC/USDT"
                        )
                    else:
                        # Diğer sınıflar için varsayılan constructor
                        instance = cls()
                    
                    self.add_result(ValidationResult(
                        name=f"class_{module_name}_{class_name}",
                        passed=True,
                        message=f"Sınıf başarıyla test edildi: {module_name}.{class_name}",
                        details={"module": module_name, "class": class_name}
                    ))
                    logger.debug(f"✅ Sınıf testi başarılı: {module_name}.{class_name}")
                    
                except Exception as e:
                    failed_classes.append(f"{module_name}.{class_name}")
                    self.add_result(ValidationResult(
                        name=f"class_{module_name}_{class_name}",
                        passed=False,
                        message=f"Sınıf testi başarısız: {module_name}.{class_name} - {e}",
                        details={"module": module_name, "class": class_name, "error": str(e)}
                    ))
                    logger.error(f"❌ Sınıf testi başarısız: {module_name}.{class_name} - {e}")
        
        finally:
            # Python path'i geri yükle
            sys.path = original_path
        
        if failed_classes:
            logger.error(f"❌ {len(failed_classes)} sınıf testi başarısız: {', '.join(failed_classes)}")
        else:
            logger.info("✅ Tüm kritik sınıflar başarıyla test edildi")
        
        return len(failed_classes) == 0

    def validate_code_quality(self) -> bool:
        """🔍 Kod kalitesi kontrolü"""
        
        logger.info("🔍 Kod kalitesi kontrolü...")
        
        python_files = list(self.project_root.rglob("*.py"))
        syntax_errors = []
        
        for py_file in python_files:
            # __pycache__ ve .git klasörlerini atla
            if "__pycache__" in str(py_file) or ".git" in str(py_file):
                continue
                
            try:
                with open(py_file, 'r', encoding='utf-8') as f:
                    source = f.read()
                
                # Syntax kontrolü
                ast.parse(source)
                
                self.add_result(ValidationResult(
                    name=f"syntax_{py_file.name}",
                    passed=True,
                    message=f"Syntax doğru: {py_file.relative_to(self.project_root)}",
                    details={"file": str(py_file.relative_to(self.project_root))}
                ))
                
            except SyntaxError as e:
                syntax_errors.append(str(py_file.relative_to(self.project_root)))
                self.add_result(ValidationResult(
                    name=f"syntax_{py_file.name}",
                    passed=False,
                    message=f"Syntax hatası: {py_file.relative_to(self.project_root)} - Line {e.lineno}: {e.msg}",
                    details={"file": str(py_file.relative_to(self.project_root)), "error": str(e)}
                ))
                logger.error(f"❌ Syntax hatası: {py_file.relative_to(self.project_root)} - Line {e.lineno}: {e.msg}")
                
            except Exception as e:
                logger.debug(f"Dosya okunamadı: {py_file} - {e}")
        
        if syntax_errors:
            logger.error(f"❌ {len(syntax_errors)} dosyada syntax hatası")
        else:
            logger.info(f"✅ {len(python_files)} Python dosyası syntax kontrolünden geçti")
        
        return len(syntax_errors) == 0

    def run_full_validation(self) -> Dict[str, Any]:
        """🔬 Tam sistem doğrulaması"""
        
        logger.info("🔬 TAM SİSTEM DOĞRULAMASI BAŞLIYOR...")
        
        validation_start = datetime.now(timezone.utc)
        
        # Doğrulama adımları
        validations = [
            ("Directory Structure", self.validate_directory_structure),
            ("Critical Files", self.validate_critical_files),
            ("Requirements", self.validate_requirements),
            ("Imports", self.validate_imports),
            ("Class Instantiation", self.validate_class_instantiation),
            ("Code Quality", self.validate_code_quality)
        ]
        
        passed_validations = 0
        
        for validation_name, validation_func in validations:
            logger.info(f"🔍 {validation_name} doğrulaması...")
            try:
                result = validation_func()
                if result:
                    passed_validations += 1
                    logger.info(f"✅ {validation_name} PASSED")
                else:
                    logger.error(f"❌ {validation_name} FAILED")
            except Exception as e:
                logger.error(f"❌ {validation_name} doğrulama hatası: {e}")
        
        validation_end = datetime.now(timezone.utc)
        duration = (validation_end - validation_start).total_seconds()
        
        # Sonuç raporu
        total_validations = len(validations)
        success_rate = (passed_validations / total_validations) * 100
        
        validation_summary = {
            "timestamp": validation_end.isoformat(),
            "duration_seconds": duration,
            "total_validations": total_validations,
            "passed_validations": passed_validations,
            "failed_validations": total_validations - passed_validations,
            "success_rate_percent": success_rate,
            "critical_failures": len(self.critical_failures),
            "warnings": len(self.warnings),
            "overall_status": "PASSED" if success_rate >= 80 else "FAILED"
        }
        
        # Raporu kaydet
        self.save_validation_report(validation_summary)
        
        # Konsol çıktısı
        logger.info("="*80)
        logger.info("🔬 SİSTEM DOĞRULAMA RAPORU")
        logger.info("="*80)
        logger.info(f"📊 Toplam Doğrulama: {total_validations}")
        logger.info(f"✅ Başarılı: {passed_validations}")
        logger.info(f"❌ Başarısız: {total_validations - passed_validations}")
        logger.info(f"📈 Başarı Oranı: {success_rate:.1f}%")
        logger.info(f"🚨 Kritik Hatalar: {len(self.critical_failures)}")
        logger.info(f"⚠️ Uyarılar: {len(self.warnings)}")
        logger.info(f"⏱️ Süre: {duration:.2f} saniye")
        logger.info(f"🏆 GENEL DURUM: {validation_summary['overall_status']}")
        logger.info("="*80)
        
        return validation_summary

    def save_validation_report(self, summary: Dict[str, Any]) -> None:
        """💾 Doğrulama raporunu kaydet"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        report_file = self.project_root / "logs" / f"validation_report_{timestamp}.json"
        
        detailed_report = {
            "summary": summary,
            "validation_results": [
                {
                    "name": result.name,
                    "passed": result.passed,
                    "message": result.message,
                    "details": result.details,
                    "timestamp": result.timestamp.isoformat()
                }
                for result in self.validation_results
            ]
        }
        
        with open(report_file, 'w', encoding='utf-8') as f:
            json.dump(detailed_report, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Doğrulama raporu kaydedildi: {report_file}")

    def install_git_hooks(self) -> bool:
        """🪝 Git hooks kurulumu"""
        
        logger.info("🪝 Git hooks kurulumu...")
        
        git_hooks_dir = self.project_root / ".git" / "hooks"
        
        if not git_hooks_dir.exists():
            logger.error("❌ Git repository bulunamadı")
            return False
        
        # Pre-commit hook içeriği
        pre_commit_content = f"""#!/bin/bash
# Phoenix System Validation Pre-commit Hook
echo "🛡️ Running Phoenix system validation..."
python "{self.project_root}/scripts/validate_system.py" --pre-commit
exit $?
"""
        
        pre_commit_file = git_hooks_dir / "pre-commit"
        
        try:
            with open(pre_commit_file, 'w') as f:
                f.write(pre_commit_content)
            
            # Executable yap
            pre_commit_file.chmod(0o755)
            
            logger.info("✅ Pre-commit hook kuruldu")
            return True
            
        except Exception as e:
            logger.error(f"❌ Pre-commit hook kurulum hatası: {e}")
            return False


def main():
    """Ana çalıştırma fonksiyonu"""
    
    parser = argparse.ArgumentParser(
        description="Phoenix System Validator - Comprehensive System Health Check",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  python validate_system.py --full-validation        # Tam doğrulama
  python validate_system.py --pre-commit            # Pre-commit kontrolü  
  python validate_system.py --ci-cd                 # CI/CD pipeline kontrolü
  python validate_system.py --install-hooks         # Git hooks kurulumu
        """
    )
    
    parser.add_argument('--full-validation', action='store_true', help='Tam sistem doğrulaması')
    parser.add_argument('--pre-commit', action='store_true', help='Pre-commit doğrulaması')
    parser.add_argument('--ci-cd', action='store_true', help='CI/CD pipeline doğrulaması')
    parser.add_argument('--install-hooks', action='store_true', help='Git hooks kurulumu')
    parser.add_argument('--project-root', default='.', help='Proje kök dizini')
    
    args = parser.parse_args()
    
    # Validator oluştur
    validator = PhoenixSystemValidator(project_root=args.project_root)
    
    try:
        if args.install_hooks:
            success = validator.install_git_hooks()
            sys.exit(0 if success else 1)
        
        elif args.pre_commit:
            # Pre-commit için hızlı kontroller
            logger.info("🪝 Pre-commit doğrulaması...")
            critical_passed = validator.validate_critical_files()
            syntax_passed = validator.validate_code_quality()
            
            success = critical_passed and syntax_passed
            logger.info(f"🪝 Pre-commit doğrulama: {'PASSED' if success else 'FAILED'}")
            sys.exit(0 if success else 1)
        
        elif args.ci_cd:
            # CI/CD için kapsamlı kontroller
            summary = validator.run_full_validation()
            success = summary['overall_status'] == 'PASSED'
            sys.exit(0 if success else 1)
        
        else:
            # Varsayılan: tam doğrulama
            summary = validator.run_full_validation()
            success = summary['overall_status'] == 'PASSED'
            sys.exit(0 if success else 1)
            
    except KeyboardInterrupt:
        logger.info("🛑 Doğrulama kullanıcı tarafından durduruldu")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Beklenmedik hata: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()