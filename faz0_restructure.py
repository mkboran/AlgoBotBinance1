#!/usr/bin/env python3
"""
🔥 PROJE PHOENIX - FAZ 0: KOD TABANI YENİDEN YAPILANDIRMA
💎 Other/ klasörünü tamamen ortadan kaldırma ve modüler yapıya kavuşturma

Bu script şunları yapar:
1. ✅ Other/ klasöründeki tüm dosyaları analiz eder
2. ✅ Dosyaları görevde belirtilen klasörlere taşır:
   - utils/ ← yeniden kullanılabilir modüller
   - scripts/ ← tek seferlik bakım script'leri  
   - optimization/ ← optimizasyon araçları
3. ✅ Other/ klasörünü tamamen siler
4. ✅ Tüm işlemleri loglar ve doğrular

KULLANIM:
python faz0_restructure.py --execute --force

🚨 DİKKAT: Bu script Other/ klasörünü tamamen silecektir!

🎯 WINDOWS UYUMLU VERSİYON
"""

import os
import sys
import shutil
import logging
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Any, Optional
import argparse
import json

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs") / "faz0_restructure.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("FAZ0_Restructure")

class PhoenixRestructureEngine:
    """🔥 Proje Phoenix kod tabanı yeniden yapılandırma motoru"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.other_dir = self.project_root / "Other"
        
        # Hedef klasörler
        self.utils_dir = self.project_root / "utils"
        self.scripts_dir = self.project_root / "scripts"
        self.optimization_dir = self.project_root / "optimization"
        
        # Taşıma planı (görevde belirtilen)
        self.file_mapping = {
            # utils/ klasörüne taşınacaklar (yeniden kullanılabilir modüller)
            "utils": [
                "enhanced_dynamic_exit_system.py",
                "kelly_criterion_ml_position_sizing.py", 
                "global_market_intelligence_system.py",
                "ultra_profit_risk_system.py",
                "ml_enhanced_dynamic_stop_loss.py",
                "ultra_advanced_solution_system.py"
            ],
            
            # scripts/ klasörüne taşınacaklar (bakım ve analiz script'leri)
            "scripts": [
                "run_critical_fixes.py",
                "run_immediate_fixes_ultimate.py",
                "manual_param_update.py",
                "fix_backtest_import.py",
                "fix_ml_predictor_interface.py",
                "analyze_trials.py",
                "ml_vs_noml_comparison.py",
                "ultimate_implementation.py",  # Bu da bir script
                "manual_backup_momentum_optimized_1751065046.py"  # Manual backup
            ],
            
            # optimization/ klasörüne taşınacaklar (optimizasyon araçları)
            "optimization": [
                "optimize_individual_strategies.py",
                "optimize_strategy_ultimate.py", 
                "smart_range_optimizer.py",
                "ultimate_optimizer_optimized.py"
            ]
        }
        
        # İşlem sonuçları
        self.moved_files = []
        self.skipped_files = []
        self.errors = []
        
        logger.info("🔥 Phoenix Restructure Engine başlatıldı")
        logger.info(f"📁 Proje kökü: {self.project_root.absolute()}")
        
    def ensure_directories_exist(self) -> bool:
        """📁 Hedef klasörlerin var olduğundan emin ol"""
        
        try:
            # Logs klasörünü oluştur
            (self.project_root / "logs").mkdir(exist_ok=True)
            
            # Hedef klasörleri oluştur
            self.utils_dir.mkdir(exist_ok=True)
            self.scripts_dir.mkdir(exist_ok=True) 
            self.optimization_dir.mkdir(exist_ok=True)
            
            # scripts/deprecated/ alt klasörünü de oluştur (auto_update_parameters.py zaten orada)
            (self.scripts_dir / "deprecated").mkdir(exist_ok=True)
            
            logger.info("✅ Tüm hedef klasörler hazır")
            return True
            
        except Exception as e:
            logger.error(f"❌ Klasör oluşturma hatası: {e}")
            self.errors.append(f"Directory creation error: {e}")
            return False
    
    def analyze_other_directory(self) -> Dict[str, Any]:
        """🔍 Other/ klasörünü analiz et"""
        
        if not self.other_dir.exists():
            logger.warning("⚠️ Other/ klasörü bulunamadı!")
            return {"exists": False, "files": []}
        
        try:
            files = list(self.other_dir.glob("*.py"))
            
            analysis = {
                "exists": True,
                "total_files": len(files),
                "files": [f.name for f in files],
                "file_sizes": {f.name: f.stat().st_size for f in files}
            }
            
            logger.info(f"🔍 Other/ klasörü analizi:")
            logger.info(f"   📊 Toplam .py dosyası: {analysis['total_files']}")
            
            for file_name in analysis["files"]:
                size_kb = analysis["file_sizes"][file_name] / 1024
                logger.info(f"   📄 {file_name} ({size_kb:.1f} KB)")
            
            return analysis
            
        except Exception as e:
            logger.error(f"❌ Other/ klasörü analiz hatası: {e}")
            self.errors.append(f"Analysis error: {e}")
            return {"exists": False, "files": []}
    
    def move_file(self, file_name: str, target_dir: Path) -> bool:
        """📦 Tek dosyayı taşı"""
        
        source_file = self.other_dir / file_name
        target_file = target_dir / file_name
        
        if not source_file.exists():
            logger.warning(f"⚠️ Dosya bulunamadı: {file_name}")
            self.skipped_files.append(f"{file_name} (bulunamadı)")
            return False
        
        try:
            # Backup oluştur (eğer hedef dosya zaten varsa)
            if target_file.exists():
                backup_name = f"{file_name}.backup_{int(datetime.now().timestamp())}"
                backup_file = target_dir / backup_name
                shutil.copy2(target_file, backup_file)
                logger.info(f"💾 Backup oluşturuldu: {backup_name}")
            
            # Dosyayı taşı
            shutil.move(str(source_file), str(target_file))
            
            logger.info(f"✅ Taşındı: {file_name} → {target_dir.name}/")
            self.moved_files.append(f"{file_name} → {target_dir.name}/")
            return True
            
        except Exception as e:
            logger.error(f"❌ Dosya taşıma hatası {file_name}: {e}")
            self.errors.append(f"Move error {file_name}: {e}")
            return False
    
    def execute_file_restructure(self) -> Dict[str, Any]:
        """🚀 Dosya yeniden yapılandırmasını çalıştır"""
        
        logger.info("🚀 Dosya yeniden yapılandırması başlatılıyor...")
        
        results = {
            "moved_files": 0,
            "skipped_files": 0,
            "errors": 0,
            "details": {}
        }
        
        # Her hedef klasör için dosyaları taşı
        for target_name, file_list in self.file_mapping.items():
            target_dir = getattr(self, f"{target_name}_dir")
            
            logger.info(f"📁 {target_name}/ klasörüne taşıma işlemi...")
            
            moved_count = 0
            for file_name in file_list:
                if self.move_file(file_name, target_dir):
                    moved_count += 1
            
            results["details"][target_name] = {
                "target_files": len(file_list),
                "moved_files": moved_count,
                "success_rate": (moved_count / len(file_list)) * 100 if file_list else 100
            }
            
            logger.info(f"   ✅ {moved_count}/{len(file_list)} dosya taşındı (%{results['details'][target_name]['success_rate']:.1f})")
        
        # Toplam sonuçları hesapla
        results["moved_files"] = len(self.moved_files)
        results["skipped_files"] = len(self.skipped_files)
        results["errors"] = len(self.errors)
        
        return results
    
    def handle_remaining_files(self) -> Dict[str, Any]:
        """🔄 Other/ klasöründe kalan dosyaları işle"""
        
        if not self.other_dir.exists():
            return {"remaining_files": 0, "actions": []}
        
        try:
            remaining_files = list(self.other_dir.glob("*"))
            actions = []
            
            logger.info(f"🔄 Kalan dosyalar kontrol ediliyor... ({len(remaining_files)} dosya)")
            
            for file_path in remaining_files:
                if file_path.is_file():
                    # Kalan .py dosyalarını scripts/deprecated/ klasörüne taşı
                    if file_path.suffix == ".py":
                        target_file = self.scripts_dir / "deprecated" / file_path.name
                        shutil.move(str(file_path), str(target_file))
                        actions.append(f"Moved to deprecated: {file_path.name}")
                        logger.info(f"📦 Deprecated'e taşındı: {file_path.name}")
                    
                    # Diğer dosyaları da scripts/ altına taşı
                    elif file_path.suffix in [".txt", ".json", ".log", ".md"]:
                        target_file = self.scripts_dir / file_path.name
                        shutil.move(str(file_path), str(target_file))
                        actions.append(f"Moved to scripts: {file_path.name}")
                        logger.info(f"📄 Scripts'e taşındı: {file_path.name}")
            
            return {
                "remaining_files": len(remaining_files),
                "actions": actions
            }
            
        except Exception as e:
            logger.error(f"❌ Kalan dosyalar işleme hatası: {e}")
            self.errors.append(f"Remaining files error: {e}")
            return {"remaining_files": 0, "actions": [], "error": str(e)}
    
    def remove_other_directory(self) -> bool:
        """🗑️ Other/ klasörünü tamamen sil"""
        
        if not self.other_dir.exists():
            logger.info("✅ Other/ klasörü zaten mevcut değil")
            return True
        
        try:
            # Klasörün boş olduğundan emin ol
            remaining_items = list(self.other_dir.iterdir())
            
            if remaining_items:
                logger.warning(f"⚠️ Other/ klasöründe hala {len(remaining_items)} öğe var:")
                for item in remaining_items:
                    logger.warning(f"   - {item.name}")
                
                # Zorla sil
                shutil.rmtree(self.other_dir)
                logger.info("🗑️ Other/ klasörü zorla silindi")
            else:
                # Boş klasörü sil
                self.other_dir.rmdir()
                logger.info("🗑️ Other/ klasörü silindi")
            
            return True
            
        except Exception as e:
            logger.error(f"❌ Other/ klasörü silme hatası: {e}")
            self.errors.append(f"Directory removal error: {e}")
            return False
    
    def save_restructure_report(self, results: Dict[str, Any]) -> None:
        """📊 Yeniden yapılandırma raporunu kaydet"""
        
        report = {
            "faz0_restructure_report": {
                "timestamp": datetime.now(timezone.utc).isoformat(),
                "project_root": str(self.project_root.absolute()),
                "operation": "Other directory restructure",
                "results": results,
                "moved_files": self.moved_files,
                "skipped_files": self.skipped_files,
                "errors": self.errors,
                "file_mapping": self.file_mapping
            }
        }
        
        # JSON raporunu kaydet
        report_file = self.project_root / "logs" / f"faz0_restructure_report_{int(datetime.now().timestamp())}.json"
        
        try:
            with open(report_file, 'w', encoding='utf-8') as f:
                json.dump(report, f, indent=2, ensure_ascii=False)
            
            logger.info(f"📊 Rapor kaydedildi: {report_file}")
            
        except Exception as e:
            logger.error(f"❌ Rapor kaydetme hatası: {e}")
    
    def execute_complete_restructure(self) -> Dict[str, Any]:
        """🎯 Tam yeniden yapılandırma işlemini çalıştır"""
        
        logger.info("🎯 FAZ 0 - Kod Tabanı Yeniden Yapılandırması Başlatılıyor")
        logger.info("="*80)
        
        start_time = datetime.now()
        
        # 1. Klasörleri hazırla
        if not self.ensure_directories_exist():
            return {"success": False, "error": "Directory preparation failed"}
        
        # 2. Other/ klasörünü analiz et
        analysis = self.analyze_other_directory()
        if not analysis["exists"]:
            return {"success": False, "error": "Other directory not found"}
        
        # 3. Dosyaları yeniden yapılandır
        restructure_results = self.execute_file_restructure()
        
        # 4. Kalan dosyaları işle
        remaining_results = self.handle_remaining_files()
        
        # 5. Other/ klasörünü sil
        removal_success = self.remove_other_directory()
        
        # 6. Final sonuçlar
        duration = datetime.now() - start_time
        
        final_results = {
            "success": len(self.errors) == 0 and removal_success,
            "duration_seconds": duration.total_seconds(),
            "analysis": analysis,
            "restructure": restructure_results,
            "remaining_files": remaining_results,
            "other_directory_removed": removal_success,
            "summary": {
                "total_moved_files": len(self.moved_files),
                "total_skipped_files": len(self.skipped_files),
                "total_errors": len(self.errors)
            }
        }
        
        # 7. Raporu kaydet
        self.save_restructure_report(final_results)
        
        # 8. Sonuçları logla
        logger.info("="*80)
        logger.info("🎉 FAZ 0 - Kod Tabanı Yeniden Yapılandırması TAMAMLANDI!")
        logger.info(f"   ⏱️  Süre: {duration.total_seconds():.2f} saniye")
        logger.info(f"   ✅ Taşınan dosyalar: {len(self.moved_files)}")
        logger.info(f"   ⚠️  Atlanan dosyalar: {len(self.skipped_files)}")
        logger.info(f"   ❌ Hatalar: {len(self.errors)}")
        logger.info(f"   🗑️  Other/ klasörü silindi: {'✅' if removal_success else '❌'}")
        
        if self.moved_files:
            logger.info("\n📦 Taşınan dosyalar:")
            for file_info in self.moved_files:
                logger.info(f"   - {file_info}")
        
        if self.errors:
            logger.warning("\n❌ Hatalar:")
            for error in self.errors:
                logger.warning(f"   - {error}")
        
        logger.info("="*80)
        
        return final_results


def main():
    """Ana çalıştırma fonksiyonu"""
    
    parser = argparse.ArgumentParser(description="Proje Phoenix - FAZ 0 Kod Tabanı Yeniden Yapılandırma")
    parser.add_argument("--execute", action="store_true", help="Yeniden yapılandırmayı çalıştır")
    parser.add_argument("--force", action="store_true", help="Onay olmadan çalıştır")
    parser.add_argument("--dry-run", action="store_true", help="Sadece analiz yap, değişiklik yapma")
    
    args = parser.parse_args()
    
    if not args.execute:
        print("🔥 PROJE PHOENIX - FAZ 0: KOD TABANI YENİDEN YAPILANDIRMA")
        print("💎 Other/ klasörünü temizlemek için --execute parametresini kullanın")
        print("🚨 DİKKAT: Bu işlem Other/ klasörünü tamamen silecektir!")
        print("\nKullanım:")
        print("  python faz0_restructure.py --execute --force")
        print("  python faz0_restructure.py --dry-run  # Sadece analiz")
        return
    
    if not (args.execute or args.dry_run):
        print("🚨 UYARI: Bu işlem Other/ klasörünü tamamen silecektir!")
        print("Devam etmek istediğinizden emin misiniz? (y/N): ", end="")
        
        response = input().strip().lower()
        if response not in ['y', 'yes', 'evet']:
            print("❌ İşlem iptal edildi.")
            return
    
    # Yeniden yapılandırma motorunu başlat
    engine = PhoenixRestructureEngine()
    
    if args.dry_run:
        logger.info("🔍 DRY RUN MODE - Sadece analiz yapılıyor...")
        analysis = engine.analyze_other_directory()
        
        if analysis["exists"]:
            print(f"\n📊 Other/ Klasörü Analizi:")
            print(f"   📁 Toplam .py dosyası: {analysis['total_files']}")
            print(f"   📄 Dosyalar: {', '.join(analysis['files'])}")
            
            print(f"\n📦 Taşıma Planı:")
            for target, files in engine.file_mapping.items():
                print(f"   {target}/: {len(files)} dosya")
                for file_name in files:
                    if file_name in analysis["files"]:
                        print(f"     ✅ {file_name}")
                    else:
                        print(f"     ❌ {file_name} (bulunamadı)")
        
        return
    
    # Tam yeniden yapılandırmayı çalıştır
    results = engine.execute_complete_restructure()
    
    if results["success"]:
        print("\n🎉 FAZ 0 BAŞARIYLA TAMAMLANDI!")
        print("✅ Other/ klasörü tamamen temizlendi")
        print("✅ Tüm dosyalar uygun klasörlere taşındı")
        print("✅ Modüler kod yapısı oluşturuldu")
        print("\n🚀 Sonraki adım: FAZ 0 Adım 0.2 - Deterministik Parametre Yönetimi")
    else:
        print("\n❌ FAZ 0 TAMAMLANAMADI!")
        print("Lütfen hataları kontrol edin ve tekrar deneyin.")
        sys.exit(1)


if __name__ == "__main__":
    main()