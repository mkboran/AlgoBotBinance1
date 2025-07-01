#!/usr/bin/env python3
"""
🔍 PROJE PHOENIX - FAZ 0: BAĞIMLILIK ANALİZİ VE REQUIREMENTS GENERATOR
💎 Statik kod analizi ile eksiksiz bağımlılık grafiği oluşturma

Bu sistem şunları yapar:
1. ✅ Tüm .py dosyalarını statik olarak analiz eder
2. ✅ Import ifadelerinden bağımlılık grafiği çıkarır  
3. ✅ Gerçekten kullanılan kütüphaneleri belirler
4. ✅ Sıfırdan eksiksiz requirements.txt oluşturur
5. ✅ Döngüsel bağımlılıkları tespit eder
6. ✅ Kullanılmayan import'ları bulur

KULLANIM:
python dependency_analyzer.py --analyze-all --generate-requirements
python dependency_analyzer.py --check-circular --optimize-imports
"""

import ast
import os
import sys
import logging
import importlib
import subprocess
from pathlib import Path
from datetime import datetime, timezone
from typing import Dict, List, Set, Any, Optional, Tuple
import argparse
import json
import re
import traceback
from collections import defaultdict, deque
try:
    import networkx as nx
    NETWORKX_AVAILABLE = True
except ImportError:
    NETWORKX_AVAILABLE = False

# Logging için logs klasörünü oluştur
Path("logs").mkdir(exist_ok=True)

# Logging yapılandırması
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout),
        logging.FileHandler(Path("logs") / "dependency_analysis.log", mode='w', encoding='utf-8')
    ]
)
logger = logging.getLogger("DependencyAnalyzer")

class ImportAnalyzer(ast.NodeVisitor):
    """AST tabanlı import analizi sınıfı"""
    
    def __init__(self):
        self.imports = set()
        self.from_imports = set()
        self.local_imports = set()
        self.import_details = []
    
    def visit_Import(self, node):
        for alias in node.names:
            module_name = alias.name
            self.imports.add(module_name)
            self.import_details.append({
                "type": "import",
                "module": module_name,
                "alias": alias.asname,
                "line": node.lineno
            })
        self.generic_visit(node)
    
    def visit_ImportFrom(self, node):
        if node.module:
            module_name = node.module
            self.from_imports.add(module_name)
            
            # Local vs external imports
            if module_name.startswith('.') or module_name in ['utils', 'strategies', 'optimization']:
                self.local_imports.add(module_name)
            
            for alias in node.names:
                self.import_details.append({
                    "type": "from_import",
                    "module": module_name,
                    "name": alias.name,
                    "alias": alias.asname,
                    "line": node.lineno
                })
        self.generic_visit(node)

class DependencyAnalyzer:
    """💎 Proje Phoenix Bağımlılık Analizi Motoru"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        
        # Logs klasörünü oluştur (yoksa otomatik oluştur)
        logs_dir = self.project_root / "logs"
        logs_dir.mkdir(exist_ok=True)
        
        # Log dosyasını oluştur (yoksa)
        log_file = logs_dir / "dependency_analysis.log"
        if not log_file.exists():
            log_file.touch()
        
        # Analiz sonuçları
        self.file_imports = {}
        self.all_external_imports = set()
        self.all_local_imports = set()
        self.dependency_graph = nx.DiGraph() if NETWORKX_AVAILABLE else {}
        
        # Standart kütüphane modülleri (bu modülleri requirements'a eklemeyelim)
        self.stdlib_modules = {
            'os', 'sys', 'datetime', 'time', 'json', 'csv', 'math', 'random',
            'collections', 'itertools', 'functools', 'operator', 'pathlib',
            'logging', 'argparse', 'subprocess', 'threading', 'multiprocessing',
            'asyncio', 'typing', 'dataclasses', 'enum', 're', 'ast', 'inspect',
            'warnings', 'traceback', 'tempfile', 'shutil', 'glob', 'pickle',
            'sqlite3', 'urllib', 'http', 'email', 'base64', 'hashlib',
            'uuid', 'decimal', 'fractions', 'statistics', 'copy', 'gc'
        }
        
        logger.info("🔍 Dependency Analyzer başlatıldı")
        logger.info(f"📁 Proje kökü: {self.project_root.absolute()}")
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """📄 Tek dosyanın import analizini yap"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                source = f.read()
            
            tree = ast.parse(source)
            analyzer = ImportAnalyzer()
            analyzer.visit(tree)
            
            # External imports (non-stdlib, non-local)
            external_imports = set()
            for imp in analyzer.imports.union(analyzer.from_imports):
                root_module = imp.split('.')[0]
                if (root_module not in self.stdlib_modules and 
                    not imp.startswith('.') and 
                    root_module not in ['utils', 'strategies', 'optimization', 'backtesting', 'scripts']):
                    external_imports.add(root_module)
            
            analysis = {
                "file_path": str(file_path.relative_to(self.project_root)),
                "imports": list(analyzer.imports),
                "from_imports": list(analyzer.from_imports),
                "local_imports": list(analyzer.local_imports),
                "external_imports": list(external_imports),
                "import_details": analyzer.import_details,
                "total_imports": len(analyzer.imports) + len(analyzer.from_imports),
                "lines_of_code": len(source.splitlines())
            }
            
            # Dependency graph'a ekle
            if NETWORKX_AVAILABLE:
                file_node = str(file_path.relative_to(self.project_root))
                for imp in analyzer.imports.union(analyzer.from_imports):
                    self.dependency_graph.add_edge(file_node, imp)
            
            # Global setlere ekle
            self.all_external_imports.update(external_imports)
            
            return analysis
            
        except SyntaxError as e:
            logger.error(f"❌ Syntax error in {file_path}: {e}")
            return {"error": f"Syntax error: {e}", "file_path": str(file_path)}
        except Exception as e:
            logger.error(f"❌ Error analyzing {file_path}: {e}")
            return {"error": str(e), "file_path": str(file_path)}
    
    def analyze_all_files(self) -> Dict[str, Any]:
        """📂 Tüm Python dosyalarını analiz et"""
        
        logger.info("📂 Tüm Python dosyaları analiz ediliyor...")
        
        # Python dosyalarını bul
        python_files = []
        for pattern in ["*.py", "**/*.py"]:
            python_files.extend(self.project_root.glob(pattern))
        
        # __pycache__ ve .git klasörlerini filtrele
        python_files = [
            f for f in python_files 
            if "__pycache__" not in str(f) and ".git" not in str(f)
        ]
        
        logger.info(f"📊 {len(python_files)} Python dosyası bulundu")
        
        # Her dosyayı analiz et
        analysis_results = {}
        successful_analyses = 0
        
        for py_file in python_files:
            file_analysis = self.analyze_file(py_file)
            file_key = str(py_file.relative_to(self.project_root))
            analysis_results[file_key] = file_analysis
            
            if "error" not in file_analysis:
                successful_analyses += 1
                self.file_imports[file_key] = file_analysis
        
        logger.info(f"✅ {successful_analyses}/{len(python_files)} dosya başarıyla analiz edildi")
        
        return {
            "total_files": len(python_files),
            "successful_analyses": successful_analyses,
            "failed_analyses": len(python_files) - successful_analyses,
            "all_external_imports": sorted(list(self.all_external_imports)),
            "total_external_imports": len(self.all_external_imports),
            "file_analyses": analysis_results
        }
    
    def check_circular_dependencies(self) -> Dict[str, Any]:
        """🔄 Döngüsel bağımlılık kontrolü"""
        
        logger.info("🔄 Döngüsel bağımlılık kontrolü...")
        
        if not NETWORKX_AVAILABLE:
            logger.warning("⚠️ NetworkX bulunamadı, döngüsel bağımlılık analizi atlanıyor")
            return {"error": "NetworkX not available"}
        
        cycles = []
        
        try:
            # Find cycles in dependency graph
            cycle_generator = nx.simple_cycles(self.dependency_graph)
            for cycle in cycle_generator:
                if len(cycle) > 1:  # Self-loops'ları dahil etme
                    cycles.append(cycle)
        
        except Exception as e:
            logger.error(f"❌ Döngüsel bağımlılık kontrolü hatası: {e}")
            return {"error": str(e)}
        
        if cycles:
            logger.warning(f"⚠️ {len(cycles)} döngüsel bağımlılık tespit edildi")
            for i, cycle in enumerate(cycles):
                logger.warning(f"  Döngü {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
        else:
            logger.info("✅ Döngüsel bağımlılık bulunamadı")
        
        return {
            "has_cycles": len(cycles) > 0,
            "cycle_count": len(cycles),
            "cycles": cycles
        }
    
    def detect_unused_imports(self) -> Dict[str, Any]:
        """🗑️ Kullanılmayan import'ları tespit et"""
        
        logger.info("🗑️ Kullanılmayan importlar tespit ediliyor...")
        
        unused_imports = {}
        
        for file_path, analysis in self.file_imports.items():
            file_unused = []
            
            try:
                # Dosyayı tekrar oku
                full_path = self.project_root / file_path
                with open(full_path, 'r', encoding='utf-8') as f:
                    content = f.read()
                
                # Import details'leri kontrol et
                for import_detail in analysis.get("import_details", []):
                    if import_detail["type"] == "import":
                        module_name = import_detail["module"]
                        alias = import_detail.get("alias", module_name)
                        
                        # Kullanım kontrolü (basit)
                        if alias not in content.split('\n')[import_detail["line"]:]:
                            # Import satırından sonra kullanılmıyor
                            import_usage_count = content.count(alias)
                            if import_usage_count <= 1:  # Sadece import satırında geçiyor
                                file_unused.append(import_detail)
            
            except Exception as e:
                logger.debug(f"Unused import detection error for {file_path}: {e}")
            
            if file_unused:
                unused_imports[file_path] = file_unused
        
        total_unused = sum(len(imports) for imports in unused_imports.values())
        
        if total_unused > 0:
            logger.warning(f"⚠️ {total_unused} kullanılmayan import tespit edildi")
        else:
            logger.info("✅ Kullanılmayan import bulunamadı")
        
        return {
            "total_unused": total_unused,
            "files_with_unused": len(unused_imports),
            "unused_imports": unused_imports
        }
    
    def _get_package_version(self, package_name: str) -> Optional[str]:
        """📦 Paket versiyonunu al"""
        
        try:
            # pip show komutu ile versiyon bilgisi al
            result = subprocess.run(
                [sys.executable, "-m", "pip", "show", package_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        return line.split(':')[1].strip()
            
            return None
            
        except Exception as e:
            logger.debug(f"Version detection error for {package_name}: {e}")
            return None
    
    def generate_requirements_txt(self, include_versions: bool = True) -> str:
        """📋 Requirements.txt içeriği oluştur"""
        
        logger.info("📋 Requirements.txt oluşturuluyor...")
        
        # Kategorilere ayır
        categorized_packages = self._categorize_packages()
        
        requirements_lines = [
            "# ========================================================================================",
            "# 🚀 PROJE PHOENIX - AUTO-GENERATED REQUIREMENTS",
            "# 💎 Production-Ready Dependencies for Algorithmic Trading Platform",
            "# ========================================================================================",
            ""
        ]
        
        for category, packages in categorized_packages.items():
            if packages:
                requirements_lines.append(f"# {category}")
                requirements_lines.append("# " + "="*60)
                
                for package in sorted(packages):
                    if include_versions:
                        version = self._get_package_version(package)
                        if version:
                            requirements_lines.append(f"{package}>={version}")
                        else:
                            requirements_lines.append(f"{package}  # Version not detected")
                    else:
                        requirements_lines.append(package)
                
                requirements_lines.append("")
        
        return '\n'.join(requirements_lines)
    
    def _categorize_packages(self) -> Dict[str, List[str]]:
        """📦 Paketleri kategorilere ayır"""
        
        categories = {
            "Core Data Science": [],
            "Machine Learning": [],
            "Trading & Finance": [],
            "Async & Networking": [],
            "Optimization": [],
            "Visualization": [],
            "Testing & Development": [],
            "System Utilities": [],
            "Other": []
        }
        
        # Kategori mapping'i
        category_mapping = {
            "Core Data Science": ["pandas", "numpy", "scipy", "scikit-learn", "sklearn"],
            "Machine Learning": ["torch", "pytorch", "tensorflow", "keras", "xgboost", "lightgbm", "catboost"],
            "Trading & Finance": ["ccxt", "pandas_ta", "ta", "backtrader", "zipline", "quantlib"],
            "Async & Networking": ["aiohttp", "asyncio-mqtt", "websockets", "requests"],
            "Optimization": ["optuna", "hyperopt", "skopt", "scikit-optimize"],
            "Visualization": ["matplotlib", "seaborn", "plotly", "bokeh"],
            "Testing & Development": ["pytest", "unittest", "mock", "coverage", "black", "flake8"],
            "System Utilities": ["psutil", "python-dateutil", "pytz", "pydantic", "python-dotenv"]
        }
        
        # Paketleri kategorilere ata
        categorized = set()
        
        for category, keywords in category_mapping.items():
            for package in self.all_external_imports:
                if any(keyword in package.lower() for keyword in keywords):
                    categories[category].append(package)
                    categorized.add(package)
        
        # Kategorize edilmeyenleri "Other"a ekle
        for package in self.all_external_imports:
            if package not in categorized:
                categories["Other"].append(package)
        
        # Boş kategorileri kaldır
        return {k: v for k, v in categories.items() if v}
    
    def optimize_imports(self) -> Dict[str, Any]:
        """🔧 Import optimizasyon önerileri"""
        
        logger.info("🔧 Import optimizasyon analizi...")
        
        optimizations = {
            "has_optimizations": False,
            "unused_imports": [],
            "duplicate_imports": [],
            "suggestions": []
        }
        
        # Her dosya için optimizasyon önerileri
        for file_path, file_analysis in self.file_imports.items():
            file_optimizations = self._analyze_file_import_optimizations(file_analysis)
            
            if file_optimizations["has_optimizations"]:
                optimizations["has_optimizations"] = True
                optimizations["suggestions"].append({
                    "file": file_path,
                    "optimizations": file_optimizations
                })
        
        return optimizations
    
    def _analyze_file_import_optimizations(self, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """📄 Dosya bazında import optimizasyon analizi"""
        
        optimizations = {
            "has_optimizations": False,
            "unused_imports": [],
            "duplicate_imports": [],
            "suggestions": []
        }
        
        # Import details'leri analiz et
        import_details = file_analysis.get("import_details", [])
        
        # Duplicate import detection (basit)
        seen_imports = set()
        for imp in import_details:
            import_key = f"{imp['type']}:{imp['module']}"
            if import_key in seen_imports:
                optimizations["duplicate_imports"].append(imp)
                optimizations["has_optimizations"] = True
            seen_imports.add(import_key)
        
        return optimizations
    
    def save_analysis_results(self, analysis_results: Dict[str, Any]) -> None:
        """💾 Analiz sonuçlarını kaydet"""
        
        timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
        
        # JSON raporu
        json_file = self.project_root / "logs" / f"dependency_analysis_{timestamp}.json"
        
        with open(json_file, 'w', encoding='utf-8') as f:
            json.dump(analysis_results, f, indent=2, ensure_ascii=False)
        
        logger.info(f"💾 Analiz sonuçları kaydedildi: {json_file}")
        
        # Requirements.txt
        requirements_content = self.generate_requirements_txt()
        requirements_file = self.project_root / "requirements.txt"
        
        # Backup oluştur
        if requirements_file.exists():
            backup_file = self.project_root / f"requirements.txt.backup_{timestamp}"
            requirements_file.rename(backup_file)
            logger.info(f"💾 Eski requirements.txt backup: {backup_file}")
        
        with open(requirements_file, 'w', encoding='utf-8') as f:
            f.write(requirements_content)
        
        logger.info(f"✅ Yeni requirements.txt oluşturuldu: {requirements_file}")
        
        # Dependency graph (GraphML format)
        if NETWORKX_AVAILABLE and self.dependency_graph.nodes():
            graph_file = self.project_root / "logs" / f"dependency_graph_{timestamp}.graphml"
            nx.write_graphml(self.dependency_graph, graph_file)
            logger.info(f"📈 Dependency graph kaydedildi: {graph_file}")
    
    def run_comprehensive_analysis(self) -> Dict[str, Any]:
        """🔬 Kapsamlı bağımlılık analizi"""
        
        logger.info("🔬 KAPSAMLI BAĞIMLILIK ANALİZİ BAŞLIYOR...")
        
        analysis_start = datetime.now(timezone.utc)
        
        # Ana analiz
        main_analysis = self.analyze_all_files()
        
        # Döngüsel bağımlılık kontrolü
        circular_analysis = self.check_circular_dependencies()
        
        # Kullanılmayan import'lar
        unused_analysis = self.detect_unused_imports()
        
        # Import optimizasyonları
        optimization_analysis = self.optimize_imports()
        
        analysis_end = datetime.now(timezone.utc)
        duration = (analysis_end - analysis_start).total_seconds()
        
        # Comprehensive sonuç
        comprehensive_results = {
            "timestamp": analysis_end.isoformat(),
            "duration_seconds": duration,
            "project_root": str(self.project_root.absolute()),
            "analysis_summary": {
                "total_files_analyzed": main_analysis["successful_analyses"],
                "total_external_imports": len(self.all_external_imports),
                "has_circular_dependencies": circular_analysis.get("has_cycles", False),
                "total_unused_imports": unused_analysis.get("total_unused", 0),
                "optimization_opportunities": optimization_analysis.get("has_optimizations", False)
            },
            "main_analysis": main_analysis,
            "circular_dependencies": circular_analysis,
            "unused_imports": unused_analysis,
            "import_optimizations": optimization_analysis
        }
        
        # Sonuçları kaydet
        self.save_analysis_results(comprehensive_results)
        
        # Konsol özeti
        logger.info("="*80)
        logger.info("🔬 BAĞIMLILIK ANALİZİ RAPORU")
        logger.info("="*80)
        logger.info(f"📊 Analiz edilen dosya: {main_analysis['successful_analyses']}")
        logger.info(f"📦 External import: {len(self.all_external_imports)}")
        logger.info(f"🔄 Döngüsel bağımlılık: {'VAR' if circular_analysis.get('has_cycles', False) else 'YOK'}")
        logger.info(f"🗑️ Kullanılmayan import: {unused_analysis.get('total_unused', 0)}")
        logger.info(f"🔧 Optimizasyon fırsatı: {'VAR' if optimization_analysis.get('has_optimizations', False) else 'YOK'}")
        logger.info(f"⏱️ Analiz süresi: {duration:.2f} saniye")
        logger.info("="*80)
        
        return comprehensive_results


def main():
    """Ana çalıştırma fonksiyonu"""
    
    parser = argparse.ArgumentParser(
        description="Phoenix Dependency Analyzer - Comprehensive Project Analysis",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  python dependency_analyzer.py --analyze-all                    # Tam analiz
  python dependency_analyzer.py --generate-requirements          # Requirements.txt oluştur
  python dependency_analyzer.py --check-circular                 # Döngüsel bağımlılık kontrolü
  python dependency_analyzer.py --optimize-imports               # Import optimizasyonu
  python dependency_analyzer.py --full-analysis                  # Kapsamlı analiz
        """
    )
    
    parser.add_argument('--analyze-all', action='store_true', help='Tüm projeyi analiz et')
    parser.add_argument('--generate-requirements', action='store_true', help='Requirements.txt oluştur')
    parser.add_argument('--check-circular', action='store_true', help='Döngüsel bağımlılık kontrolü')
    parser.add_argument('--optimize-imports', action='store_true', help='Import optimizasyonu')
    parser.add_argument('--full-analysis', action='store_true', help='Kapsamlı analiz (hepsi)')
    parser.add_argument('--include-versions', action='store_true', default=True, help='Requirements.txt\'ye versiyon bilgisi ekle')
    parser.add_argument('--project-root', default='.', help='Proje kök dizini')
    
    args = parser.parse_args()
    
    # Analyzer oluştur
    analyzer = DependencyAnalyzer(project_root=args.project_root)
    
    try:
        if args.full_analysis or (not any([args.analyze_all, args.generate_requirements, 
                                          args.check_circular, args.optimize_imports])):
            # Varsayılan: kapsamlı analiz
            results = analyzer.run_comprehensive_analysis()
            
        else:
            # Belirli analizler
            if args.analyze_all:
                results = analyzer.analyze_all_files()
                logger.info(f"✅ {results['successful_analyses']} dosya analiz edildi")
            
            if args.check_circular:
                circular_results = analyzer.check_circular_dependencies()
                if circular_results.get("has_cycles"):
                    logger.warning(f"⚠️ {circular_results['cycle_count']} döngüsel bağımlılık bulundu")
                else:
                    logger.info("✅ Döngüsel bağımlılık yok")
            
            if args.optimize_imports:
                opt_results = analyzer.optimize_imports()
                if opt_results.get("has_optimizations"):
                    logger.info("🔧 Import optimizasyon fırsatları bulundu")
                else:
                    logger.info("✅ Import'lar optimize")
            
            if args.generate_requirements:
                # İlk önce analiz yap
                analyzer.analyze_all_files()
                # Sonra requirements oluştur
                req_content = analyzer.generate_requirements_txt(args.include_versions)
                req_file = analyzer.project_root / "requirements.txt"
                with open(req_file, 'w') as f:
                    f.write(req_content)
                logger.info(f"✅ Requirements.txt oluşturuldu: {req_file}")
        
        logger.info("🎉 Bağımlılık analizi tamamlandı!")
        
    except KeyboardInterrupt:
        logger.info("🛑 Analiz kullanıcı tarafından durduruldu")
        sys.exit(130)
    except Exception as e:
        logger.error(f"❌ Beklenmedik hata: {e}")
        traceback.print_exc()
        sys.exit(1)


if __name__ == "__main__":
    main()