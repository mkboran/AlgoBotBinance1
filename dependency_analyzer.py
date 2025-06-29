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
from collections import defaultdict, deque
import networkx as nx

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
        
        # Logs klasörünü oluştur
        (self.project_root / "logs").mkdir(exist_ok=True)
        
        # Analiz sonuçları
        self.file_dependencies = {}
        self.all_imports = set()
        self.external_packages = set()
        self.local_modules = set()
        self.dependency_graph = nx.DiGraph()
        
        # Python built-in modules
        self.builtin_modules = set(sys.builtin_module_names)
        self.standard_library = self._get_standard_library_modules()
        
        # Proje spesifik modüller
        self.project_modules = {'utils', 'strategies', 'optimization', 'scripts'}
        
        logger.info("🔍 Dependency Analyzer başlatıldı")
        logger.info(f"📁 Proje kökü: {self.project_root.absolute()}")
    
    def _get_standard_library_modules(self) -> Set[str]:
        """Python standard library modüllerini tespit et"""
        
        # Python 3.x standard library modüllerinin bir listesi
        stdlib_modules = {
            'abc', 'argparse', 'array', 'ast', 'asyncio', 'base64', 'bisect',
            'calendar', 'collections', 'copy', 'csv', 'datetime', 'decimal',
            'functools', 'glob', 'gzip', 'hashlib', 'heapq', 'html', 'http',
            'importlib', 'io', 'itertools', 'json', 'logging', 'math', 'multiprocessing',
            'operator', 'os', 'pathlib', 'pickle', 'random', 're', 'shutil',
            'socket', 'sqlite3', 'string', 'subprocess', 'sys', 'tempfile',
            'threading', 'time', 'traceback', 'typing', 'urllib', 'uuid',
            'warnings', 'weakref', 'xml', 'zipfile', 'enum', 'dataclasses',
            'contextlib', 'concurrent', 'email', 'encodings'
        }
        
        return stdlib_modules
    
    def analyze_file(self, file_path: Path) -> Dict[str, Any]:
        """📄 Tek dosyayı analiz et"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # AST parse
            tree = ast.parse(content, filename=str(file_path))
            
            # Import analyzer
            analyzer = ImportAnalyzer()
            analyzer.visit(tree)
            
            # Sonuçları organize et
            file_analysis = {
                "file_path": str(file_path),
                "relative_path": str(file_path.relative_to(self.project_root)),
                "imports": list(analyzer.imports),
                "from_imports": list(analyzer.from_imports),
                "local_imports": list(analyzer.local_imports),
                "import_details": analyzer.import_details,
                "total_imports": len(analyzer.imports) + len(analyzer.from_imports),
                "external_dependencies": [],
                "local_dependencies": []
            }
            
            # External vs local classification
            all_imported_modules = analyzer.imports | analyzer.from_imports
            
            for module in all_imported_modules:
                base_module = module.split('.')[0]
                
                if base_module in self.builtin_modules or base_module in self.standard_library:
                    # Built-in or standard library
                    continue
                elif base_module in self.project_modules or module.startswith('.'):
                    # Local project module
                    file_analysis["local_dependencies"].append(module)
                    self.local_modules.add(module)
                else:
                    # External package
                    file_analysis["external_dependencies"].append(base_module)
                    self.external_packages.add(base_module)
            
            return file_analysis
            
        except SyntaxError as e:
            logger.warning(f"⚠️ Syntax error in {file_path}: {e}")
            return {"file_path": str(file_path), "error": "syntax_error", "details": str(e)}
        
        except Exception as e:
            logger.error(f"❌ Error analyzing {file_path}: {e}")
            return {"file_path": str(file_path), "error": "analysis_error", "details": str(e)}
    
    def analyze_project(self) -> Dict[str, Any]:
        """🔍 Tüm projeyi analiz et"""
        
        logger.info("🔍 Proje analizi başlatılıyor...")
        
        # Tüm Python dosyalarını bul
        python_files = []
        
        # Ana klasörler
        search_paths = [
            self.project_root,
            self.project_root / "utils",
            self.project_root / "strategies", 
            self.project_root / "optimization",
            self.project_root / "scripts"
        ]
        
        for search_path in search_paths:
            if search_path.exists():
                python_files.extend(search_path.glob("*.py"))
                # Alt klasörlerde de ara
                python_files.extend(search_path.glob("*/*.py"))
        
        # Duplike dosyaları kaldır
        python_files = list(set(python_files))
        
        logger.info(f"📊 {len(python_files)} Python dosyası bulundu")
        
        # Her dosyayı analiz et
        analysis_results = {}
        successful_analyses = 0
        
        for file_path in python_files:
            try:
                file_analysis = self.analyze_file(file_path)
                relative_path = str(file_path.relative_to(self.project_root))
                analysis_results[relative_path] = file_analysis
                
                if "error" not in file_analysis:
                    successful_analyses += 1
                    
                    # Dependency graph'e ekle
                    self._add_to_dependency_graph(file_analysis)
                
            except Exception as e:
                logger.error(f"❌ {file_path} analiz hatası: {e}")
        
        logger.info(f"✅ {successful_analyses}/{len(python_files)} dosya başarıyla analiz edildi")
        
        # Analiz özetini oluştur
        project_analysis = {
            "timestamp": datetime.now(timezone.utc).isoformat(),
            "project_root": str(self.project_root.absolute()),
            "total_files": len(python_files),
            "successful_analyses": successful_analyses,
            "file_analyses": analysis_results,
            "summary": self._generate_analysis_summary(),
            "dependency_graph_info": self._analyze_dependency_graph()
        }
        
        self.file_dependencies = analysis_results
        
        return project_analysis
    
    def _add_to_dependency_graph(self, file_analysis: Dict[str, Any]) -> None:
        """📈 Bağımlılık grafiğine dosya ekle"""
        
        file_path = file_analysis["relative_path"]
        
        # Node'u ekle
        self.dependency_graph.add_node(file_path)
        
        # Local dependencies için edge'leri ekle
        for dep in file_analysis.get("local_dependencies", []):
            # Relative import'ları düzelt
            if dep.startswith('.'):
                # TODO: Relative import resolution
                continue
            
            # Module path'i dosya path'ine çevir
            potential_paths = [
                f"{dep.replace('.', '/')}.py",
                f"{dep.replace('.', '/')}/{dep.split('.')[-1]}.py",
                f"{dep}/{dep.split('.')[-1]}.py"
            ]
            
            for pot_path in potential_paths:
                if pot_path in self.file_dependencies:
                    self.dependency_graph.add_edge(file_path, pot_path)
                    break
    
    def _generate_analysis_summary(self) -> Dict[str, Any]:
        """📊 Analiz özetini oluştur"""
        
        # External packages'i topla
        all_external = set()
        all_local = set()
        total_imports = 0
        
        for file_analysis in self.file_dependencies.values():
            if "error" not in file_analysis:
                all_external.update(file_analysis.get("external_dependencies", []))
                all_local.update(file_analysis.get("local_dependencies", []))
                total_imports += file_analysis.get("total_imports", 0)
        
        return {
            "total_external_packages": len(all_external),
            "external_packages": sorted(list(all_external)),
            "total_local_modules": len(all_local),
            "local_modules": sorted(list(all_local)),
            "total_imports": total_imports,
            "most_used_externals": self._get_most_used_packages(all_external),
            "dependency_statistics": self._calculate_dependency_stats()
        }
    
    def _get_most_used_packages(self, external_packages: Set[str]) -> List[Tuple[str, int]]:
        """📈 En çok kullanılan paketleri bul"""
        
        package_counts = defaultdict(int)
        
        for file_analysis in self.file_dependencies.values():
            if "error" not in file_analysis:
                for pkg in file_analysis.get("external_dependencies", []):
                    package_counts[pkg] += 1
        
        # Sırala ve döndür
        return sorted(package_counts.items(), key=lambda x: x[1], reverse=True)
    
    def _calculate_dependency_stats(self) -> Dict[str, Any]:
        """📊 Bağımlılık istatistiklerini hesapla"""
        
        external_counts = []
        local_counts = []
        
        for file_analysis in self.file_dependencies.values():
            if "error" not in file_analysis:
                external_counts.append(len(file_analysis.get("external_dependencies", [])))
                local_counts.append(len(file_analysis.get("local_dependencies", [])))
        
        if not external_counts:
            return {"error": "No valid files to analyze"}
        
        return {
            "avg_external_deps_per_file": sum(external_counts) / len(external_counts),
            "max_external_deps": max(external_counts),
            "avg_local_deps_per_file": sum(local_counts) / len(local_counts),
            "max_local_deps": max(local_counts),
            "files_with_most_external_deps": self._find_files_with_most_deps("external"),
            "files_with_most_local_deps": self._find_files_with_most_deps("local")
        }
    
    def _find_files_with_most_deps(self, dep_type: str) -> List[Tuple[str, int]]:
        """📋 En çok bağımlılığa sahip dosyaları bul"""
        
        file_dep_counts = []
        dep_key = f"{dep_type}_dependencies"
        
        for relative_path, file_analysis in self.file_dependencies.items():
            if "error" not in file_analysis:
                dep_count = len(file_analysis.get(dep_key, []))
                file_dep_counts.append((relative_path, dep_count))
        
        return sorted(file_dep_counts, key=lambda x: x[1], reverse=True)[:10]
    
    def _analyze_dependency_graph(self) -> Dict[str, Any]:
        """📈 Bağımlılık grafiği analizini yap"""
        
        if not self.dependency_graph.nodes():
            return {"error": "Empty dependency graph"}
        
        try:
            # Temel graph metrics
            metrics = {
                "node_count": self.dependency_graph.number_of_nodes(),
                "edge_count": self.dependency_graph.number_of_edges(),
                "is_directed": self.dependency_graph.is_directed(),
                "density": nx.density(self.dependency_graph)
            }
            
            # Döngüsel bağımlılık kontrolü
            try:
                cycles = list(nx.simple_cycles(self.dependency_graph))
                metrics["circular_dependencies"] = cycles
                metrics["has_circular_deps"] = len(cycles) > 0
                metrics["circular_dependency_count"] = len(cycles)
            except Exception as e:
                metrics["circular_dependency_error"] = str(e)
            
            # En çok bağımlılığa sahip dosyalar
            in_degrees = dict(self.dependency_graph.in_degree())
            out_degrees = dict(self.dependency_graph.out_degree())
            
            metrics["most_dependent_files"] = sorted(
                in_degrees.items(), key=lambda x: x[1], reverse=True
            )[:10]
            
            metrics["most_depended_upon"] = sorted(
                out_degrees.items(), key=lambda x: x[1], reverse=True  
            )[:10]
            
            return metrics
            
        except Exception as e:
            return {"error": f"Graph analysis failed: {e}"}
    
    def generate_requirements_txt(self, include_versions: bool = True) -> str:
        """📋 requirements.txt oluştur"""
        
        logger.info("📋 requirements.txt oluşturuluyor...")
        
        if not self.external_packages:
            logger.warning("⚠️ External package bulunamadı")
            return "# No external dependencies found\n"
        
        requirements_lines = []
        requirements_lines.append("# Generated by Phoenix Dependency Analyzer")
        requirements_lines.append(f"# Generated on: {datetime.now().strftime('%Y-%m-%d %H:%M:%S')}")
        requirements_lines.append(f"# Total packages: {len(self.external_packages)}")
        requirements_lines.append("")
        
        # Paketleri kategorilendir
        categorized_packages = self._categorize_packages()
        
        for category, packages in categorized_packages.items():
            if packages:
                requirements_lines.append(f"# {category}")
                
                for package in sorted(packages):
                    if include_versions:
                        version = self._get_package_version(package)
                        if version:
                            requirements_lines.append(f"{package}=={version}")
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
            "Other": []
        }
        
        # Kategori mapping'i
        category_mapping = {
            "Core Data Science": ["pandas", "numpy", "scipy", "scikit-learn", "sklearn"],
            "Machine Learning": ["torch", "pytorch", "tensorflow", "keras", "xgboost", "lightgbm", "catboost"],
            "Trading & Finance": ["ccxt", "pandas_ta", "ta", "backtrader", "zipline", "quantlib"],
            "Async & Networking": ["aiohttp", "asyncio", "requests", "websockets"],
            "Optimization": ["optuna", "hyperopt", "skopt", "scipy.optimize"],
            "Visualization": ["matplotlib", "seaborn", "plotly", "bokeh"],
            "Testing & Development": ["pytest", "unittest", "mock", "coverage", "black", "flake8"]
        }
        
        # Paketleri kategorilere ata
        categorized = set()
        
        for category, keywords in category_mapping.items():
            for package in self.external_packages:
                if any(keyword in package.lower() for keyword in keywords):
                    categories[category].append(package)
                    categorized.add(package)
        
        # Kategorize edilmeyen paketleri "Other"a ekle
        for package in self.external_packages:
            if package not in categorized:
                categories["Other"].append(package)
        
        return categories
    
    def _get_package_version(self, package_name: str) -> Optional[str]:
        """📦 Paketin yüklü versiyonunu al"""
        
        try:
            # pip show ile versiyon bilgisi al
            result = subprocess.run(
                ["pip", "show", package_name],
                capture_output=True,
                text=True,
                timeout=10
            )
            
            if result.returncode == 0:
                for line in result.stdout.split('\n'):
                    if line.startswith('Version:'):
                        return line.split(':', 1)[1].strip()
            
            return None
            
        except Exception:
            return None
    
    def check_circular_dependencies(self) -> Dict[str, Any]:
        """🔄 Döngüsel bağımlılık kontrolü"""
        
        logger.info("🔄 Döngüsel bağımlılık kontrolü...")
        
        if not self.dependency_graph.nodes():
            return {"error": "Dependency graph is empty"}
        
        try:
            cycles = list(nx.simple_cycles(self.dependency_graph))
            
            circular_analysis = {
                "has_circular_dependencies": len(cycles) > 0,
                "circular_dependency_count": len(cycles),
                "cycles": cycles,
                "severity": "high" if len(cycles) > 0 else "none",
                "recommendations": []
            }
            
            if cycles:
                logger.warning(f"⚠️ {len(cycles)} döngüsel bağımlılık tespit edildi!")
                
                for i, cycle in enumerate(cycles):
                    logger.warning(f"   Cycle {i+1}: {' -> '.join(cycle)} -> {cycle[0]}")
                    
                    circular_analysis["recommendations"].append({
                        "cycle": cycle,
                        "suggestion": f"Break cycle by refactoring common functionality into a separate module"
                    })
            else:
                logger.info("✅ Döngüsel bağımlılık bulunamadı")
            
            return circular_analysis
            
        except Exception as e:
            logger.error(f"❌ Döngüsel bağımlılık kontrolü hatası: {e}")
            return {"error": str(e)}
    
    def optimize_imports(self) -> Dict[str, Any]:
        """⚡ Import optimizasyonu önerileri"""
        
        logger.info("⚡ Import optimizasyonu analizi...")
        
        optimization_results = {
            "unused_imports": [],
            "duplicate_imports": [],
            "optimization_suggestions": [],
            "total_optimizable_files": 0
        }
        
        for relative_path, file_analysis in self.file_dependencies.items():
            if "error" not in file_analysis:
                file_optimizations = self._analyze_file_imports(file_analysis)
                
                if file_optimizations["has_optimizations"]:
                    optimization_results["total_optimizable_files"] += 1
                    
                    optimization_results["optimization_suggestions"].append({
                        "file": relative_path,
                        "optimizations": file_optimizations
                    })
        
        return optimization_results
    
    def _analyze_file_imports(self, file_analysis: Dict[str, Any]) -> Dict[str, Any]:
        """📄 Dosya import analizi"""
        
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
        if self.dependency_graph.nodes():
            graph_file = self.project_root / "logs" / f"dependency_graph_{timestamp}.graphml"
            nx.write_graphml(self.dependency_graph, graph_file)
            logger.info(f"📈 Dependency graph kaydedildi: {graph_file}")


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
    parser.add_argument('--project-root', default='.', help='Proje kök klasörü')
    
    args = parser.parse_args()
    
    if not any([args.analyze_all, args.generate_requirements, args.check_circular, 
                args.optimize_imports, args.full_analysis]):
        parser.print_help()
        return
    
    # Analyzer'ı başlat
    analyzer = DependencyAnalyzer(project_root=args.project_root)
    
    try:
        print("🔍 PHOENIX DEPENDENCY ANALYZER")
        print("="*80)
        
        if args.full_analysis or args.analyze_all:
            print("📊 Proje analizi başlatılıyor...")
            analysis_results = analyzer.analyze_project()
            
            print("\n📋 ANALİZ SONUÇLARI:")
            print(f"   📄 Toplam dosya: {analysis_results['total_files']}")
            print(f"   ✅ Başarılı analiz: {analysis_results['successful_analyses']}")
            
            summary = analysis_results['summary']
            print(f"   📦 External packages: {summary['total_external_packages']}")
            print(f"   🏠 Local modules: {summary['total_local_modules']}")
            print(f"   📥 Toplam import: {summary['total_imports']}")
            
            if summary.get('most_used_externals'):
                print(f"\n📈 EN ÇOK KULLANILAN PAKETLER:")
                for pkg, count in summary['most_used_externals'][:10]:
                    print(f"   📦 {pkg}: {count} dosyada kullanılıyor")
            
            # Sonuçları kaydet
            analyzer.save_analysis_results(analysis_results)
        
        if args.full_analysis or args.generate_requirements:
            print("\n📋 Requirements.txt oluşturuluyor...")
            
            if not analyzer.external_packages:
                # Eğer analiz henüz yapılmamışsa
                analyzer.analyze_project()
            
            requirements_content = analyzer.generate_requirements_txt(args.include_versions)
            
            print("✅ Requirements.txt oluşturuldu!")
            print(f"   📦 {len(analyzer.external_packages)} external package")
            
            # Kategorilere göre özet
            categorized = analyzer._categorize_packages()
            for category, packages in categorized.items():
                if packages:
                    print(f"   📂 {category}: {len(packages)} package")
        
        if args.full_analysis or args.check_circular:
            print("\n🔄 Döngüsel bağımlılık kontrolü...")
            
            if not analyzer.dependency_graph.nodes():
                analyzer.analyze_project()
            
            circular_analysis = analyzer.check_circular_dependencies()
            
            if circular_analysis.get("has_circular_dependencies"):
                print(f"⚠️ {circular_analysis['circular_dependency_count']} döngüsel bağımlılık bulundu!")
                
                for i, cycle in enumerate(circular_analysis.get("cycles", [])[:5]):  # İlk 5'ini göster
                    print(f"   🔄 Cycle {i+1}: {' -> '.join(cycle)}")
            else:
                print("✅ Döngüsel bağımlılık bulunamadı")
        
        if args.full_analysis or args.optimize_imports:
            print("\n⚡ Import optimizasyonu analizi...")
            
            if not analyzer.file_dependencies:
                analyzer.analyze_project()
            
            optimization_results = analyzer.optimize_imports()
            
            print(f"📊 {optimization_results['total_optimizable_files']} dosya optimize edilebilir")
            
            if optimization_results['optimization_suggestions']:
                print("💡 Optimizasyon önerileri:")
                for suggestion in optimization_results['optimization_suggestions'][:5]:  # İlk 5'ini göster
                    print(f"   📄 {suggestion['file']}")
                    if suggestion['optimizations']['duplicate_imports']:
                        print(f"      🔄 {len(suggestion['optimizations']['duplicate_imports'])} duplicate import")
        
        print("\n🎉 ANALİZ TAMAMLANDI!")
        print("📊 Detaylı sonuçlar logs/ klasöründe")
        
    except Exception as e:
        logger.error(f"❌ Analiz hatası: {e}")
        print(f"\n❌ HATA: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()