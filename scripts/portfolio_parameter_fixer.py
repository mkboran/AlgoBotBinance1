#!/usr/bin/env python3
"""
🔧 PROJE PHOENIX - PORTFOLIO PARAMETER STANDARDIZATION SCRIPT
💎 Tüm dosyalarda Portfolio() parametrelerini standartlaştırır

Bu script şunları yapar:
1. ✅ Tüm Python dosyalarını tarar
2. ✅ Portfolio() çağrılarını bulur
3. ✅ Yanlış parametreleri düzeltir
4. ✅ Backup oluşturur
5. ✅ Değişiklikleri raporlar

DOĞRU PARAMETRE: Portfolio(initial_capital_usdt=1000.0)

KULLANIM:
python scripts/portfolio_parameter_fixer.py --fix-all --create-backup
"""

import os
import re
import shutil
from pathlib import Path
from datetime import datetime
from typing import List, Tuple, Dict
import argparse

class PortfolioParameterFixer:
    """🔧 Portfolio Parameter Standardization Engine"""
    
    def __init__(self, project_root: str = "."):
        self.project_root = Path(project_root)
        self.backup_dir = self.project_root / "backups" / f"portfolio_fix_{datetime.now().strftime('%Y%m%d_%H%M%S')}"
        
        # Logs klasörünü oluştur
        (self.project_root / "logs").mkdir(exist_ok=True)
        
        # Düzeltme kuralları
        self.fix_patterns = [
            # Portfolio(initial_balance=...) -> Portfolio(initial_capital_usdt=...)
            (
                r'Portfolio\s*\(\s*initial_balance\s*=\s*([^)]+)\)',
                r'Portfolio(initial_capital_usdt=\1)'
            ),
            # Portfolio(balance=...) -> Portfolio(initial_capital_usdt=...)
            (
                r'Portfolio\s*\(\s*balance\s*=\s*([^)]+)\)',
                r'Portfolio(initial_capital_usdt=\1)'
            ),
            # Portfolio(capital=...) -> Portfolio(initial_capital_usdt=...)
            (
                r'Portfolio\s*\(\s*capital\s*=\s*([^)]+)\)',
                r'Portfolio(initial_capital_usdt=\1)'
            ),
            # Portfolio() -> Portfolio(initial_capital_usdt=1000.0)
            (
                r'Portfolio\s*\(\s*\)',
                r'Portfolio(initial_capital_usdt=1000.0)'
            ),
            # Portfolio(1000) -> Portfolio(initial_capital_usdt=1000)
            (
                r'Portfolio\s*\(\s*(\d+(?:\.\d+)?)\s*\)',
                r'Portfolio(initial_capital_usdt=\1)'
            )
        ]
        
        print("🔧 Portfolio Parameter Fixer başlatıldı")
        print(f"📁 Proje kökü: {self.project_root.absolute()}")
    
    def find_python_files(self) -> List[Path]:
        """🔍 Python dosyalarını bul"""
        
        python_files = []
        
        # Kök dizinde ve alt klasörlerde .py dosyalarını bul
        for pattern in ["*.py", "**/*.py"]:
            python_files.extend(self.project_root.glob(pattern))
        
        # __pycache__ ve .git klasörlerini filtrele
        python_files = [
            f for f in python_files 
            if "__pycache__" not in str(f) and ".git" not in str(f)
        ]
        
        print(f"📊 {len(python_files)} Python dosyası bulundu")
        return python_files
    
    def analyze_file(self, file_path: Path) -> Dict[str, any]:
        """📄 Dosyayı analiz et ve Portfolio çağrılarını tespit et"""
        
        try:
            with open(file_path, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Portfolio çağrılarını bul
            portfolio_calls = []
            
            for line_num, line in enumerate(content.splitlines(), 1):
                if 'Portfolio(' in line:
                    portfolio_calls.append({
                        'line_number': line_num,
                        'original_line': line.strip(),
                        'needs_fix': self._needs_fix(line)
                    })
            
            return {
                'file_path': file_path,
                'portfolio_calls': portfolio_calls,
                'total_calls': len(portfolio_calls),
                'needs_fix': any(call['needs_fix'] for call in portfolio_calls),
                'content': content
            }
            
        except Exception as e:
            return {
                'file_path': file_path,
                'error': str(e),
                'portfolio_calls': [],
                'total_calls': 0,
                'needs_fix': False
            }
    
    def _needs_fix(self, line: str) -> bool:
        """🔍 Bu satırın düzeltilmesi gerekiyor mu?"""
        
        # Eğer zaten doğru parametre kullanılıyorsa düzeltmeye gerek yok
        if 'initial_capital_usdt=' in line:
            return False
        
        # Portfolio() çağrısı var ama doğru parametre yok
        if 'Portfolio(' in line:
            return True
        
        return False
    
    def fix_file(self, file_analysis: Dict[str, any], create_backup: bool = True) -> Dict[str, any]:
        """🔧 Dosyayı düzelt"""
        
        file_path = file_analysis['file_path']
        content = file_analysis['content']
        
        if not file_analysis['needs_fix']:
            return {
                'file_path': file_path,
                'fixed': False,
                'reason': 'No fixes needed',
                'changes': []
            }
        
        try:
            # Backup oluştur
            if create_backup:
                self._create_backup(file_path)
            
            # Düzeltmeleri uygula
            fixed_content = content
            changes = []
            
            for pattern, replacement in self.fix_patterns:
                matches = re.finditer(pattern, fixed_content, re.MULTILINE | re.DOTALL)
                for match in matches:
                    original = match.group(0)
                    new_text = re.sub(pattern, replacement, original)
                    changes.append({
                        'original': original,
                        'fixed': new_text,
                        'position': match.span()
                    })
                
                fixed_content = re.sub(pattern, replacement, fixed_content, flags=re.MULTILINE | re.DOTALL)
            
            # Değişiklik varsa dosyayı kaydet
            if fixed_content != content:
                with open(file_path, 'w', encoding='utf-8') as f:
                    f.write(fixed_content)
                
                return {
                    'file_path': file_path,
                    'fixed': True,
                    'changes': changes,
                    'total_changes': len(changes)
                }
            else:
                return {
                    'file_path': file_path,
                    'fixed': False,
                    'reason': 'No changes made',
                    'changes': []
                }
                
        except Exception as e:
            return {
                'file_path': file_path,
                'fixed': False,
                'error': str(e),
                'changes': []
            }
    
    def _create_backup(self, file_path: Path) -> None:
        """💾 Dosyanın backup'ını oluştur"""
        
        # Backup dizinini oluştur
        self.backup_dir.mkdir(parents=True, exist_ok=True)
        
        # Relative path'i koru
        rel_path = file_path.relative_to(self.project_root)
        backup_path = self.backup_dir / rel_path
        
        # Backup klasör yapısını oluştur
        backup_path.parent.mkdir(parents=True, exist_ok=True)
        
        # Dosyayı kopyala
        shutil.copy2(file_path, backup_path)
    
    def run_analysis(self) -> Dict[str, any]:
        """🔬 Tam analiz yap"""
        
        print("🔬 Portfolio parameter analizi başlıyor...")
        
        python_files = self.find_python_files()
        
        analysis_results = {
            'total_files': len(python_files),
            'files_with_portfolio_calls': 0,
            'files_needing_fixes': 0,
            'total_portfolio_calls': 0,
            'calls_needing_fixes': 0,
            'file_analyses': []
        }
        
        for py_file in python_files:
            file_analysis = self.analyze_file(py_file)
            analysis_results['file_analyses'].append(file_analysis)
            
            if file_analysis['total_calls'] > 0:
                analysis_results['files_with_portfolio_calls'] += 1
                analysis_results['total_portfolio_calls'] += file_analysis['total_calls']
                
                if file_analysis['needs_fix']:
                    analysis_results['files_needing_fixes'] += 1
                    # Düzeltilmesi gereken çağrı sayısını hesapla
                    calls_needing_fix = sum(1 for call in file_analysis['portfolio_calls'] if call['needs_fix'])
                    analysis_results['calls_needing_fixes'] += calls_needing_fix
        
        return analysis_results
    
    def run_fixes(self, create_backup: bool = True) -> Dict[str, any]:
        """🔧 Tüm düzeltmeleri uygula"""
        
        print("🔧 Portfolio parameter düzeltmeleri başlıyor...")
        
        analysis_results = self.run_analysis()
        
        fix_results = {
            'total_files_processed': 0,
            'files_fixed': 0,
            'files_with_errors': 0,
            'total_changes': 0,
            'fix_details': []
        }
        
        for file_analysis in analysis_results['file_analyses']:
            if file_analysis.get('needs_fix', False):
                fix_results['total_files_processed'] += 1
                
                fix_result = self.fix_file(file_analysis, create_backup)
                fix_results['fix_details'].append(fix_result)
                
                if fix_result.get('fixed', False):
                    fix_results['files_fixed'] += 1
                    fix_results['total_changes'] += fix_result.get('total_changes', 0)
                elif 'error' in fix_result:
                    fix_results['files_with_errors'] += 1
        
        return fix_results
    
    def print_analysis_report(self, analysis_results: Dict[str, any]) -> None:
        """📊 Analiz raporunu yazdır"""
        
        print("\n" + "="*80)
        print("📊 PORTFOLIO PARAMETER ANALİZ RAPORU")
        print("="*80)
        print(f"📁 Toplam dosya: {analysis_results['total_files']}")
        print(f"📄 Portfolio çağrısı olan dosya: {analysis_results['files_with_portfolio_calls']}")
        print(f"🔧 Düzeltilmesi gereken dosya: {analysis_results['files_needing_fixes']}")
        print(f"📞 Toplam Portfolio çağrısı: {analysis_results['total_portfolio_calls']}")
        print(f"❌ Düzeltilmesi gereken çağrı: {analysis_results['calls_needing_fixes']}")
        
        if analysis_results['files_needing_fixes'] > 0:
            print("\n🔧 DÜZELTİLMESİ GEREKEN DOSYALAR:")
            for file_analysis in analysis_results['file_analyses']:
                if file_analysis.get('needs_fix', False):
                    file_path = file_analysis['file_path']
                    rel_path = file_path.relative_to(self.project_root)
                    print(f"   ❌ {rel_path}")
                    
                    for call in file_analysis['portfolio_calls']:
                        if call['needs_fix']:
                            print(f"      Line {call['line_number']}: {call['original_line']}")
        
        print("="*80)
    
    def print_fix_report(self, fix_results: Dict[str, any]) -> None:
        """📊 Düzeltme raporunu yazdır"""
        
        print("\n" + "="*80)
        print("🔧 PORTFOLIO PARAMETER DÜZELTİME RAPORU")
        print("="*80)
        print(f"📁 İşlenen dosya: {fix_results['total_files_processed']}")
        print(f"✅ Düzeltilen dosya: {fix_results['files_fixed']}")
        print(f"❌ Hatalı dosya: {fix_results['files_with_errors']}")
        print(f"🔄 Toplam değişiklik: {fix_results['total_changes']}")
        
        if fix_results['files_fixed'] > 0:
            print("\n✅ DÜZELTİLEN DOSYALAR:")
            for fix_detail in fix_results['fix_details']:
                if fix_detail.get('fixed', False):
                    file_path = fix_detail['file_path']
                    rel_path = file_path.relative_to(self.project_root)
                    changes = fix_detail.get('total_changes', 0)
                    print(f"   ✅ {rel_path} ({changes} değişiklik)")
        
        if fix_results['files_with_errors'] > 0:
            print("\n❌ HATALI DOSYALAR:")
            for fix_detail in fix_results['fix_details']:
                if 'error' in fix_detail:
                    file_path = fix_detail['file_path']
                    rel_path = file_path.relative_to(self.project_root)
                    error = fix_detail['error']
                    print(f"   ❌ {rel_path}: {error}")
        
        print("="*80)


def main():
    """Ana çalıştırma fonksiyonu"""
    
    parser = argparse.ArgumentParser(
        description="Portfolio Parameter Standardization Tool",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Kullanım Örnekleri:
  python scripts/portfolio_parameter_fixer.py --analyze-only          # Sadece analiz yap
  python scripts/portfolio_parameter_fixer.py --fix-all               # Tüm dosyaları düzelt
  python scripts/portfolio_parameter_fixer.py --fix-all --no-backup   # Backup olmadan düzelt
        """
    )
    
    parser.add_argument('--analyze-only', action='store_true', help='Sadece analiz yap, düzeltme yapma')
    parser.add_argument('--fix-all', action='store_true', help='Tüm Portfolio parametrelerini düzelt')
    parser.add_argument('--no-backup', action='store_true', help='Backup oluşturma')
    parser.add_argument('--project-root', default='.', help='Proje kök dizini')
    
    args = parser.parse_args()
    
    # Fixer oluştur
    fixer = PortfolioParameterFixer(project_root=args.project_root)
    
    try:
        if args.analyze_only:
            # Sadece analiz
            analysis_results = fixer.run_analysis()
            fixer.print_analysis_report(analysis_results)
            
        elif args.fix_all:
            # Analiz + düzeltme
            analysis_results = fixer.run_analysis()
            fixer.print_analysis_report(analysis_results)
            
            if analysis_results['files_needing_fixes'] > 0:
                print(f"\n🔧 {analysis_results['files_needing_fixes']} dosyada düzeltme başlıyor...")
                
                create_backup = not args.no_backup
                if create_backup:
                    print(f"💾 Backup dizini: {fixer.backup_dir}")
                
                fix_results = fixer.run_fixes(create_backup=create_backup)
                fixer.print_fix_report(fix_results)
                
                if fix_results['files_fixed'] > 0:
                    print(f"\n🎉 {fix_results['files_fixed']} dosya başarıyla düzeltildi!")
                else:
                    print("\n⚠️ Hiçbir dosya düzeltilemedi!")
            else:
                print("\n✅ Düzeltilmesi gereken dosya bulunamadı!")
        
        else:
            # Varsayılan: sadece analiz
            analysis_results = fixer.run_analysis()
            fixer.print_analysis_report(analysis_results)
            print("\n💡 Düzeltme yapmak için --fix-all parametresini kullanın")
            
    except KeyboardInterrupt:
        print("\n🛑 İşlem kullanıcı tarafından durduruldu")
    except Exception as e:
        print(f"\n❌ Beklenmedik hata: {e}")


if __name__ == "__main__":
    main()