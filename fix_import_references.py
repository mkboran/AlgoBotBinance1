#!/usr/bin/env python3
"""
🔧 IMPORT REFERANSLARI DÜZELTİCİ
💎 Yanlış import'ları bulan ve düzelten script

Sorun: Bazı dosyalar 'create_objective_fixed' import etmeye çalışıyor
Çözüm: 'optimization.objective_fixed' olarak değiştir
"""

import os
import re
from pathlib import Path
from typing import List, Tuple, Dict

def find_problematic_imports() -> List[Tuple[Path, List[str]]]:
    """🔍 Problematik import'ları bul"""
    
    print("🔍 Problematik import'lar aranıyor...")
    
    # Tüm Python dosyalarını tara
    python_files = []
    for pattern in ["*.py", "**/*.py"]:
        python_files.extend(Path(".").glob(pattern))
    
    # __pycache__ ve .git filtrele
    python_files = [f for f in python_files if "__pycache__" not in str(f) and ".git" not in str(f)]
    
    problematic_files = []
    
    for py_file in python_files:
        try:
            with open(py_file, 'r', encoding='utf-8') as f:
                content = f.read()
            
            # Problematik import'ları ara
            problematic_lines = []
            
            for line_num, line in enumerate(content.splitlines(), 1):
                if 'create_objective_fixed' in line and ('import' in line or 'from' in line):
                    problematic_lines.append(f"Line {line_num}: {line.strip()}")
            
            if problematic_lines:
                problematic_files.append((py_file, problematic_lines))
                print(f"❌ {py_file}: {len(problematic_lines)} problematik import")
                for line in problematic_lines:
                    print(f"   {line}")
        
        except Exception as e:
            print(f"⚠️ {py_file} okunamadı: {e}")
    
    return problematic_files

def fix_import_in_file(file_path: Path) -> Dict[str, any]:
    """🔧 Dosyadaki import'ları düzelt"""
    
    try:
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        original_content = content
        fixes_applied = []
        
        # Düzeltme kuralları
        fix_patterns = [
            # from create_objective_fixed import ... -> from optimization.objective_fixed import ...
            (
                r'from\s+create_objective_fixed\s+import\s+(.+)',
                r'from optimization.objective_fixed import \1',
                "Fixed 'from create_objective_fixed import'"
            ),
            # import create_objective_fixed -> import optimization.objective_fixed as create_objective_fixed
            (
                r'import\s+create_objective_fixed(?!\s*\.)',
                r'import optimization.objective_fixed as create_objective_fixed',
                "Fixed 'import create_objective_fixed'"
            ),
            # create_objective_fixed.func() -> optimization.objective_fixed.func() 
            # Bu durumda alias kullandık, değişiklik gerekmez
        ]
        
        # Düzeltmeleri uygula
        for pattern, replacement, description in fix_patterns:
            matches = re.finditer(pattern, content, re.MULTILINE)
            for match in matches:
                fixes_applied.append({
                    'original': match.group(0),
                    'fixed': re.sub(pattern, replacement, match.group(0)),
                    'description': description
                })
            
            content = re.sub(pattern, replacement, content, flags=re.MULTILINE)
        
        # Değişiklik varsa kaydet
        if content != original_content:
            # Backup oluştur
            backup_path = file_path.with_suffix(f'.backup_{int(os.time.time())}')
            with open(backup_path, 'w', encoding='utf-8') as f:
                f.write(original_content)
            
            # Yeni içeriği kaydet
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(content)
            
            return {
                'success': True,
                'fixes_applied': fixes_applied,
                'backup_created': str(backup_path)
            }
        else:
            return {
                'success': True,
                'fixes_applied': [],
                'message': 'No changes needed'
            }
    
    except Exception as e:
        return {
            'success': False,
            'error': str(e)
        }

def cleanup_unnecessary_files():
    """🗑️ Gereksiz dosyaları temizle"""
    
    print("🗑️ Gereksiz dosyalar temizleniyor...")
    
    # create_objective_fixed.py dosyası artık gereksiz
    create_obj_file = Path("create_objective_fixed.py")
    if create_obj_file.exists():
        try:
            # Backup oluştur
            backup_file = create_obj_file.with_suffix('.backup_removed')
            create_obj_file.rename(backup_file)
            print(f"✅ create_objective_fixed.py → {backup_file}")
        except Exception as e:
            print(f"⚠️ create_objective_fixed.py silinirken hata: {e}")

def main():
    """Ana düzeltme fonksiyonu"""
    
    print("🔧 IMPORT REFERANSLARI DÜZELTİLİYOR...")
    print("="*60)
    
    # 1. Problematik import'ları bul
    problematic_files = find_problematic_imports()
    
    if not problematic_files:
        print("✅ Problematik import bulunamadı!")
        return
    
    print(f"\n🔧 {len(problematic_files)} dosyada import düzeltme başlıyor...\n")
    
    # 2. Her dosyayı düzelt
    total_fixes = 0
    successful_fixes = 0
    
    for file_path, _ in problematic_files:
        print(f"🔧 Düzeltiliyor: {file_path}")
        
        result = fix_import_in_file(file_path)
        
        if result['success']:
            if result['fixes_applied']:
                successful_fixes += 1
                total_fixes += len(result['fixes_applied'])
                print(f"   ✅ {len(result['fixes_applied'])} düzeltme yapıldı")
                for fix in result['fixes_applied']:
                    print(f"      • {fix['description']}")
                if 'backup_created' in result:
                    print(f"      💾 Backup: {result['backup_created']}")
            else:
                print(f"   📁 Zaten doğru")
        else:
            print(f"   ❌ Hata: {result['error']}")
    
    # 3. Gereksiz dosyaları temizle
    cleanup_unnecessary_files()
    
    print("="*60)
    print(f"🎯 SONUÇ:")
    print(f"   📁 İncelenen dosya: {len(problematic_files)}")
    print(f"   ✅ Başarıyla düzeltilen: {successful_fixes}")
    print(f"   🔧 Toplam düzeltme: {total_fixes}")
    
    if successful_fixes > 0:
        print("\n🎉 IMPORT DÜZELTMELERI TAMAMLANDI!")
        print("✅ Artık import testini tekrar çalıştırabilirsiniz:")
        print("   python test_imports.py")
    else:
        print("\n⚠️ Düzeltme yapılamadı veya gerekli değildi")

if __name__ == "__main__":
    main()