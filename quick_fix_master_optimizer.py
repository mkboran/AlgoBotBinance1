#!/usr/bin/env python3
"""
🔧 HIZLI MANUEL DÜZELTİM
💎 Sadece master_optimizer.py'daki import'ı düzelt
"""

from pathlib import Path

def fix_master_optimizer():
    """master_optimizer.py'daki import'ı düzelt"""
    
    file_path = Path("optimization/master_optimizer.py")
    
    if not file_path.exists():
        print("❌ optimization/master_optimizer.py bulunamadı")
        return False
    
    try:
        # Dosyayı oku
        with open(file_path, 'r', encoding='utf-8') as f:
            content = f.read()
        
        # Backup oluştur
        backup_path = file_path.with_suffix('.backup_manual')
        with open(backup_path, 'w', encoding='utf-8') as f:
            f.write(content)
        
        # Import'ı düzelt
        original_import = "from create_objective_fixed import run_objective"
        new_import = "from optimization.objective_fixed import run_objective"
        
        if original_import in content:
            fixed_content = content.replace(original_import, new_import)
            
            # Dosyayı kaydet
            with open(file_path, 'w', encoding='utf-8') as f:
                f.write(fixed_content)
            
            print("✅ optimization/master_optimizer.py düzeltildi")
            print(f"   📁 Backup: {backup_path}")
            print(f"   🔧 Değişiklik: {original_import} → {new_import}")
            return True
        else:
            print("📁 optimization/master_optimizer.py zaten doğru")
            return True
            
    except Exception as e:
        print(f"❌ Hata: {e}")
        return False

def main():
    print("🔧 HIZLI MANUEL DÜZELTİM BAŞLIYOR...")
    print("="*50)
    
    success = fix_master_optimizer()
    
    print("="*50)
    if success:
        print("🎉 MANUEL DÜZELTİM TAMAMLANDI!")
        print("✅ Artık import testini çalıştırabilirsiniz:")
        print("   python test_imports.py")
    else:
        print("❌ Düzeltme başarısız")

if __name__ == "__main__":
    main()