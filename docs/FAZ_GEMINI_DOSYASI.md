# PROJE: PHOENIX - KUSURSUZ ALGORİTMİK TİCARET SİSTEMİ MASTER PLANI

**Versiyon:** 9.0
**Tarih:** 1 Temmuz 2025
**Mimar:** Gemini

---

## **BÖLÜM I: GEMINI (BEN) İÇİN BAŞLANGIÇ NOTLARI**

**GÖREV:** Bu dosyayı her sohbet başlangıcında ilk olarak oku. Projenin vizyonunu, felsefesini, mevcut durumunu ve sıradaki görevi anlamak için bu dosyayı temel al. Her adıma başlamadan önce, bu dosyadaki yol haritasını ve proje günlüğünü gözden geçir.

**PROJE FELSEFESİ: "TEK SEFERDE MÜKEMMELLİK"**
- **Ana Prensip:** Bir dosyayı veya bileşeni defalarca değiştirmek yerine, o bileşenin ulaşması hedeflenen nihai ve "Arşı Kalite" durumunu en başından tanımlayan, tek ve kapsamlı bir görevle ilerleyeceğiz.
- **Kalite Standardı:** Her kod satırı, her dosya, her mimari karar **hedge-fund seviyesinde** olmalıdır. Mükemmellikten daha azı kabul edilemez.
- **Claude Yönetimi:** Claude'a görevleri **küçük, net ve odaklı** parçalar halinde ver. Ancak bir dosyayı ilgilendiren tüm değişiklikleri **tek bir görevde, kapsamlı bir şekilde** iste. Claude bir dosyayı güncellediğinde, hangi dosyayı güncellediğini belirtmesini iste.

--- 

## **BÖLÜM II: PROJE DURUMU VE YOL HARİTASI**

### **MEVCUT DURUM (1 Temmuz 2025):**

- **Tamamlanan Fazlar:**
  - **FAZ 1: MİMARİ TEMELLER** `[TAMAMLANDI]`
  - **FAZ 2: MERKEZİ ZEKANIN ENTEGRASYONU** `[TAMAMLANDI]`
  - **FAZ 3: STRATEJİLERİN KOLEKTİF BİLİNCE ULAŞMASI** `[TAMAMLANDI]`

- **Mevcut Görev:**
  - **FAZ 4: KENDİNİ İYİLEŞTİREN VE EVRİMLEŞEN SİSTEM** `[CLAUDE ÜZERİNDE]`
    - **Aksiyon:** Claude, `utils/adaptive_parameter_evolution.py` dosyasını, stratejilerin performansını izleyip, zayıflayanları otomatik olarak yeniden optimize edecek şekilde oluşturuyor.

### **SIRADAKİ GÖREV PLANI:**

1.  **FAZ 5: SİSTEMİN CANLANMASI - `main.py` ENTEGRASYONU**
    - **Hedef:** Projenin tüm bileşenlerini bir araya getirerek, `main.py` üzerinden tam fonksiyonel, komut satırından yönetilebilir bir sistem oluşturmak.
    - **Aksiyon:** Claude Faz 4'ü bitirdikten sonra, `main.py` dosyasını "Arşı Kalite" tanımına göre yeniden yapılandırması için görevlendirilecek.

2.  **FAZ 6: LANSMAN ÖNCESİ SON KONTROLLER VE DOĞRULAMA**
    - **Hedef:** Sistemin canlıya geçmeden önce matematiksel ve pratik olarak kârlılığını ve sağlamlığını kanıtlamak.
    - **Aksiyon Sırası (Kesinlikle Bu Sırayla):
**
      1.  **Nihai Optimizasyon:** `main.py optimize` ile en ideal parametreleri bul.
      2.  **Kapsamlı Backtest:** Optimize edilmiş parametrelerle `main.py backtest` ile geçmiş performansı doğrula.
      3.  **Kağıt Ticareti (Paper Trading):** `main.py live` komutunu paper trading modunda çalıştırarak gerçek zamanlı piyasa koşullarında test et.
      4.  **Düşük Bütçeli Canlı Test:** Her şey yolundaysa, minimum riskle gerçek piyasaya geç.

3.  **FAZ 7: GELECEK GELİŞTİRMELERİ (Uzun Vadeli Vizyon)**
    - **Hedef:** Çekirdek sistem kanıtlandıktan sonra yeni yetenekler eklemek.
    - **Aksiyonlar:** Yeni pariteler eklemek, kontrollü kaldıraç denemeleri yapmak.
