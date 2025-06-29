# PROJE: PHOENIX - KUSURSUZ ALGORİTMİK TİCARET SİSTEMİ MASTER PLANI

**Versiyon:** 1.0
**Tarih:** 29 Haziran 2025
**Mimar:** Gemini

---

## **BÖLÜM I: VİZYON VE FELSEFE**

### **1.1. Misyon**

Bu projenin temel amacı, mevcut dağınık ve reaktif kod yığınını, kendi kendini iyileştiren, piyasa koşullarına dinamik olarak adapte olan, matematiksel olarak sağlam temellere oturan ve otonom bir şekilde ultra-yüksek kârlılık hedefleyen, kurumsal düzeyde bir algoritmik ticaret sistemine dönüştürmektir. Bu sistem, sadece kâr etmeyi değil, aynı zamanda riski akıllıca yönetmeyi ve uzun vadede sürdürülebilir olmayı hedefler.

### **1.2. Temel Felsefe**

1.  **Mimari Önce Gelir:** Sağlam bir mimari olmadan, en iyi strateji bile başarısız olur. Sistem, modüler, genişletilebilir ve test edilebilir olmalıdır.
2.  **Determinizm ve Güvenilirlik:** Sistemdeki her süreç, özellikle parametre yönetimi ve testler, %100 güvenilir ve tekrar edilebilir olmalıdır. "Yama" script'lerine yer yoktur.
3.  **Veri Odaklı Kararlar:** Her optimizasyon ve değişiklik, varsayımlara değil, kapsamlı backtest ve çapraz doğrulama verilerine dayanmalıdır.
4.  **Otonom Adaptasyon:** Sistem, piyasa rejimlerindeki değişikliklere manuel müdahale olmadan adapte olabilmeli, kendi parametrelerini ve strateji ağırlıklarını zamanla evrimleştirebilmelidir.
5.  **Risk, Kârın Ayrılmaz Parçasıdır:** Risk yönetimi, bir yan görev değil, sistemin çekirdek bir bileşenidir. Risk, sadece sınırlanmaz; akıllıca yönetilir ve fiyatlanır.

---

## **BÖLÜM II: GELİŞTİRME FAZLARI**

### **FAZ 0: SİSTEM BÜTÜNLÜĞÜ VE MİMARİ RÖNESANS (Süre: 2-3 Gün)**

**Amaç:** Kod tabanındaki kaosu ortadan kaldırıp, üzerine bir imparatorluk kurulabilecek sağlam temelleri atmak.

*   **Adım 0.1: Kod Tabanı Arkeolojisi ve Yeniden Yapılandırma**
    *   **Görev:** `Other/` klasörünü tamamen yok etmek. İçindeki tüm dosyalar analiz edilecek ve ait oldukları yerlere taşınacak.
    *   **Taşıma Planı:**
        *   `utils/` altına: `enhanced_dynamic_exit_system.py`, `kelly_criterion_ml_position_sizing.py`, `global_market_intelligence_system.py` gibi yeniden kullanılabilir modüller.
        *   `scripts/` (Yeni Klasör) altına: `run_critical_fixes.py`, `manual_param_update.py` gibi tek seferlik bakım ve analiz script'leri.
        *   `optimization/` (Yeni Klasör) altına: `smart_range_optimizer.py`, `ultimate_optimizer_optimized.py` gibi optimizasyon araçları.
    *   **Neden?** Modülerlik, okunabilirlik ve yönetilebilirlik için bu ayrım şarttır. Kritik sistem bileşenleri "Diğer" olarak etiketlenemez.

*   **Adım 0.2: Deterministik Parametre Yönetimi**
    *   **Görev:** `auto_update_parameters.py` script'ini ve regex tabanlı kod değiştirme mantığını tamamen ortadan kaldırmak.
    *   **Yeni Sistem:**
        1.  Tüm optimizasyon script'leri, en iyi parametreleri strateji adına göre (`momentum_optimized_params.json`, `bollinger_ml_params.json` gibi) standart bir **JSON dosyasına** yazacak.
        2.  `utils/config.py` veya doğrudan strateji dosyaları, başlangıçta bu JSON dosyalarını okuyarak parametreleri dinamik olarak belleğe yükleyecek.
    *   **Neden?** Kaynak kodunu programatik olarak değiştirmek, en büyük mimari günahtır. Bu yeni sistem, %100 güvenilir, hatasız ve versiyon kontrolü dostu bir yapı sağlar.

*   **Adım 0.3: Merkezi Sistem Çekirdeği (`main.py`)**
    *   **Görev:** `main.py` ve `main_phase5_integration.py` dosyalarını birleştirerek tek ve güçlü bir giriş noktası oluşturmak.
    *   **Yeni `main.py` Fonksiyonları:**
        *   `python main.py live`: Canlı ticareti başlatır.
        *   `python main.py backtest --strategy momentum --start 2023-01-01`: Belirtilen strateji için backtest çalıştırır.
        *   `python main.py optimize --strategy all --trials 5000`: Tüm stratejileri optimize eder.
    *   **Neden?** Sistemin nasıl çalıştırılacağı konusunda belirsizliği ortadan kaldırır ve standart bir operasyonel arayüz sunar.

*   **Adım 0.4: Bağımlılık Grafiği ve `requirements.txt`'nin Yeniden Doğuşu**
    *   **Görev:** Tüm `*.py` dosyalarını statik olarak analiz edip, tüm `import` ifadelerinden bir bağımlılık grafiği çıkarmak.
    *   **Sonuç:** Bu grafe göre, projenin gerçekten ihtiyaç duyduğu tüm kütüphaneleri içeren, sıfırdan ve eksiksiz bir `requirements.txt` dosyası oluşturulacak.
    *   **Neden?** Projenin taşınabilirliğini ve farklı ortamlarda sorunsuz çalışmasını garanti altına alır.

*   **Adım 0.5: Otomatik Doğrulama ve CI/CD Temelleri**
    *   **Görev:** Proje kök dizinine bir `validate_system.py` script'i eklemek. Bu script, temel importların yapılıp yapılamadığını, ana sınıfların (Portfolio, MomentumStrategy vb.) başlatılıp başlatılamadığını kontrol edecek.
    *   **Entegrasyon:** `pre-commit hook` olarak ayarlanarak, bozuk kodun ana depoya (repository) gönderilmesi engellenecek.
    *   **Neden?** Sistemin kendi kendini korumasını sağlar ve gelecekteki "acil durum fix" script'lerine olan ihtiyacı ortadan kaldırır.

### **FAZ 1: BİREYSEL STRATEJİ OPTİMİZASYONUNUN ZİRVESİ (Süre: 1 Hafta)**

**Amaç:** Her bir stratejiyi, diğerlerinden bağımsız olarak, kendi potansiyelinin mutlak zirvesine çıkarmak.

*   **Adım 1.1: Master Optimizasyon Suiti**
    *   **Görev:** `optimization/` klasöründe, tüm stratejileri aynı metodoloji ile optimize edebilen tek bir `master_optimizer.py` script'i oluşturmak.
    *   **Neden?** Optimizasyon sürecini standartlaştırır ve tekrarlanabilir hale getirir.

*   **Adım 1.2: Çok Periyotlu ve Çapraz Doğrulamalı Optimizasyon**
    *   **Görev:** Optimizasyonu tek bir veri seti üzerinde değil, farklı piyasa rejimlerini (Boğa, Ayı, Yatay, Yüksek Volatilite) temsil eden en az 3 farklı tarih aralığında çalıştırmak.
    *   **Metodoloji:** Her periyotta bulunan en iyi parametre setlerinin ortalaması veya en sık tekrar edeni, "sağlam" (robust) parametre seti olarak kabul edilecek.
    *   **Neden?** Bu, "overfitting" (aşırı uyum) sorununu kökünden çözer ve stratejinin her piyasa koşulunda çalışmasını sağlar.

*   **Adım 1.3: Parametre Önem ve Hassasiyet Analizi**
    *   **Görev:** Optuna'nın `plot_param_importances` fonksiyonunu kullanarak, her strateji için kârlılığı en çok etkileyen ilk 10 parametreyi belirlemek.
    *   **Sonuç:** Gelecekteki optimizasyonlarda sadece bu "yüksek etkili" parametrelere odaklanarak zaman kazanılacak.
    *   **Neden?** %20'lik parametre, kârın %80'ini getirir. Bu %20'yi bulmak, verimliliği artırır.

### **FAZ 2: GELİŞMİŞ SİSTEMLERİN SENFONİK ENTEGRASYONU (Süre: 3-4 Gün)**

**Amaç:** `utils/` klasörüne taşınan "gömülü hazineleri" (dinamik çıkış, Kelly, küresel zeka) tüm stratejilerin temel bir parçası haline getirmek.

*   **Adım 2.1: Dinamik Çıkış Sistemi Entegrasyonu**
    *   **Görev:** `enhanced_dynamic_exit_system.py`'deki mantık, tüm stratejiler için temel bir `BaseStrategy` sınıfına entegre edilecek. Stratejiler, bu temel sınıftan miras alacak.
    *   **Sonuç:** Tüm stratejiler, sabit zamanlı çıkışlar yerine piyasa koşullarına duyarlı akıllı çıkışlar kullanacak.

*   **Adım 2.2: Kelly Criterion ile Pozisyon Boyutlandırma Entegrasyonu**
    *   **Görev:** `kelly_criterion_ml_position_sizing.py`'deki mantık, `BaseStrategy` sınıfına entegre edilecek.
    *   **Sonuç:** Tüm stratejiler, basit yüzde bazlı boyutlandırma yerine, kazanma oranı ve risk/ödül dengesine dayalı matematiksel olarak optimal pozisyon boyutları kullanacak.

*   **Adım 2.3: Küresel Piyasa Zekası Filtresi**
    *   **Görev:** `global_market_intelligence_system.py`'den gelen veriler (VIX, DXY, SPY korelasyonu), `StrategyCoordinator` (bkz. Faz 3) seviyesinde bir "risk filtresi" olarak uygulanacak.
    *   **Mantık:** Eğer küresel piyasalarda "riskten kaçış" (risk-off) modu aktifse, tüm stratejilerin pozisyon boyutları otomatik olarak %50 azaltılacak veya yeni pozisyon açılması geçici olarak durdurulacak.
    *   **Neden?** Kripto piyasası, küresel likiditeden bağımsız değildir. Bu filtre, büyük çöküşlerden korunmayı sağlar.

### **FAZ 3: KUANTUM-İLHAMLI PORTFÖY OPTİMİZASYONU (ENSEMBLE) (Süre: 1 Hafta)**

**Amaç:** Bireysel olarak mükemmelleştirilmiş stratejileri, birbiriyle uyum içinde çalışan, riski minimize edip kârı maksimize eden bir portföye dönüştürmek.

*   **Adım 3.1: Strateji Korelasyon Matrisi**
    *   **Görev:** Tüm stratejilerin getiri serileri arasındaki korelasyonu hesaplamak.
    *   **Neden?** Birbiriyle yüksek korelasyona sahip stratejilere aynı anda yatırım yapmak, riski artırır. Bu matris, çeşitlendirmenin temelidir.

*   **Adım 3.2: Risk Paritesi ve Sharpe Oranı Optimizasyonu**
    *   **Görev:** `utils/portfolio_strategy_manager.py`'yi kullanarak, her stratejinin portföydeki ağırlığını, toplam riske olan katkısını eşitleyecek (Risk Parity) ve portföyün genel Sharpe oranını maksimize edecek şekilde dinamik olarak belirlemek.
    *   **Sonuç:** Tek bir stratejinin başarısızlığı, tüm portföyü çökertemez.

*   **Adım 3.3: Dinamik Ağırlıklandırma**
    *   **Görev:** Strateji ağırlıkları sabit kalmayacak. Son 30 gündeki performansına ve mevcut piyasa rejimine (örn: "TRENDING_UP" rejiminde Momentum stratejisinin ağırlığını artır) göre haftalık olarak otomatik güncellenecek.

### **FAZ 4: ÜRETİM ORTAMI VE KAOS MÜHENDİSLİĞİ (Süre: 1 Hafta)**

**Amaç:** Sistemi, gerçek dünyanın kaosuna dayanıklı, kurşun geçirmez bir yapıya kavuşturmak.

*   **Adım 4.1: Dockerizasyon ve Tek Komutla Dağıtım**
    *   **Görev:** Tüm sistem, `Dockerfile` ve `docker-compose.yml` ile paketlenecek. `docker-compose up` komutuyla tüm sistem (bot, veritabanı, izleme) ayağa kalkacak.

*   **Adım 4.2: Canlı Veri Akışı ve Gecikme Simülasyonu**
    *   **Görev:** Backtest sistemine, Binance API'sinden gelen gerçek zamanlı veri akışını ve olası ağ gecikmelerini simüle eden bir modül eklenecek.
    *   **Neden?** Canlı ticaretin en büyük sorunlarından biri olan gecikme ve veri akışı kesintilerine karşı sistemi test eder.

*   **Adım 4.3: Stres Testleri ve Kaos Mühendisliği**
    *   **Görev:** Sistemin dayanıklılığını test etmek için otomatik senaryolar oluşturulacak:
        *   **Kara Kuğu Simülasyonu:** Fiyatın aniden %30 düştüğü bir senaryo.
        *   **API Kesinti Simülasyonu:** Binance API'sinin 5 dakika boyunca yanıt vermediği bir senaryo.
        *   **Veri Bozulma Simülasyonu:** Gelen OHLCV verisine rastgele "gürültü" eklenmesi.
    *   **Neden?** Beklenmedik durumlarda sistemin nasıl tepki vereceğini görmek ve "fail-safe" (güvenli mod) mekanizmalarını doğrulamak için.

*   **Adım 4.4: Kademeli Canlıya Geçiş (Canary Deployment)**
    *   **Görev:** Canlı ticarete toplam sermayenin sadece %5'i ile başlanacak. Sistem bir hafta boyunca kârlı ve stabil çalışırsa, sermaye kademeli olarak artırılacak.

---
