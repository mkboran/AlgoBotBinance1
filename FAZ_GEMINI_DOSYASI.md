# PROJE: PHOENIX - KUSURSUZ ALGORİTMİK TİCARET SİSTEMİ MASTER PLANI

**Versiyon:** 2.0
**Tarih:** 30 Haziran 2025
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

### **FAZ 1: MİMARİ TEMELLERİN TAMAMLANMASI (Süre: 1-2 Gün)**

**Amaç:** Faz 0'da atlanan kritik eksikleri gidererek, üzerine bir imparatorluk kurulabilecek, sarsılmaz temelleri atmak.

*   **Adım 1.1: Merkezi Sistem Çekirdeği (`main.py`)**
    *   **Görev:** Proje kök dizininde, tüm sistem operasyonlarını (canlı ticaret, backtest, optimizasyon) yönetecek tek ve güçlü bir `main.py` giriş noktası oluşturmak.
    *   **Yapı:** `argparse` kullanarak komut satırından yönetilebilir bir arayüz sunacak:
        *   `python main.py live`: Canlı ticareti başlatır.
        *   `python main.py backtest --strategy momentum`: Belirtilen strateji için backtest çalıştırır.
        *   `python main.py optimize --strategy all`: Tüm stratejileri optimize eder.
    *   **Neden?** Sistemin nasıl çalıştırılacağı konusundaki belirsizliği ortadan kaldırır ve standart bir operasyonel arayüz sunar. Bu, kurumsal düzeydeki sistemlerin temel taşıdır.

*   **Adım 1.2: Stratejilerin Ortak Beyni (`BaseStrategy`)**
    *   **Görev:** `strategies/` klasörü içinde, tüm stratejilerin miras alacağı, ortak yetenekleri barındıracak bir `base_strategy.py` ve içinde `BaseStrategy` sınıfı oluşturmak.
    *   **Entegrasyon:** Mevcut tüm strateji sınıfları (`EnhancedMomentumStrategy`, `BollingerMLStrategy` vb.) bu temel sınıftan türetilecek.
    *   **Neden?** Kod tekrarını önler, gelecekteki entegrasyonları (dinamik çıkış, Kelly Criterion vb.) tek bir merkezden yönetmeyi sağlar ve mimariyi sağlamlaştırır.

*   **Adım 1.3: Bağımlılıkların Kesinleştirilmesi (`requirements.txt`)**
    *   **Görev:** `dependency_analyzer.py` script'ini kullanarak projenin tüm `import` ifadelerini analiz etmek ve bu analize göre sıfırdan, temiz ve eksiksiz bir `requirements.txt` dosyası oluşturmak.
    *   **Neden?** Projenin taşınabilirliğini, farklı ortamlarda sorunsuz çalışmasını ve bağımlılık kaosunu tamamen ortadan kaldırmayı garanti altına alır.

### **FAZ 2: GELİŞMİŞ SİSTEMLERİN SENFONİK ENTEGRASYONU (Süre: 3-4 Gün)**

**Amaç:** `utils/` klasöründeki "gömülü hazineleri" (dinamik çıkış, Kelly, küresel zeka) `BaseStrategy` üzerinden tüm stratejilerin temel bir parçası haline getirmek.

*   **Adım 2.1: Dinamik Çıkış Sistemi Entegrasyonu**
    *   **Görev:** `enhanced_dynamic_exit_system.py`'deki mantık, `BaseStrategy` sınıfına entegre edilecek.
    *   **Sonuç:** Tüm stratejiler, sabit zamanlı çıkışlar yerine piyasa koşullarına duyarlı akıllı çıkışlar kullanacak.

*   **Adım 2.2: Kelly Criterion ile Pozisyon Boyutlandırma Entegrasyonu**
    *   **Görev:** `kelly_criterion_ml_position_sizing.py`'deki mantık, `BaseStrategy` sınıfına entegre edilecek.
    *   **Sonuç:** Tüm stratejiler, basit yüzde bazlı boyutlandırma yerine, kazanma oranı ve risk/ödül dengesine dayalı matematiksel olarak optimal pozisyon boyutları kullanacak.

*   **Adım 2.3: Küresel Piyasa Zekası Filtresi**
    *   **Görev:** `global_market_intelligence_system.py`'den gelen veriler, `StrategyCoordinator` (bkz. Faz 4) seviyesinde bir "risk filtresi" olarak uygulanacak.
    *   **Mantık:** Küresel piyasalarda "riskten kaçış" (risk-off) modu aktifse, tüm stratejilerin pozisyon boyutları otomatik olarak %50 azaltılacak veya yeni pozisyon açılması geçici olarak durdurulacak.

### **FAZ 3: BİREYSEL STRATEJİ OPTİMİZASYONUNUN ZİRVESİ (Süre: 1 Hafta)**

**Amaç:** Her bir stratejiyi, diğerlerinden bağımsız olarak, kendi potansiyelinin mutlak zirvesine çıkarmak.

*   **Adım 3.1: Master Optimizasyon Suiti**
    *   **Görev:** `optimization/` klasöründe, tüm stratejileri aynı metodoloji ile optimize edebilen tek bir `master_optimizer.py` script'i oluşturmak.

*   **Adım 3.2: Çok Periyotlu ve Çapraz Doğrulamalı Optimizasyon**
    *   **Görev:** Optimizasyonu tek bir veri seti üzerinde değil, farklı piyasa rejimlerini (Boğa, Ayı, Yatay) temsil eden en az 3 farklı tarih aralığında çalıştırmak.

*   **Adım 3.3: Parametre Önem ve Hassasiyet Analizi**
    *   **Görev:** Optuna'nın `plot_param_importances` fonksiyonunu kullanarak, her strateji için kârlılığı en çok etkileyen ilk 10 parametreyi belirlemek.

### **FAZ 4: KUANTUM-İLHAMLI PORTFÖY OPTİMİZASYONU (ENSEMBLE) (Süre: 1 Hafta)**

**Amaç:** Bireysel olarak mükemmelleştirilmiş stratejileri, birbiriyle uyum içinde çalışan, riski minimize edip kârı maksimize eden bir portföye dönüştürmek.

*   **Adım 4.1: Strateji Korelasyon Matrisi**
    *   **Görev:** Tüm stratejilerin getiri serileri arasındaki korelasyonu hesaplamak.

*   **Adım 4.2: Risk Paritesi ve Sharpe Oranı Optimizasyonu**
    *   **Görev:** Her stratejinin portföydeki ağırlığını, toplam riske olan katkısını eşitleyecek (Risk Parity) ve portföyün genel Sharpe oranını maksimize edecek şekilde dinamik olarak belirlemek.

*   **Adım 4.3: Dinamik Ağırlıklandırma**
    *   **Görev:** Strateji ağırlıkları sabit kalmayacak. Son 30 gündeki performansına ve mevcut piyasa rejimine göre haftalık olarak otomatik güncellenecek.

### **FAZ 5: ÜRETİM ORTAMI VE KAOS MÜHENDİSLİĞİ (Süre: 1 Hafta)**

**Amaç:** Sistemi, gerçek dünyanın kaosuna dayanıklı, kurşun geçirmez bir yapıya kavuşturmak.

*   **Adım 5.1: Dockerizasyon ve Tek Komutla Dağıtım**
*   **Adım 5.2: Canlı Veri Akışı ve Gecikme Simülasyonu**
*   **Adım 5.3: Stres Testleri ve Kaos Mühendisliği**
*   **Adım 5.4: Kademeli Canlıya Geçiş (Canary Deployment)**
