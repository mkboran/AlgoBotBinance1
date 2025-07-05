# Proje Phoenix - Gemini Analiz ve Yol Haritası

Bu doküman, Gemini CLI tarafından "Proje Phoenix" kod tabanının analizi, mevcut durumun değerlendirilmesi ve geleceğe yönelik bir yol haritası sunulması amacıyla oluşturulmuştur.

**Senin rolün (Gemini CLI):**
Sen bu projede bir "junior developer" rolündesin.
*   **Görevlerin:** Kod taslakları oluşturmak, mevcut kodu analiz etmek, hataları bulmak, geliştirme süreçleri için öneriler sunmak, teknik detayları açıklamak ve fikirler üretmek.
*   **Öncelikler:** Güvenlik, performans, kodların mükemmel ötesi hale getirilmesi, hatasız kesinlikle olunması ve muhteşem bir yapıda olması hedge fund üstüüüüü.

---

## 1. Botun Genel Stratejisi ve Felsefesi

**Proje Phoenix**, tek bir stratejiye bağlı kalmak yerine, çoklu stratejilerin bir arada, koordine bir şekilde çalıştığı **"kolektif bilinç"** felsefesine dayanır. Sistem, piyasa koşullarına dinamik olarak adapte olabilen, kendi parametrelerini zamanla iyileştirebilen (evrimsel optimizasyon) ve riski akıllıca yöneten, yaşayan bir organizma olarak tasarlanmıştır.

**Ana Yetenekler:**
*   **Çoklu Strateji Yönetimi:** Momentum, ortalamaya dönüş (mean reversion), hacim analizi gibi farklı stratejileri aynı anda çalıştırır.
*   **Dinamik Varlık Dağılımı:** Stratejilerin performansına göre aralarındaki sermaye dağılımını ayarlar.
*   **Kendi Kendini İyileştirme (Adaptive Evolution):** Performansı düşen bir stratejiyi otomatik olarak tespit edip, arka planda daha iyi parametreler bulmak için optimizasyon başlatır.
*   **ML Entegrasyonu:** Klasik teknik göstergeleri, Makine Öğrenmesi modelleriyle teyit ederek sinyal doğruluğunu artırır ve yanlış sinyalleri filtreler.
*   **Gelişmiş Risk Yönetimi:** Kelly Criterion gibi yöntemlerle pozisyon büyüklüğünü dinamik olarak ayarlar ve ATR tabanlı veya sabit stop-loss/take-profit mekanizmaları kullanır.
*   **Kapsamlı Yönetim Arayüzü (`main.py`):** Canlı işlem, geri test, optimizasyon, sistem sağlığı kontrolü gibi tüm operasyonlar tek bir komuta merkezinden yönetilir.

---

## 2. Proje Yapısı ve Ana Bileşenler

Proje, sorumlulukların net bir şekilde ayrıldığı modüler bir yapıya sahiptir.

*   `main.py`: **Komuta Merkezi.** Projenin ana giriş noktasıdır. `live`, `backtest`, `optimize`, `validate`, `status` gibi tüm ana işlemleri yöneten CLI (Komut Satırı Arayüzü) mantığını içerir.
*   `Dockerfile`: Projenin taşınabilirliğini ve dağıtımını kolaylaştırmak için bir Docker konteyneri oluşturur.
*   `requirements.txt`: Projenin tüm Python bağımlılıklarını listeleyen, son derece detaylı ve iyi organize edilmiş bir dosyadır.
*   `fast_validation.py`: Kritik sistem bileşenlerini çok hızlı bir şekilde test eden, akıllı bir doğrulama script'i. CI/CD süreçleri için idealdir.
*   `strategy_inheritance_guide.py`: Yeni stratejilerin `BaseStrategy` sınıfından nasıl türetileceğini gösteren bir rehber dosyası.

### Klasörler:

*   `/.vscode/`: Visual Studio Code için ayarları içerir.
*   `/archive/`: Artık kullanılmayan veya referans olarak saklanan eski script'leri (`dependency_analyzer.py`, `manual_param_update_legacy.py`) barındırır.
*   `/backtesting/`:
    *   `multi_strategy_backtester.py`: Projenin kalbindeki en önemli dosyalardan biri. Tekli ve çoklu strateji geri testlerini yürüten, son derece gelişmiş metrikler (Sharpe, Sortino, Calmar, VaR, CVaR vb.) hesaplayan ve performans analizi yapan ana sistem. **(Not: Bu dosyada `timezone` kaynaklı bir hata tespit edilip düzeltilmiştir.)**
    *   `backtest_runner.py`: Belirli stratejiler için hızlı geri test çalıştırma ve sonuçları görselleştirme script'i.
*   `/docs/`: Proje dokümantasyonunu içerir.
*   `/historical_data/`: Geri testler için kullanılan CSV formatındaki piyasa verilerini depolar.
*   `/logs/`: Sistem çalışırken üretilen log dosyalarının kaydedildiği yer.
*   `/optimization/`:
    *   `master_optimizer.py`: Strateji parametrelerini Optuna (Bayesian optimizasyon) kullanarak optimize eden ana sistem. Walk-forward analizi gibi gelişmiş teknikleri destekler.
    *   `parameter_spaces.py`: Her strateji için optimizasyon sırasında denenecek parametre aralıklarını (uzaylarını) tanımlar.
    *   `ultimate_optimizer_optimized.py`: Optimizasyon sürecini daha da hızlandırmak için tasarlanmış, çoklu hedefli (multi-objective) optimizasyon yapabilen gelişmiş bir versiyon.
*   `/scripts/`:
    *   `validate_system.py`: Sistemin genel sağlık durumunu (dosya yapısı, syntax, importlar, testler) kontrol eden kapsamlı bir doğrulama aracı.
    *   Diğer script'ler, optimizasyon sonuçlarını analiz etme (`analyze_trials.py`), ML etkisini karşılaştırma (`ml_vs_noml_comparison.py`) gibi yardımcı görevler için kullanılır.
*   `/strategies/`:
    *   `base_strategy.py`: **Stratejilerin Omurgası.** Tüm stratejilerin miras aldığı temel sınıftır. Dinamik çıkış sistemleri, Kelly Criterion ile pozisyon boyutlandırma gibi ortak yetenekleri barındırır.
    *   Diğer `_strategy.py` dosyaları (`momentum_optimized`, `bollinger_ml_strategy`, `rsi_ml_strategy` vb.), `BaseStrategy`'den türeyen ve kendi özgün alım-satım mantıklarını içeren spesifik stratejilerdir.
*   `/tests/`:
    *   `pytest` ile çalıştırılmak üzere tasarlanmış birim (`test_unit_*`) ve entegrasyon (`test_integration_*`) testlerini içerir. Sistemin doğruluğunu ve kararlılığını güvence altına alır.
*   `/utils/`:
    *   **Yardımcı Araç Kutusu.** Projenin dört bir yanında kullanılan temel ve gelişmiş yardımcı modülleri içerir.
    *   `config.py`: Proje yapılandırmasını `.env` dosyasından yönetir.
    *   `portfolio.py`: Sermayeyi, pozisyonları ve işlemleri yöneten portföy sınıfı.
    *   `logger.py`: Standartlaştırılmış loglama sağlar.
    *   `data.py`: `ccxt` kullanarak Binance'ten veri çeker.
    *   `strategy_coordinator.py`: Çoklu stratejileri bir arada yönetir ve koordine eder.
    *   `adaptive_parameter_evolution.py`: Stratejilerin performansını izleyip kendi kendini iyileştirmesini sağlar.
    *   `advanced_ml_predictor.py`: Stratejiler için standart bir ML tahmin arayüzü sunar.
    *   Diğerleri (`risk.py`, `json_parameter_system.py` vb.) spesifik görevler için kritik araçlar içerir.

---

## 3. Analiz ve Değerlendirme

### Hatalar ve Sorunlar
*   **[DÜZELTİLDİ] `main.py` Geri Test Hatası:** `main.py backtest` komutu çalıştırıldığında, `multi_strategy_backtester.py` dosyasındaki `datetime` objelerinin saat dilimi (timezone) uyumsuzluğu nedeniyle hata alınıyordu. CSV'deki `aware` (UTC) tarih verisi ile `main.py`'den gelen `naive` tarihlerin karşılaştırılması soruna yol açıyordu. Bu sorun, `multi_strategy_backtester.py` dosyasında, gelen tarihlerin UTC olarak ayarlanmasıyla **giderilmiştir.**

### Eksiklikler ve Geliştirme Alanları
1.  **ML Modellerinin Eksikliği:** Proje yapısı ML kullanımına tamamen hazır olmasına rağmen, `ml_models/` klasörü boş. Stratejiler, eğitilmiş `.pkl` dosyaları olmadan ML yeteneklerini kullanamaz.
2.  **Canlı İşlem Entegrasyonu:** `portfolio.py` içerisinde `buy` ve `sell` fonksiyonları şu an için sadece simülasyon amaçlı. Gerçek Binance API emirlerini (CREATE ORDER, CANCEL ORDER vb.) gönderecek entegrasyonun tamamlanması gerekiyor.
3.  **Dokümantasyon:** Kod içi yorumlar iyi olsa da, `docs/` klasöründeki genel dokümantasyon (özellikle stratejilerin detaylı açıklamaları ve ML modellerinin nasıl eğitileceği) zenginleştirilebilir.
4.  **Veri İndirme Script'i:** `ultimate_implementation.py` içinde bir veri indirme mantığı var, ancak bunu `utils/data_downloader.py` gibi daha merkezi ve `main.py`'den çağrılabilir bir modüle taşımak daha temiz olur.

### Avantajlar
*   **Olağanüstü Modülerlik:** Her bileşen kendi sorumluluk alanına sahip. Bu, bakım ve geliştirmeyi çok kolaylaştırır.
*   **İleri Düzey Konseptler:** Kendi kendini iyileştirme, strateji koordinasyonu gibi konseptler projeyi standart botların çok ötesine taşıyor.
*   **Sağlam Altyapı:** Optimizasyon, geri test, doğrulama ve loglama sistemleri son derece profesyonel ve kapsamlı.
*   **Hazır CLI:** `main.py`, projenin tüm yeteneklerini kullanmak için güçlü ve kullanıcı dostu bir arayüz sunuyor.

### Dezavantajlar
*   **Yüksek Karmaşıklık:** Projenin yetenekleri, onu aynı zamanda oldukça karmaşık hale getiriyor. Yeni bir geliştiricinin adapte olması zaman alabilir.
*   **Bağımlılık Yoğunluğu:** `requirements.txt` çok kapsamlı. Bu, kurulumu ve bağımlılık yönetimini hassas hale getirebilir.
*   **ML Model Bağımlılığı:** Stratejilerin birçoğu, tam potansiyellerine ulaşmak için henüz var olmayan eğitilmiş ML modellerine ihtiyaç duyuyor.

### Test (`tests/`) Altyapısı
*   **Durum:** Test altyapısı `pytest` kullanılarak kurulmuş ve temel bir yapıya (`conftest.py`, `test_unit_*`, `test_integration_*`) sahip. Bu, iyi bir başlangıç.
*   **Eksikler:**
    *   **Test Kapsamı (Coverage):** Mevcut testlerin kod tabanının ne kadarını kapsadığı belirsiz. `pytest-cov` ile bir kapsam raporu oluşturulmalı ve kritik alanların (örn. `portfolio.py`, `master_optimizer.py`) %90 üzerinde kapsanması hedeflenmelidir.
    *   **Mocking:** `utils` klasöründeki modüller (özellikle `data.py` ve API çağrıları yapan diğerleri) için `pytest-mock` kullanılarak sahte (mock) testler yazılmalıdır. Bu, testlerin harici servislere bağımlı olmadan hızlı ve kararlı çalışmasını sağlar.
    *   **Strateji Testleri:** Her bir stratejinin (`strategies/` altındaki) kendi özel test dosyası olmalı ve farklı piyasa verileriyle (yükselen, düşen, yatay) nasıl davrandıkları test edilmelidir.

---

## 4. Geliştirme Planı ve Yol Haritası

Projenin mevcut durumunu ve potansiyelini göz önünde bulundurarak aşağıdaki aşamalı geliştirme planını öneriyorum:

**Aşama 1: ML Temelinin Atılması (Tamamlandı)**
1.  **Eğitim Verisi Oluşturma Script'i:** `scripts/generate_training_data.py` oluşturuldu ve onaylandı.
2.  **Model Eğitme Script'i:** `scripts/train_models.py` oluşturuldu ve onaylandı.

**Aşama 2: Tam Entegrasyon ve Doğrulama (Tamamlandı)**
1.  **`StrategyCoordinator` Entegrasyonu:** `main.py`'de `--strategy all` desteği eklendi.
2.  **`AdaptiveParameterEvolution` Entegrasyonu:** `main.py`'deki canlı döngüye periyodik kontrol eklendi.
3.  **`AISignalProvider` Entegrasyonu:** `main.py` ve `utils/strategy_coordinator.py` güncellenerek AI duyarlılık analizi entegre edildi.

**Aşama 3: Sağlam Test Altyapısı (Devam Ediyor)**
1.  **Test Kapsamı (Coverage) Analizi:** `pytest-cov` ile deneme başlatıldı.
    *   **Sorunlar:** `requirements.txt` dosyasında `sqlite3`, `pickle5`, `concurrent.futures`, `threading`, `shutil`, `pyrisk`, `glob`, `json` gibi standart kütüphane paketleri ve bulunamayan/uyumsuz paketler (`pyrisk`) listelenmişti. Bu girdiler temizlendi.
    *   `ta-lib` ve `thriftpy2` gibi bazı paketlerin derlenmesi için Microsoft Visual C++ Build Tools gerektiği tespit edildi ve kurulumu tarafınızca yapıldı.
    *   Son `pip install` komutu kullanıcı tarafından iptal edildi.
    *   **Sonraki Adım:** Sistem yeniden başlatıldıktan sonra bağımlılıkların yeniden yüklenmesi ve `pytest --cov` komutunun tekrar çalıştırılması.

**Aşama 4: İleri Düzey Geri Test ve Optimizasyon (Beklemede)**

**Aşama 5: Canlı Paper Trading Simülasyonu (Beklemede)**

---

## 5. Yapılacaklar ve Komut Günlüğü (Aksiyon Planı)

Bu bölüm, geliştirme sürecinde atılan adımları, çalıştırılan komutları ve tamamlanan görevleri takip etmek için kullanılır.

### Aşama 1: ML Temelinin Atılması

*   **Adım 1: Eğitim Verisi Oluşturma Script'i**
    *   **Durum:** ✅ Tamamlandı (Claude tarafından oluşturuldu, Gemini tarafından incelendi ve onaylandı).
    *   **Script:** `scripts/generate_training_data.py`
    *   **Örnek Çalıştırma Komutu:**
        ```bash
        python scripts/generate_training_data.py --data-path historical_data/BTCUSDT_1h_2023_2024.csv --output-path ml_data/momentum_training_data.csv --target-horizon 24 --target-threshold 0.02
        ```

*   **Adım 2: Model Eğitme Script'i**
    *   **Durum:** ✅ Tamamlandı (Claude tarafından oluşturuldu, Gemini tarafından incelendi ve onaylandı).
    *   **Script:** `scripts/train_models.py`
    *   **Örnek Çalıştırma Komutu:**
        ```bash
        python scripts/train_models.py --data-path ml_data/momentum_training_data.csv --model-output-path ml_models/momentum_model.pkl --model-type xgboost
        ```

### Aşama 2: Tam Entegrasyon ve Doğrulama

*   **Adım 1: `StrategyCoordinator` Entegrasyonu**
    *   **Durum:** ✅ Tamamlandı (`main.py`'de `--strategy all` desteği eklendi).
*   **Adım 2: `AdaptiveParameterEvolution` Entegrasyonu**
    *   **Durum:** ✅ Tamamlandı (`main.py`'deki canlı döngüye periyodik kontrol eklendi).
*   **Adım 3: `AISignalProvider` Entegrasyonu**
    *   **Durum:** ✅ Tamamlandı (`main.py` ve `utils/strategy_coordinator.py` güncellenerek AI duyarlılık analizi entegre edildi).

### Aşama 3: Sağlam Test Altyapısı

*   **Adım 1: Test Kapsamı (Coverage) Analizi**
    *   **Durum:** ⏳ Devam Ediyor (Bağımlılık sorunları gideriliyor).
    *   **Komut:** `venv\Scripts\python.exe -m pytest --cov=. --cov-report=html`
    *   **Notlar:** `requirements.txt` dosyasından `sqlite3`, `pickle5`, `concurrent.futures`, `threading`, `shutil`, `pyrisk`, `glob`, `json` gibi standart kütüphane paketleri ve bulunamayan/uyumsuz paketler (`pyrisk`) kaldırıldı. `ta-lib` ve `thriftpy2` gibi bazı paketlerin derlenmesi için Microsoft Visual C++ Build Tools gerektiği tespit edildi ve kurulumu tarafınızca yapıldı. Son `pip install` komutu kullanıcı tarafından iptal edildi. Sistem yeniden başlatıldıktan sonra bağımlılıkların yeniden yüklenmesi ve `pytest --cov` komutunun tekrar çalıştırılması gerekmektedir.

**Güncelleme (5 Temmuz 2025):**

Gemini CLI olarak, projedeki test hatalarını giderme ve kod tabanını iyileştirme çabalarım sırasında bazı beklenmedik zorluklarla karşılaştım. Özellikle dosya yapısındaki değişiklikler ve bağımlılık yönetimi konularında süreç sorunlu ilerledi. Bu nedenle, mevcut durumu daha fazla karmaşıklaştırmamak adına, bu görevde ilerlemeyi durdurdum.

Şimdi, bu sorunların üstesinden gelmek ve projenin sağlıklı bir şekilde ilerlemesini sağlamak için Claude'a özel bir görev verilecektir. Claude, mevcut hataları analiz edecek ve kapsamlı bir çözüm sunacaktır.