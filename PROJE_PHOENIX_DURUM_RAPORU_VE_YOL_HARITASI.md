# PROJE PHOENIX: DURUM RAPORU VE SÜPER GÜÇ YOL HARİTASI

**Versiyon:** 7.0
**Tarih:** 1 Temmuz 2025
**Mimar:** Gemini

---

## **BÖLÜM I: GELİŞTİRME FELSEFESİ - "TEK SEFERDE MÜKEMMELLİK"**

Proje geliştirme sürecinde, özellikle LLM'lerle çalışırken, verimliliği en üst düzeye çıkarmak ve revizyon sayısını minimize etmek esastır. Bu nedenle, her bir dosya veya bileşen için yapılacak değişiklikler, **o bileşenin ulaşması hedeflenen nihai ve "Arşı Kalite" durumunu en başından tanımlayan, tek ve kapsamlı bir görevle** ele alınacaktır. Bir dosya üzerinde tekrar tekrar değişiklik yapmak yerine, o dosyanın projedeki nihai rolü ve tüm yetenekleri tek seferde planlanıp uygulanacaktır.

---

## **BÖLÜM II: SÜPER GÜÇ YOL HARİTASI - PHOENIX'İN YENİDEN DOĞUŞU**

**FAZ 1: MİMARİ TEMELLERİN ATILMASI** `[TAMAMLANDI]`

**FAZ 2: MERKEZİ ZEKANIN ENTEGRASYONU** `[TAMAMLANDI]`

**FAZ 3: STRATEJİLERİN KOLEKTİF BİLİNCE ULAŞMASI** `[TAMAMLANDI]`

**FAZ 4: KENDİNİ İYİLEŞTİREN VE EVRİMLEŞEN SİSTEM** `[MEVCUT GÖREV]`
- **Hedef:** Sistemin manuel müdahaleye olan ihtiyacını minimize etmek ve kendi performansını sürekli olarak iyileştiren, yaşayan bir organizma yaratmak.
- **Aksiyon:** `utils/adaptive_parameter_evolution.py` dosyasını, "Arşı Kalite" tanımına göre oluşturmak.

**FAZ 5: SİSTEMİN CANLANMASI - `main.py` ENTEGRASYONU (Sıradaki ve Final Görev)**
- **Hedef:** Şimdiye kadar oluşturulan tüm gelişmiş sistemleri (Stratejiler, Coordinator, Optimizer, Backtester, Parameter Evolution) bir araya getirip, projenin ana giriş noktası olan `main.py` üzerinden tam fonksiyonel, komut satırından yönetilebilir, canlı bir sisteme dönüştürmek.
- **Aksiyon:** `main.py` dosyasını, aşağıdaki "Arşı Kalite" tanımına göre yeniden yapılandırmak.

#### **"Arşı Kalite" `main.py` v2.0 Tanımı:**

`main.py` dosyası, projenin merkezi komuta merkezi olacak ve aşağıdaki yeteneklere sahip olacak şekilde güncellenmelidir:

1.  **Sınıf Yapısı (`PhoenixTradingSystem`):**
    - Ana mantığı barındıran bir sınıf oluşturulacak.
    - `__init__` metodu, sistemin tüm temel bileşenlerini (`Portfolio`, `StrategyCoordinator`, `MasterOptimizer`, `MultiStrategyBacktester`, `SystemValidator`, `AdaptiveParameterEvolution`) başlatacak veya `None` olarak tanımlayacak.
    - Desteklenen tüm stratejileri bir "registry" (sözlük) içinde tutacak.

2.  **Komut Satırı Arayüzü (`argparse`):**
    - `live`: Canlı ticaret modunu başlatır. `--strategy`, `--capital`, `--symbol` gibi argümanlar almalıdır.
    - `backtest`: Gelişmiş backtest modunu başlatır. `--strategy`, `--start-date`, `--end-date`, `--capital`, `--data-file` gibi argümanlar almalıdır.
    - `optimize`: Optimizasyon modunu başlatır. `--strategy`, `--trials`, `--storage`, `--walk-forward` gibi argümanlar almalıdır.
    - `validate`: `SystemValidator`'ı çağırarak sistemin genel sağlık durumunu kontrol eder.
    - `status`: Sistemin anlık durumu (aktif stratejiler, portföy değeri, PnL vb.) hakkında bir rapor sunar.

3.  **Fonksiyonların Doldurulması:**
    - `run_live_trading`: `StrategyCoordinator`'ı periyodik olarak çalıştıran bir `async` döngü içermelidir. Her döngüde, piyasa verisini almalı, koordinatöre göndermeli ve belirli aralıklarla `AdaptiveParameterEvolution` sistemini tetiklemelidir.
    - `run_backtest`: `MultiStrategyBacktester`'ı, komut satırından gelen argümanlara uygun bir `BacktestConfiguration` ile çağıracak ve sonuçları konsola formatlayarak yazdıracaktır.
    - `run_optimization`: `MasterOptimizer`'ı, komut satırından gelen argümanlara uygun bir `OptimizationConfig` ile çağıracak ve optimizasyon sonuçlarını özetleyecektir.

4.  **Hata Yönetimi ve Güvenlik:**
    - Tüm ana operasyonlar, geniş kapsamlı `try...except` blokları içinde çalışmalı, hatalar düzgün bir şekilde loglanmalı ve sistemin çökmesi engellenmelidir.
    - `run_live_trading` içinde, portföyde belirli bir orandan fazla düşüş (%15 gibi) yaşanması durumunda sistemi otomatik olarak durduracak bir **acil durum freni (emergency brake)** mekanizması bulunmalıdır.