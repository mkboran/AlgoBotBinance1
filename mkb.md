Claude için Görev Metni:


  Görev: Proje Phoenix'in 5. ve son fazı olan "Sistemin Canlanması"nı gerçekleştir. Bu görev, projenin tüm gelişmiş bileşenlerini bir araya
  getirerek, main.py dosyasını tam fonksiyonel, komut satırından yönetilebilir bir "Komuta Merkezi" haline getirmektir.


  Dosya: main.py (Bu dosyayı aşağıdaki "Arşı Kalite" tanımına göre tamamen yeniden yapılandır.)

  "Arşı Kalite" `main.py` v2.0 Gereksinimleri:


  Aşağıdaki tüm yetenekleri, main.py dosyası içinde tek seferde ve eksiksiz bir şekilde implemente etmelisin.


  1. Gerekli Import'lar:
      - asyncio, argparse, logging, sys, pathlib gibi temel modülleri import et.
      - Projenin diğer tüm ana bileşenlerini import et: Portfolio, StrategyCoordinator, MasterOptimizer, MultiStrategyBacktester, SystemValidator,
        AdaptiveParameterEvolution ve tüm strateji sınıfları.


  2. `PhoenixTradingSystem` Sınıfı:
      - Projenin ana mantığını yönetecek bu sınıfı oluştur.
      - __init__ metodu içinde, projenin tüm ana nesnelerini (portfolio, coordinator, optimizer vb.) başlat. Başlangıçta bu nesneler None olabilir
        ve ilgili metod çağrıldığında (run_live, run_backtest vb.) oluşturulabilirler.
      - Desteklenen tüm stratejileri (EnhancedMomentumStrategy, BollingerMLStrategy vb.) bir sözlük (registry) içinde tut.


  3. Komut Satırı Arayüzü (`argparse`):
      - main() fonksiyonu içinde, argparse kullanarak aşağıdaki komutları ve argümanları tanımla:
        - `live`: Canlı ticaret modu. Gerekli argümanlar: --strategy, --capital, --symbol.
        - `backtest`: Backtest modu. Gerekli argümanlar: --strategy, --start-date, --end-date, --capital, --data-file.
        - `optimize`: Optimizasyon modu. Gerekli argümanlar: --strategy, --trials, --storage, --walk-forward.
        - `validate`: Sistem sağlık kontrolü. Argüman gerektirmez.
        - `status`: Sistem durumu raporu. Argüman gerektirmez.


  4. Ana Fonksiyonların Doldurulması:
      - `run_live_trading(self, args)`:
        - StrategyCoordinator'ı ve AdaptiveParameterEvolution'ı başlat.
        - Sonsuz bir async döngü oluştur.
        - Döngü içinde, BinanceFetcher ile piyasa verisini çek.
        - StrategyCoordinator.coordinate_strategies() metodunu çağırarak stratejileri çalıştır.
        - Belirli periyotlarla (örn: her saat başı) AdaptiveParameterEvolution.monitor_strategies() metodunu çağırarak kendini optimizasyon
          mekanizmasını tetikle.
        - Acil Durum Freni: Portföy değerini sürekli kontrol et. Eğer toplam değer, başlangıç sermayesinin %15'inden fazla düşerse, tüm açık
          pozisyonları kapat ve döngüyü sonlandır.
      - `run_backtest(self, args)`:
        - MultiStrategyBacktester nesnesini oluştur.
        - argparse'dan gelen argümanları kullanarak bir BacktestConfiguration nesnesi oluştur.
        - backtester.run_backtest() metodunu çağır.
        - Dönen sonuçları, anlaşılır bir formatta konsola yazdır.
      - `run_optimization(self, args)`:
        - MasterOptimizer nesnesini oluştur.
        - argparse'dan gelen argümanları kullanarak bir OptimizationConfig nesnesi oluştur.
        - optimizer.optimize_single_strategy() veya optimizer.optimize_all_strategies() metodunu çağır.
        - Optimizasyon sonuçlarını özetleyerek konsola yazdır.
      - `validate_system(self)` ve `show_status(self)`: İlgili sınıfları (SystemValidator ve PhoenixTradingSystem'in kendi durum metodları)
        çağırarak sonuçları formatlı bir şekilde konsola yazdır.


  Sonuç Beklentisi:
  Bu görevin sonunda, main.py dosyası, projenin tüm yeteneklerini bir araya getiren, kullanıcıların komut satırından kolayca canlı ticaret, backtest
   ve optimizasyon yapabileceği, son derece güçlü ve profesyonel bir "komuta merkezi" haline gelmiş olmalıdır.