#!/usr/bin/env python3
"""
🔄 STRATEGY INHERITANCE GÜNCELLEMESI
💎 Mevcut Stratejileri BaseStrategy'den Inherit Etme Rehberi

Bu dosya, mevcut stratejilerin BaseStrategy'den miras alacak şekilde 
nasıl güncelleneceğini gösterir.

GÜNCELLEME ADIMLARI:
1. BaseStrategy import'u ekle
2. Class tanımını güncelle (inherit from BaseStrategy)
3. __init__ metodunu super() ile güncelle
4. analyze_market metodunu override et
5. calculate_position_size metodunu override et

📍 DOSYA: strategy_inheritance_guide.py  
📁 KONUM: strategies/
🔄 DURUM: rehber dosyası
"""

# ÖRNEK 1: ENHANCED MOMENTUM STRATEGY GÜNCELLEMESİ
# ================================================================

# ❌ ESKİ KOD (momentum_optimized.py - satır 40 civarı):
"""
class EnhancedMomentumStrategy:
    def __init__(
        self, 
        portfolio: Portfolio, 
        symbol: str = "BTC/USDT",
        # diğer parametreler...
    ):
        self.strategy_name = "EnhancedMomentum"
        self.portfolio = portfolio
        self.symbol = symbol
        # ... diğer init kodları
"""

# ✅ YENİ KOD (momentum_optimized.py güncellenmiş hali):
"""
from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, create_signal

class EnhancedMomentumStrategy(BaseStrategy):
    def __init__(
        self, 
        portfolio: Portfolio, 
        symbol: str = "BTC/USDT",
        # diğer parametreler...
        **kwargs
    ):
        # BaseStrategy'den inherit et
        super().__init__(
            portfolio=portfolio,
            symbol=symbol,
            strategy_name="EnhancedMomentum",
            **kwargs
        )
        
        # Strategy-specific initialization
        self.ema_short = kwargs.get('ema_short', 12)
        self.ema_medium = kwargs.get('ema_medium', 21)
        self.ema_long = kwargs.get('ema_long', 59)
        # ... diğer parametreler
        
        self.logger.info("✅ Enhanced Momentum Strategy initialized")
    
    async def analyze_market(self, data: pd.DataFrame) -> TradingSignal:
        '''Market analizi ve sinyal üretimi'''
        try:
            # Mevcut analiz kodunuzu buraya taşıyın
            # calculate_signals() metodunuzun içeriği
            
            # Örnek sinyal oluşturma:
            if self._should_buy(data):
                return create_signal(
                    signal_type=SignalType.BUY,
                    confidence=0.8,
                    price=self.current_price,
                    reasons=["EMA crossover", "RSI oversold", "High volume"]
                )
            elif self._should_sell(data):
                return create_signal(
                    signal_type=SignalType.SELL,
                    confidence=0.7,
                    price=self.current_price,
                    reasons=["Profit target reached", "RSI overbought"]
                )
            else:
                return create_signal(
                    signal_type=SignalType.HOLD,
                    confidence=0.5,
                    price=self.current_price,
                    reasons=["No clear signal"]
                )
                
        except Exception as e:
            self.logger.error(f"❌ Market analysis error: {e}")
            return None
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        '''Position size hesaplama'''
        try:
            # Mevcut position sizing kodunuzu buraya taşıyın
            # _calculate_position_size() metodunuzun içeriği
            
            base_size = self.portfolio.balance * (self.base_position_size_pct / 100)
            
            # Confidence-based sizing
            confidence_multiplier = signal.confidence
            adjusted_size = base_size * confidence_multiplier
            
            # Min/max limits
            adjusted_size = max(self.min_position_usdt, adjusted_size)
            adjusted_size = min(self.max_position_usdt, adjusted_size)
            
            return adjusted_size
            
        except Exception as e:
            self.logger.error(f"❌ Position size calculation error: {e}")
            return self.min_position_usdt
    
    def _should_buy(self, data: pd.DataFrame) -> bool:
        '''Buy signal logic'''
        # Mevcut buy condition kodunuzu buraya taşıyın
        # get_buy_signals() metodunuzun içeriği
        pass
    
    def _should_sell(self, data: pd.DataFrame) -> bool:
        '''Sell signal logic'''
        # Mevcut sell condition kodunuzu buraya taşıyın
        # get_sell_signals() metodunuzun içeriği
        pass
"""

# ÖRNEK 2: DİĞER STRATEJİLER İÇİN ŞABLON
# ================================================================

# BollingerRSIStrategy için:
"""
from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, create_signal

class BollingerRSIStrategy(BaseStrategy):
    def __init__(self, portfolio: Portfolio, symbol: str = "BTC/USDT", **kwargs):
        super().__init__(
            portfolio=portfolio,
            symbol=symbol,
            strategy_name="BollingerRSI",
            **kwargs
        )
        
        # Strategy-specific parameters
        self.bb_period = kwargs.get('bb_period', 20)
        self.bb_std = kwargs.get('bb_std', 2.0)
        self.rsi_period = kwargs.get('rsi_period', 14)
        
        self.logger.info("✅ Bollinger RSI Strategy initialized")
    
    async def analyze_market(self, data: pd.DataFrame) -> TradingSignal:
        # Bollinger Bands + RSI analysis
        pass
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        # Position sizing logic
        pass
"""

# RSIMLStrategy için:
"""
from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, create_signal

class RSIMLStrategy(BaseStrategy):
    def __init__(self, portfolio: Portfolio, symbol: str = "BTC/USDT", **kwargs):
        super().__init__(
            portfolio=portfolio,
            symbol=symbol,
            strategy_name="RSIML",
            **kwargs
        )
        
        # Strategy-specific parameters
        self.rsi_period = kwargs.get('rsi_period', 14)
        self.ml_model = kwargs.get('ml_model', 'xgboost')
        
        self.logger.info("✅ RSI ML Strategy initialized")
    
    async def analyze_market(self, data: pd.DataFrame) -> TradingSignal:
        # RSI + ML analysis
        pass
    
    def calculate_position_size(self, signal: TradingSignal) -> float:
        # ML-enhanced position sizing
        pass
"""

# GÜNCELLEME SÜRECİ ADIM ADIM:
# ================================================================

"""
1. IMPORT GÜNCELLEMESI:
   - Her strategy dosyasının başına ekleyin:
   from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, create_signal

2. CLASS DEFINITION GÜNCELLEMESI:
   - Değiştirin: class StrategyName:
   - Yenisi: class StrategyName(BaseStrategy):

3. __INIT__ METODU GÜNCELLEMESI:
   - super().__init__() çağrısı ekleyin
   - Ortak parametreleri BaseStrategy'ye delegeyin
   - Strategy-specific parametreleri local'de tutun

4. METOD İMZALARI GÜNCELLEMESI:
   - analyze_market(self, data: pd.DataFrame) -> TradingSignal
   - calculate_position_size(self, signal: TradingSignal) -> float

5. SİNYAL OLUŞTURMA GÜNCELLEMESI:
   - create_signal() helper fonksiyonunu kullanın
   - TradingSignal objesi döndürün

6. LOGGING GÜNCELLEMESI:
   - self.logger kullanın (BaseStrategy'den gelir)
   - Strategy-specific logger name otomatik

7. PORTFOLIO İNTERACTION:
   - BaseStrategy execute_signal() kullanır
   - Override etmeye gerek yok (çoğu durumda)

8. PERFORMANCE TRACKING:
   - BaseStrategy otomatik metrik takibi yapar
   - get_strategy_analytics() hazır gelir
"""

# MIGRATION CHECKLİST:
# ================================================================

"""
□ BaseStrategy import'u eklendi
□ Class inheritance (BaseStrategy) eklendi  
□ super().__init__() çağrısı eklendi
□ analyze_market() metodu implement edildi
□ calculate_position_size() metodu implement edildi
□ Mevcut analiz kodu taşındı
□ Signal creation güncellendiA
□ Logging self.logger'a geçirildi
□ Import errors çözüldü
□ Test edildi ve çalışıyor
"""

# TESTING YÖNTEMİ:
# ================================================================

"""
# Test kodu:
from utils.portfolio import Portfolio
from strategies.momentum_optimized import EnhancedMomentumStrategy
import pandas as pd

# Portfolio oluştur
portfolio = Portfolio(initial_balance=1000)

# Strategy oluştur  
strategy = EnhancedMomentumStrategy(
    portfolio=portfolio,
    symbol="BTC/USDT"
)

# Test data
test_data = pd.DataFrame({
    'close': [50000, 50100, 50200],
    'volume': [1000, 1100, 1200]
})

# Test çalıştır
signal = await strategy.analyze_market(test_data)
print(f"Signal: {signal}")

# Analytics kontrol et
analytics = strategy.get_strategy_analytics()
print(f"Analytics: {analytics}")
"""