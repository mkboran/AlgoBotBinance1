# strategies/rsi_strategy.py
#!/usr/bin/env python3
"""
📈 OPTIMIZE EDİLMİŞ RSI STRATEJİSİ - BASESTRATEGY MIGRATED
💎 BREAKTHROUGH: BaseStrategy Foundation ile Enhanced RSI Trading

ENHANCED WITH BASESTRATEGY FOUNDATION:
✅ Centralized logging system
✅ Standardized lifecycle management
✅ Performance tracking integration
✅ Risk management foundation
✅ Portfolio interface standardization
✅ Signal creation standardization

RSI Strategy Features:
- ALIM: RSI aşırı satım + Trend doğrulama + Hacim teyidi + Volatilite Kontrolü + AI Sinyal Düzeltmesi
- SATIM: RSI aşırı alım veya Kâr hedefine/Stop-loss'a ulaşma + ATR Stop

PRODUCTION READY - INSTITUTIONAL LEVEL IMPLEMENTATION
"""

import pandas as pd
import pandas_ta as ta
from typing import Optional, Tuple
# import numpy as np # Kullanılmıyor gibi, kaldırılabilir

# ✅ CORRECTED IMPORT - BaseStrategy from correct module
from strategies.base_strategy import BaseStrategy, TradingSignal, SignalType, create_signal

from utils.portfolio import Portfolio, Position # Position import edilmiş, iyi.
from utils.config import settings
from utils.ai_signal_provider import AiSignalProvider, AiSignal # AiSignal Enum importu eklendi


class RsiStrategy(BaseStrategy):
    """
    🎯 Optimize Edilmiş RSI Stratejisi with BaseStrategy Foundation:
    - ALIM: RSI aşırı satım + Trend doğrulama + Hacim teyidi + Volatilite Kontrolü + AI Sinyal Düzeltmesi
    - SATIM: RSI aşırı alım veya Kâr hedefine/Stop-loss'a ulaşma + ATR Stop
    """
    
    def __init__(self, portfolio: Portfolio, symbol: str = settings.SYMBOL, ai_provider: Optional[AiSignalProvider] = None, **kwargs):
        # ✅ BASESTRATEGY INHERITANCE - Initialize foundation first
        super().__init__(
            portfolio=portfolio,
            symbol=symbol,
            strategy_name="RSI",
            max_positions=kwargs.get('max_positions', 2),
            max_loss_pct=kwargs.get('max_loss_pct', 1.0), # 1% stop loss default
            min_profit_target_usdt=kwargs.get('min_profit_target_usdt', 2.0),
            base_position_size_pct=kwargs.get('base_position_size_pct', 15.0),
            min_position_usdt=kwargs.get('min_position_usdt', 100.0),
            max_position_usdt=kwargs.get('max_position_usdt', 300.0),
            **kwargs
        )
        
        # ✅ AI PROVIDER INTEGRATION (strategy-specific)
        self.ai_provider = ai_provider
        
        # === Strateji Parametreleri (Analiz Dokümanlarına Göre Revize Edilmiş) ===
        # Bu değerlerin settings (config.py) üzerinden gelmesi beklenir.
        # PDF (kullanıcı analizi) sayfa 17'deki önerilere göre:

        # RSI parametreleri
        self.rsi_period: int = settings.RSI_STRATEGY_RSI_PERIOD # Örn: 14 (standart)
        self.rsi_oversold_threshold: float = settings.RSI_STRATEGY_RSI_OVERSOLD
        # Öneri: 40.0 (trend filtresi olduğu için daha fazla sinyal almak amacıyla hafif yükseltildi)
        self.rsi_overbought_threshold: float = settings.RSI_STRATEGY_RSI_OVERBOUGHT
        # Öneri: 70.0 (erken satışı önlemek için yükseltildi)
        
        # Trend parametreleri
        self.short_ema_period: int = settings.RSI_STRATEGY_EMA_SHORT # Örn: 8
        self.long_ema_period: int = settings.RSI_STRATEGY_EMA_LONG   # Örn: 21
        # min_trend_strength (0.002) _check_trend içinde kullanılabilir, ancak skorlama daha esnek.
        self.trend_score_threshold: int = settings.RSI_STRATEGY_TREND_SCORE_THRESHOLD # Örn: 6 (0-10 arası skorda)
        
        # Hacim parametreleri
        self.volume_ma_period: int = settings.RSI_STRATEGY_VOLUME_MA_PERIOD # Örn: 20
        self.min_volume_factor: float = settings.RSI_STRATEGY_MIN_VOLUME_FACTOR
        # Öneri: 1.1 (hacim filtresi bir miktar gevşetildi)
        
        # Risk yönetimi (using BaseStrategy parameters + strategy specific)
        self.profit_target_percentage: float = settings.RSI_STRATEGY_TP_PERCENTAGE
        # Öneri: 0.02 (%2 kâr hedefi, trend yönünde işlem yaptığı için risk/ödül artırılıyor)
        self.stop_loss_percentage: float = settings.RSI_STRATEGY_SL_PERCENTAGE
        # Öneri: 0.01 (%1 zarar kesme)
        
        self.atr_stop_loss_multiplier: float = settings.RSI_STRATEGY_ATR_SL_MULTIPLIER
        # Öneri (genel): ATR tabanlı stop-loss için 2.0 çarpanı
        
        # Pozisyon yönetimi (enhanced with BaseStrategy foundation)
        self.position_size_pct: float = settings.RSI_STRATEGY_POSITION_SIZE_PCT # Örn: 0.15 (%15)
        # NOTE: Base position size comes from BaseStrategy now
        
        # Zamanlama kontrolleri
        self.last_trade_time = None
        self.min_time_between_trades_minutes = 5  # En az 5 dakika ara
        
        self.logger.info("📈 RSI Strategy - BaseStrategy Migration Completed")
        self.logger.info(f"   📊 RSI period: {self.rsi_period}, Oversold: {self.rsi_oversold_threshold}, Overbought: {self.rsi_overbought_threshold}")
        self.logger.info(f"   🎯 Profit target: {self.profit_target_percentage*100:.1f}%, Stop loss: {self.stop_loss_percentage*100:.1f}%")
        self.logger.info(f"   🤖 AI Provider: {'Enabled' if self.ai_provider else 'Disabled'}")

    async def analyze_market(self, data: pd.DataFrame) -> Optional[TradingSignal]:
        """
        🎯 RSI MARKET ANALYSIS - Enhanced with BaseStrategy foundation
        
        This method implements the complete RSI Strategy logic
        while leveraging BaseStrategy's standardized signal creation.
        """
        try:
            if len(data) < max(self.rsi_period, self.long_ema_period, self.volume_ma_period) + 5:
                return None
            
            # ✅ CALCULATE TECHNICAL INDICATORS
            indicators = self._calculate_rsi_indicators(data)
            
            # Store indicators for reference
            self.indicators = indicators
            
            # ✅ AI SIGNAL INTEGRATION
            ai_signal = None
            if self.ai_provider:
                try:
                    ai_signal = await self.ai_provider.get_signal(self.symbol, data)
                except Exception as e:
                    self.logger.warning(f"⚠️ AI signal failed: {e}")
            
            # ✅ BUY SIGNAL ANALYSIS
            buy_signal = self._analyze_rsi_buy_conditions(data, indicators, ai_signal)
            if buy_signal:
                return create_signal(
                    signal_type=SignalType.BUY,
                    confidence=buy_signal['confidence'],
                    price=self.current_price,
                    reasons=buy_signal['reasons']
                )
            
            # ✅ SELL SIGNAL ANALYSIS
            sell_signal = self._analyze_rsi_sell_conditions(data, indicators)
            if sell_signal:
                return create_signal(
                    signal_type=SignalType.SELL,
                    confidence=sell_signal['confidence'],
                    price=self.current_price,
                    reasons=sell_signal['reasons']
                )
            
            # ✅ HOLD SIGNAL (default)
            return create_signal(
                signal_type=SignalType.HOLD,
                confidence=0.5,
                price=self.current_price,
                reasons=["RSI in neutral zone", "Waiting for extreme levels"]
            )
            
        except Exception as e:
            self.logger.error(f"❌ RSI market analysis error: {e}")
            return None

    def calculate_position_size(self, signal: TradingSignal) -> float:
        """
        💰 RSI-SPECIFIC POSITION SIZE CALCULATION
        
        Enhanced for RSI extreme levels and reversal signals
        """
        try:
            # ✅ BASE SIZE from inherited parameters
            base_size = self.portfolio.balance * (self.base_position_size_pct / 100)
            
            # ✅ CONFIDENCE-BASED ADJUSTMENT
            confidence_multiplier = signal.confidence
            
            # ✅ RSI EXTREME BONUS
            rsi_bonus = 0.0
            if hasattr(signal, 'metadata') and 'rsi_value' in signal.metadata:
                rsi_value = signal.metadata['rsi_value']
                
                # More extreme RSI = larger position
                if rsi_value <= 25:  # Very oversold
                    rsi_bonus = 0.3
                elif rsi_value <= 30:  # Oversold
                    rsi_bonus = 0.2
                elif rsi_value >= 75:  # Very overbought (for sells)
                    rsi_bonus = 0.25
                elif rsi_value >= 70:  # Overbought
                    rsi_bonus = 0.15
            
            # ✅ AI SIGNAL BONUS
            ai_bonus = 0.0
            if 'AI confirmation' in signal.reasons:
                ai_bonus = 0.15
                self.logger.info("🤖 AI signal bonus applied: +15%")
            
            # ✅ CALCULATE FINAL SIZE
            total_multiplier = confidence_multiplier * (1.0 + rsi_bonus + ai_bonus)
            position_size = base_size * total_multiplier
            
            # ✅ APPLY LIMITS
            position_size = max(self.min_position_usdt, position_size)
            position_size = min(self.max_position_usdt, position_size)
            
            self.logger.info(f"💰 RSI Position size: ${position_size:.2f}")
            self.logger.info(f"   📊 RSI bonus: {rsi_bonus:.2f}, AI bonus: {ai_bonus:.2f}")
            
            return position_size
            
        except Exception as e:
            self.logger.error(f"❌ RSI position size calculation error: {e}")
            return self.min_position_usdt

    def _calculate_rsi_indicators(self, data: pd.DataFrame) -> dict:
        """Calculate RSI-specific technical indicators"""
        indicators = {}
        
        try:
            # RSI calculation
            indicators['rsi'] = ta.rsi(data['close'], length=self.rsi_period)
            
            # Trend indicators
            indicators['ema_short'] = ta.ema(data['close'], length=self.short_ema_period)
            indicators['ema_long'] = ta.ema(data['close'], length=self.long_ema_period)
            
            # Volume analysis
            indicators['volume_ma'] = data['volume'].rolling(window=self.volume_ma_period).mean()
            indicators['volume_ratio'] = data['volume'] / indicators['volume_ma']
            
            # ATR for stop loss calculations
            indicators['atr'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            
            # Volatility
            indicators['volatility'] = data['close'].rolling(window=20).std() / data['close'].rolling(window=20).mean()
            
        except Exception as e:
            self.logger.error(f"❌ RSI indicators calculation error: {e}")
        
        return indicators

    def _analyze_rsi_buy_conditions(self, data: pd.DataFrame, indicators: dict, ai_signal) -> Optional[dict]:
        """Analyze RSI buy signal conditions"""
        try:
            current_rsi = indicators['rsi'].iloc[-1]
            current_price = data['close'].iloc[-1]
            
            # Check timing constraints
            if self._is_too_soon_for_trade():
                return None
            
            # Check position limits
            if len(self.portfolio.positions) >= self.max_positions:
                return None
            
            # RSI oversold condition
            if current_rsi > self.rsi_oversold_threshold:
                return None
            
            quality_score = 0
            reasons = []
            
            # RSI extreme levels
            if current_rsi <= 25:
                quality_score += 4
                reasons.append(f"RSI extremely oversold ({current_rsi:.1f})")
            elif current_rsi <= 30:
                quality_score += 3
                reasons.append(f"RSI oversold ({current_rsi:.1f})")
            else:
                quality_score += 2
                reasons.append(f"RSI below threshold ({current_rsi:.1f})")
            
            # Trend confirmation
            ema_short = indicators['ema_short'].iloc[-1]
            ema_long = indicators['ema_long'].iloc[-1]
            
            if ema_short > ema_long:
                quality_score += 2
                reasons.append("EMA uptrend confirmation")
            elif ema_short < ema_long * 0.998:  # Small downtrend tolerance
                quality_score -= 1
                reasons.append("EMA slight downtrend")
            
            # Volume confirmation
            volume_ratio = indicators['volume_ratio'].iloc[-1]
            if volume_ratio >= self.min_volume_factor:
                quality_score += 2
                reasons.append(f"Volume confirmation ({volume_ratio:.2f}x)")
            
            # Volatility check
            volatility = indicators['volatility'].iloc[-1]
            if volatility < 0.05:  # Low volatility = more predictable
                quality_score += 1
                reasons.append(f"Low volatility environment ({volatility:.3f})")
            
            # AI signal confirmation
            if ai_signal and ai_signal.direction == AiSignal.BULLISH:
                quality_score += 2
                reasons.append(f"AI confirmation (confidence: {ai_signal.confidence:.2f})")
            elif ai_signal and ai_signal.direction == AiSignal.BEARISH:
                quality_score -= 2
                reasons.append(f"AI contradiction (bearish signal)")
            
            # Quality threshold
            if quality_score >= self.trend_score_threshold:
                confidence = min(0.95, (quality_score / 10.0))
                return {
                    'confidence': confidence,
                    'reasons': reasons,
                    'quality_score': quality_score,
                    'rsi_value': current_rsi
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ RSI buy conditions analysis error: {e}")
            return None

    def _analyze_rsi_sell_conditions(self, data: pd.DataFrame, indicators: dict) -> Optional[dict]:
        """Analyze RSI sell signal conditions"""
        try:
            if not self.portfolio.positions:
                return None
            
            current_price = data['close'].iloc[-1]
            current_rsi = indicators['rsi'].iloc[-1]
            atr = indicators['atr'].iloc[-1]
            
            reasons = []
            should_sell = False
            confidence = 0.5
            
            for position in self.portfolio.positions.values():
                if position.symbol != self.symbol:
                    continue
                
                # Calculate profit/loss
                profit_pct = ((current_price - position.entry_price) / position.entry_price) * 100
                profit_usdt = (current_price - position.entry_price) * position.quantity
                
                # Time-based analysis
                from datetime import datetime, timezone
                hold_time_minutes = (datetime.now(timezone.utc) - position.entry_time).total_seconds() / 60
                
                # RSI overbought conditions
                if current_rsi >= self.rsi_overbought_threshold:
                    should_sell = True
                    confidence = 0.8
                    reasons.append(f"RSI overbought ({current_rsi:.1f})")
                
                # Profit target
                if profit_pct >= (self.profit_target_percentage * 100):
                    should_sell = True
                    confidence = 0.9
                    reasons.append(f"Profit target reached ({profit_pct:.1f}%)")
                
                # Stop loss
                if profit_pct <= -(self.stop_loss_percentage * 100):
                    should_sell = True
                    confidence = 0.95
                    reasons.append(f"Stop loss triggered ({profit_pct:.1f}%)")
                
                # ATR-based stop loss
                atr_stop_price = position.entry_price - (atr * self.atr_stop_loss_multiplier)
                if current_price <= atr_stop_price:
                    should_sell = True
                    confidence = 0.9
                    reasons.append(f"ATR stop loss triggered")
                
                # Time-based exit (if holding too long without profit)
                if hold_time_minutes > 60 and profit_pct < 0.5:  # 1 hour with minimal profit
                    should_sell = True
                    confidence = 0.7
                    reasons.append(f"Time-based exit ({hold_time_minutes:.0f}min)")
            
            if should_sell:
                return {
                    'confidence': confidence,
                    'reasons': reasons,
                    'rsi_value': current_rsi
                }
            
            return None
            
        except Exception as e:
            self.logger.error(f"❌ RSI sell conditions analysis error: {e}")
            return None

    def _is_too_soon_for_trade(self) -> bool:
        """Check if enough time has passed since last trade"""
        if self.last_trade_time is None:
            return False
        
        from datetime import datetime, timezone, timedelta
        time_since_last = datetime.now(timezone.utc) - self.last_trade_time
        return time_since_last < timedelta(minutes=self.min_time_between_trades_minutes)

    # ✅ LEGACY METHODS (preserved for backward compatibility but using BaseStrategy foundation)
    
    def _check_rsi_oversold(self, df: pd.DataFrame) -> bool:
        """Legacy method - now uses analyze_market"""
        # This method is kept for backward compatibility
        # Real logic is now in analyze_market method
        current_rsi = self.indicators.get('rsi', pd.Series([50])).iloc[-1] if hasattr(self, 'indicators') else 50
        return current_rsi <= self.rsi_oversold_threshold

    def _check_trend(self, df: pd.DataFrame) -> Tuple[int, str]:
        """Legacy trend check method"""
        if not hasattr(self, 'indicators'):
            return 5, "No indicators available"
        
        ema_short = self.indicators.get('ema_short', pd.Series([0])).iloc[-1]
        ema_long = self.indicators.get('ema_long', pd.Series([0])).iloc[-1]
        
        if ema_short > ema_long:
            return 8, "Strong uptrend"
        elif ema_short > ema_long * 0.998:
            return 6, "Mild uptrend"
        else:
            return 3, "Downtrend"

    def _check_volume(self, df: pd.DataFrame) -> Tuple[bool, str]:
        """Legacy volume check method"""
        if not hasattr(self, 'indicators'):
            return False, "No indicators available"
        
        volume_ratio = self.indicators.get('volume_ratio', pd.Series([1])).iloc[-1]
        return volume_ratio >= self.min_volume_factor, f"Volume ratio: {volume_ratio:.2f}"

    def get_strategy_analytics(self) -> dict:
        """
        📊 Enhanced RSI strategy analytics with BaseStrategy integration
        """
        try:
            # Get base analytics from BaseStrategy
            base_analytics = super().get_strategy_analytics()
            
            # Add RSI-specific analytics
            rsi_analytics = {
                "rsi_specific": {
                    "parameters": {
                        "rsi_period": self.rsi_period,
                        "oversold_threshold": self.rsi_oversold_threshold,
                        "overbought_threshold": self.rsi_overbought_threshold,
                        "profit_target_pct": self.profit_target_percentage * 100,
                        "stop_loss_pct": self.stop_loss_percentage * 100
                    },
                    "current_levels": {
                        "current_rsi": self.indicators.get('rsi', pd.Series([50])).iloc[-1] if hasattr(self, 'indicators') and 'rsi' in self.indicators else None,
                        "is_oversold": self.indicators.get('rsi', pd.Series([50])).iloc[-1] <= self.rsi_oversold_threshold if hasattr(self, 'indicators') and 'rsi' in self.indicators else False,
                        "is_overbought": self.indicators.get('rsi', pd.Series([50])).iloc[-1] >= self.rsi_overbought_threshold if hasattr(self, 'indicators') and 'rsi' in self.indicators else False
                    },
                    "ai_integration": {
                        "ai_provider_enabled": self.ai_provider is not None,
                        "ai_provider_type": type(self.ai_provider).__name__ if self.ai_provider else None
                    }
                }
            }
            
            # Merge analytics
            base_analytics.update(rsi_analytics)
            return base_analytics
            
        except Exception as e:
            self.logger.error(f"❌ RSI strategy analytics error: {e}")
            return {"error": str(e)}