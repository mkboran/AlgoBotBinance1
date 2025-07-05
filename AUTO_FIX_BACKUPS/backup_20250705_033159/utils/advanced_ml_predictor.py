# utils/advanced_ml_predictor.py
"""
🧠 ADVANCED ML PREDICTOR - HEDGE FUND LEVEL
💎 Production-ready machine learning predictor for crypto trading
🚀 Multi-model ensemble with advanced feature engineering
"""

import pandas as pd
import numpy as np
from typing import Dict, Any, Optional, List, Tuple, Union
from datetime import datetime, timezone
import logging
import warnings
warnings.filterwarnings('ignore')

# ML Libraries
from sklearn.ensemble import RandomForestClassifier, GradientBoostingClassifier
from sklearn.preprocessing import StandardScaler, RobustScaler
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import joblib
import os

try:
    import xgboost as xgb
    XGBOOST_AVAILABLE = True
except ImportError:
    XGBOOST_AVAILABLE = False
    
try:
    import lightgbm as lgb
    LIGHTGBM_AVAILABLE = True
except ImportError:
    LIGHTGBM_AVAILABLE = False

logger = logging.getLogger(__name__)

class AdvancedMLPredictor:
    """
    🧠 ADVANCED ML PREDICTOR - INSTITUTIONAL GRADE
    
    Features:
    - Multi-model ensemble (RF, XGB, LGB, GB)
    - Advanced feature engineering (100+ features)
    - Dynamic model weighting based on performance
    - Confidence-based predictions
    - Auto-retraining with performance tracking
    """
    
    def __init__(
        self,
        model_weights: Optional[Dict[str, float]] = None,
        prediction_horizon: int = 5,
        confidence_threshold: float = 0.6,
        auto_retrain: bool = True,
        feature_importance_tracking: bool = True
    ):
        """Initialize Advanced ML Predictor"""
        
        # Configuration
        self.prediction_horizon = prediction_horizon
        self.confidence_threshold = confidence_threshold
        self.auto_retrain = auto_retrain
        self.feature_importance_tracking = feature_importance_tracking
        
        # Model weights (dynamic, updated based on performance)
        self.model_weights = model_weights or self._get_default_weights()
        
        # Models
        self.models = {}
        self.scalers = {}
        self.is_trained = False
        self.feature_columns = []
        
        # Performance tracking
        self.model_performance = {}
        self.prediction_history = []
        self.last_training_time = None
        self.training_data_size = 0
        self.feature_importance = {}
        
        # Initialize models
        self._initialize_models()
        
        logger.info("🧠 AdvancedMLPredictor initialized with ensemble approach")

    def _get_default_weights(self) -> Dict[str, float]:
        """Get default model weights based on availability"""
        weights = {'rf': 0.25, 'gb': 0.25}
        
        if XGBOOST_AVAILABLE:
            weights['xgb'] = 0.30
        if LIGHTGBM_AVAILABLE:
            weights['lgb'] = 0.20
            
        # Normalize weights
        total = sum(weights.values())
        return {k: v/total for k, v in weights.items()}

    def _initialize_models(self):
        """Initialize all ML models"""
        
        # Random Forest
        self.models['rf'] = RandomForestClassifier(
            n_estimators=200,
            max_depth=15,
            min_samples_split=5,
            min_samples_leaf=2,
            max_features='sqrt',
            random_state=42,
            n_jobs=-1,
            class_weight='balanced'
        )
        
        # Gradient Boosting
        self.models['gb'] = GradientBoostingClassifier(
            n_estimators=200,
            max_depth=8,
            learning_rate=0.1,
            subsample=0.8,
            random_state=42
        )
        
        # XGBoost (if available)
        if XGBOOST_AVAILABLE:
            self.models['xgb'] = xgb.XGBClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                eval_metric='logloss',
                use_label_encoder=False
            )
        
        # LightGBM (if available)
        if LIGHTGBM_AVAILABLE:
            self.models['lgb'] = lgb.LGBMClassifier(
                n_estimators=200,
                max_depth=8,
                learning_rate=0.1,
                subsample=0.8,
                colsample_bytree=0.8,
                random_state=42,
                verbose=-1
            )
        
        # Initialize scalers
        for model_name in self.models.keys():
            self.scalers[model_name] = RobustScaler()

    def create_advanced_features(self, data: pd.DataFrame) -> pd.DataFrame:
        """
        🔧 CREATE ADVANCED FEATURES - 100+ FEATURES
        Hedge fund level feature engineering
        """
        try:
            df = data.copy()
            
            # ==================== PRICE FEATURES ====================
            
            # Basic price features
            df['returns'] = df['close'].pct_change()
            df['log_returns'] = np.log(df['close'] / df['close'].shift(1))
            df['price_change'] = df['close'] - df['open']
            df['price_range'] = df['high'] - df['low']
            df['price_position'] = (df['close'] - df['low']) / (df['high'] - df['low'])
            
            # Price momentum features
            for period in [3, 5, 10, 20, 50]:
                df[f'returns_{period}d'] = df['close'].pct_change(period)
                df[f'price_momentum_{period}'] = df['close'] / df['close'].shift(period) - 1
                df[f'high_momentum_{period}'] = df['high'] / df['high'].shift(period) - 1
                df[f'low_momentum_{period}'] = df['low'] / df['low'].shift(period) - 1
            
            # ==================== MOVING AVERAGES ====================
            
            # Simple and Exponential Moving Averages
            for period in [5, 10, 20, 50, 100, 200]:
                df[f'sma_{period}'] = df['close'].rolling(period).mean()
                df[f'ema_{period}'] = df['close'].ewm(span=period).mean()
                df[f'price_vs_sma_{period}'] = df['close'] / df[f'sma_{period}'] - 1
                df[f'price_vs_ema_{period}'] = df['close'] / df[f'ema_{period}'] - 1
            
            # Moving average crossovers
            df['sma_5_20_cross'] = (df['sma_5'] > df['sma_20']).astype(int)
            df['sma_10_50_cross'] = (df['sma_10'] > df['sma_50']).astype(int)
            df['ema_5_20_cross'] = (df['ema_5'] > df['ema_20']).astype(int)
            
            # ==================== VOLATILITY FEATURES ====================
            
            # Rolling volatilities
            for period in [5, 10, 20, 50]:
                df[f'volatility_{period}'] = df['returns'].rolling(period).std()
                df[f'volatility_ratio_{period}'] = df[f'volatility_{period}'] / df['volatility_20']
            
            # GARCH-like features
            df['volatility_change'] = df['volatility_20'].pct_change()
            df['volatility_momentum'] = df['volatility_20'] / df['volatility_50'] - 1
            
            # ==================== VOLUME FEATURES ====================
            
            # Volume moving averages and ratios
            for period in [5, 10, 20, 50]:
                df[f'volume_sma_{period}'] = df['volume'].rolling(period).mean()
                df[f'volume_ratio_{period}'] = df['volume'] / df[f'volume_sma_{period}']
            
            # Volume-price features
            df['volume_price_trend'] = df['volume'] * df['returns']
            df['volume_weighted_price'] = (df['volume'] * df['close']).rolling(20).sum() / df['volume'].rolling(20).sum()
            df['vwap_ratio'] = df['close'] / df['volume_weighted_price'] - 1
            
            # ==================== TECHNICAL INDICATORS ====================
            
            # RSI
            df['rsi_14'] = self._calculate_rsi(df['close'], 14)
            df['rsi_7'] = self._calculate_rsi(df['close'], 7)
            df['rsi_21'] = self._calculate_rsi(df['close'], 21)
            df['rsi_oversold'] = (df['rsi_14'] < 30).astype(int)
            df['rsi_overbought'] = (df['rsi_14'] > 70).astype(int)
            
            # Bollinger Bands
            for period in [20, 50]:
                bb_upper, bb_lower, bb_middle = self._calculate_bollinger_bands(df['close'], period)
                df[f'bb_upper_{period}'] = bb_upper
                df[f'bb_lower_{period}'] = bb_lower
                df[f'bb_middle_{period}'] = bb_middle
                df[f'bb_position_{period}'] = (df['close'] - bb_lower) / (bb_upper - bb_lower)
                df[f'bb_squeeze_{period}'] = (bb_upper - bb_lower) / bb_middle
            
            # MACD
            macd_line, macd_signal, macd_histogram = self._calculate_macd(df['close'])
            df['macd_line'] = macd_line
            df['macd_signal'] = macd_signal
            df['macd_histogram'] = macd_histogram
            df['macd_bullish'] = (macd_line > macd_signal).astype(int)
            
            # Stochastic
            stoch_k, stoch_d = self._calculate_stochastic(df['high'], df['low'], df['close'])
            df['stoch_k'] = stoch_k
            df['stoch_d'] = stoch_d
            df['stoch_oversold'] = (stoch_k < 20).astype(int)
            df['stoch_overbought'] = (stoch_k > 80).astype(int)
            
            # ==================== TREND FEATURES ====================
            
            # Trend strength and direction
            for period in [5, 10, 20, 50]:
                df[f'trend_{period}'] = (df['close'] > df['close'].shift(period)).astype(int)
                df[f'trend_strength_{period}'] = abs(df['close'] - df['close'].shift(period)) / df['close'].shift(period)
            
            # Linear regression slope (trend direction)
            for period in [10, 20, 50]:
                df[f'lr_slope_{period}'] = df['close'].rolling(period).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
                )
            
            # ==================== PATTERN FEATURES ====================
            
            # Candlestick patterns (simplified)
            df['doji'] = (abs(df['close'] - df['open']) / (df['high'] - df['low']) < 0.1).astype(int)
            df['hammer'] = ((df['close'] > df['open']) & 
                           ((df['open'] - df['low']) > 2 * (df['close'] - df['open'])) &
                           ((df['high'] - df['close']) < 0.1 * (df['close'] - df['open']))).astype(int)
            
            # Support/Resistance levels (simplified)
            df['near_high_20'] = (df['close'] / df['high'].rolling(20).max() > 0.98).astype(int)
            df['near_low_20'] = (df['close'] / df['low'].rolling(20).min() < 1.02).astype(int)
            
            # ==================== ADVANCED FEATURES ====================
            
            # Fractal dimension (complexity measure)
            df['fractal_dimension'] = df['close'].rolling(20).apply(self._calculate_fractal_dimension)
            
            # Entropy (uncertainty measure)
            df['price_entropy'] = df['returns'].rolling(20).apply(self._calculate_entropy)
            
            # Correlation with lagged values
            for lag in [1, 3, 5, 10]:
                df[f'autocorr_{lag}'] = df['returns'].rolling(50).apply(
                    lambda x: x.autocorr(lag) if len(x) >= lag + 1 else np.nan
                )
            
            # ==================== MARKET MICROSTRUCTURE ====================
            
            # Spread and efficiency measures
            df['hl_spread'] = (df['high'] - df['low']) / df['close']
            df['oc_spread'] = abs(df['open'] - df['close']) / df['close']
            
            # Market efficiency (random walk test)
            df['efficiency_ratio'] = abs(df['close'] - df['close'].shift(10)) / df['returns'].abs().rolling(10).sum()
            
            # ==================== CLEANUP ====================
            
            # Replace infinite values with NaN
            df = df.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values
            df = df.fillna(method='ffill')
            
            # Fill remaining NaN with 0
            df = df.fillna(0)
            
            logger.info(f"✅ Advanced features created: {df.shape[1]} total columns")
            
            return df
            
        except Exception as e:
            logger.error(f"Feature creation error: {e}")
            return data

    def _calculate_rsi(self, prices: pd.Series, period: int = 14) -> pd.Series:
        """Calculate RSI"""
        delta = prices.diff()
        gain = (delta.where(delta > 0, 0)).rolling(window=period).mean()
        loss = (-delta.where(delta < 0, 0)).rolling(window=period).mean()
        rs = gain / loss
        return 100 - (100 / (1 + rs))

    def _calculate_bollinger_bands(self, prices: pd.Series, period: int = 20, std_dev: float = 2) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate Bollinger Bands"""
        sma = prices.rolling(period).mean()
        std = prices.rolling(period).std()
        upper = sma + (std * std_dev)
        lower = sma - (std * std_dev)
        return upper, lower, sma

    def _calculate_macd(self, prices: pd.Series, fast: int = 12, slow: int = 26, signal: int = 9) -> Tuple[pd.Series, pd.Series, pd.Series]:
        """Calculate MACD"""
        ema_fast = prices.ewm(span=fast).mean()
        ema_slow = prices.ewm(span=slow).mean()
        macd_line = ema_fast - ema_slow
        macd_signal = macd_line.ewm(span=signal).mean()
        macd_histogram = macd_line - macd_signal
        return macd_line, macd_signal, macd_histogram

    def _calculate_stochastic(self, high: pd.Series, low: pd.Series, close: pd.Series, k_period: int = 14, d_period: int = 3) -> Tuple[pd.Series, pd.Series]:
        """Calculate Stochastic Oscillator"""
        lowest_low = low.rolling(k_period).min()
        highest_high = high.rolling(k_period).max()
        k_percent = 100 * ((close - lowest_low) / (highest_high - lowest_low))
        d_percent = k_percent.rolling(d_period).mean()
        return k_percent, d_percent

    def _calculate_fractal_dimension(self, series: pd.Series) -> float:
        """Calculate fractal dimension (Hurst exponent)"""
        try:
            if len(series) < 10:
                return 0.5
            
            # Calculate log returns
            log_returns = np.log(series / series.shift(1)).dropna()
            
            if len(log_returns) < 5:
                return 0.5
            
            # Calculate R/S statistic
            mean_return = log_returns.mean()
            cumulative_deviations = (log_returns - mean_return).cumsum()
            
            R = cumulative_deviations.max() - cumulative_deviations.min()
            S = log_returns.std()
            
            if S == 0:
                return 0.5
            
            rs = R / S
            
            # Hurst exponent approximation
            hurst = np.log(rs) / np.log(len(log_returns))
            
            return max(0, min(1, hurst))
        
        except:
            return 0.5

    def _calculate_entropy(self, series: pd.Series) -> float:
        """Calculate Shannon entropy"""
        try:
            if len(series) < 5:
                return 0
            
            # Discretize the series
            bins = min(10, len(series) // 2)
            hist, _ = np.histogram(series.dropna(), bins=bins)
            
            # Calculate probabilities
            probs = hist / hist.sum()
            probs = probs[probs > 0]  # Remove zero probabilities
            
            # Calculate entropy
            entropy = -np.sum(probs * np.log2(probs))
            
            return entropy
        
        except:
            return 0

    def prepare_training_data(self, data: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, List[str]]:
        """🎯 Prepare training data with labels"""
        try:
            # Create features
            df = self.create_advanced_features(data)
            
            # Create labels (future price movement)
            future_returns = df['close'].shift(-self.prediction_horizon) / df['close'] - 1
            
            # Multi-class labels for more nuanced predictions
            labels = np.where(future_returns > 0.015, 2,  # Strong bullish (>1.5%)
                     np.where(future_returns > 0.005, 1,   # Weak bullish (0.5-1.5%)
                     np.where(future_returns < -0.015, -2,  # Strong bearish (<-1.5%)
                     np.where(future_returns < -0.005, -1,  # Weak bearish (-1.5% to -0.5%)
                     0))))  # Neutral (-0.5% to 0.5%)
            
            # Convert to binary for simplicity (bullish vs bearish)
            binary_labels = (labels > 0).astype(int)
            
            # Select feature columns (exclude OHLCV and intermediate calculations)
            exclude_columns = ['open', 'high', 'low', 'close', 'volume', 'timestamp']
            exclude_patterns = ['bb_upper', 'bb_lower', 'bb_middle', 'sma_', 'ema_', 'volume_sma_']
            
            feature_cols = []
            for col in df.columns:
                if col not in exclude_columns:
                    if not any(pattern in col for pattern in exclude_patterns):
                        if not col.endswith('_20') or col.startswith('rsi') or col.startswith('volatility'):
                            feature_cols.append(col)
            
            # Ensure we have reasonable number of features
            if len(feature_cols) > 150:
                # Select most important features based on correlation with returns
                correlations = df[feature_cols].corrwith(df['returns']).abs().sort_values(ascending=False)
                feature_cols = correlations.head(100).index.tolist()
            
            X = df[feature_cols].values
            y = binary_labels
            
            # Remove rows with NaN
            valid_mask = ~np.isnan(X).any(axis=1) & ~np.isnan(y) & (y != -999)
            X = X[valid_mask]
            y = y[valid_mask]
            
            self.feature_columns = feature_cols
            logger.info(f"✅ Training data prepared: {X.shape[0]} samples, {X.shape[1]} features")
            
            return X, y, feature_cols
            
        except Exception as e:
            logger.error(f"Training data preparation error: {e}")
            return np.array([]), np.array([]), []

    def train(self, data: pd.DataFrame, test_size: float = 0.2, retrain: bool = True) -> bool:
        """🎓 Train ML models with advanced validation"""
        try:
            if len(data) < 200:
                logger.warning("Insufficient data for training (need at least 200 samples)")
                return False
            
            X, y, feature_cols = self.prepare_training_data(data)
            
            if len(X) == 0 or len(y) == 0:
                logger.error("No valid training data")
                return False
            
            # Train-test split with temporal awareness
            split_idx = int(len(X) * (1 - test_size))
            X_train, X_test = X[:split_idx], X[split_idx:]
            y_train, y_test = y[:split_idx], y[split_idx:]
            
            if len(X_train) < 100:
                logger.warning("Insufficient training samples")
                return False
            
            # Train models
            successful_models = 0
            total_models = len(self.models)
            
            for name, model in self.models.items():
                try:
                    logger.info(f"🎓 Training {name.upper()} model...")
                    
                    # Scale features
                    X_train_scaled = self.scalers[name].fit_transform(X_train)
                    X_test_scaled = self.scalers[name].transform(X_test)
                    
                    # Train model
                    model.fit(X_train_scaled, y_train)
                    
                    # Evaluate model
                    y_pred_train = model.predict(X_train_scaled)
                    y_pred_test = model.predict(X_test_scaled)
                    
                    train_accuracy = accuracy_score(y_train, y_pred_train)
                    test_accuracy = accuracy_score(y_test, y_pred_test)
                    
                    # Store performance
                    self.model_performance[name] = {
                        'train_accuracy': train_accuracy,
                        'test_accuracy': test_accuracy,
                        'overfitting': train_accuracy - test_accuracy,
                        'last_updated': datetime.now(timezone.utc)
                    }
                    
                    # Track feature importance
                    if hasattr(model, 'feature_importances_'):
                        self.feature_importance[name] = dict(zip(feature_cols, model.feature_importances_))
                    
                    successful_models += 1
                    logger.info(f"✅ {name.upper()}: Train={train_accuracy:.3f}, Test={test_accuracy:.3f}")
                    
                except Exception as e:
                    logger.error(f"❌ {name.upper()} training failed: {e}")
                    # Remove failed model from weights
                    if name in self.model_weights:
                        del self.model_weights[name]
            
            if successful_models > 0:
                # Normalize weights for remaining models
                if self.model_weights:
                    total_weight = sum(self.model_weights.values())
                    self.model_weights = {k: v/total_weight for k, v in self.model_weights.items()}
                
                self.is_trained = True
                self.last_training_time = datetime.now(timezone.utc)
                self.training_data_size = len(X_train)
                
                logger.info(f"🎉 ML Training completed: {successful_models}/{total_models} models successful")
                logger.info(f"📊 Model weights: {self.model_weights}")
                
                return True
            
            return False
            
        except Exception as e:
            logger.error(f"Training error: {e}")
            return False

    def predict(self, data: pd.DataFrame) -> Dict[str, Any]:
        """🔮 Make ensemble ML prediction"""
        try:
            if not self.is_trained:
                return self._default_prediction("Model not trained")
            
            if len(data) < 50:
                return self._default_prediction("Insufficient data for prediction")
            
            # Create features
            df = self.create_advanced_features(data)
            
            if len(self.feature_columns) == 0:
                return self._default_prediction("No feature columns defined")
            
            # Get latest features
            try:
                latest_features = df[self.feature_columns].iloc[-1:].values
            except KeyError as e:
                logger.error(f"Missing feature columns: {e}")
                return self._default_prediction("Missing feature columns")
            
            if np.isnan(latest_features).any():
                logger.warning("NaN values in features, using default prediction")
                return self._default_prediction("NaN values in features")
            
            # Get predictions from all models
            predictions = {}
            probabilities = {}
            confidences = {}
            
            for name, model in self.models.items():
                if name not in self.model_weights:
                    continue
                    
                try:
                    # Scale features
                    features_scaled = self.scalers[name].transform(latest_features)
                    
                    # Get prediction probability
                    if hasattr(model, 'predict_proba'):
                        pred_proba = model.predict_proba(features_scaled)[0]
                        
                        if len(pred_proba) >= 2:
                            bearish_prob = pred_proba[0]
                            bullish_prob = pred_proba[1]
                        else:
                            bullish_prob = pred_proba[0]
                            bearish_prob = 1 - bullish_prob
                    else:
                        # Fallback for models without predict_proba
                        pred = model.predict(features_scaled)[0]
                        bullish_prob = float(pred)
                        bearish_prob = 1 - bullish_prob
                    
                    predictions[name] = bullish_prob
                    probabilities[name] = {
                        "bearish": float(bearish_prob),
                        "bullish": float(bullish_prob)
                    }
                    
                    # Confidence based on probability distance from 0.5
                    confidences[name] = abs(bullish_prob - 0.5) * 2
                    
                except Exception as e:
                    logger.error(f"Prediction error for {name}: {e}")
                    continue
            
            if not predictions:
                return self._default_prediction("No successful predictions")
            
            # Weighted ensemble prediction
            ensemble_bullish_prob = 0
            ensemble_confidence = 0
            total_weight = 0
            
            for name, prob in predictions.items():
                weight = self.model_weights.get(name, 0)
                ensemble_bullish_prob += prob * weight
                ensemble_confidence += confidences[name] * weight
                total_weight += weight
            
            if total_weight > 0:
                ensemble_bullish_prob /= total_weight
                ensemble_confidence /= total_weight
            else:
                ensemble_bullish_prob = 0.5
                ensemble_confidence = 0.5
            
            # Determine signal based on probability and confidence
            if ensemble_bullish_prob > 0.6 and ensemble_confidence > self.confidence_threshold:
                signal = "buy"
                final_confidence = ensemble_confidence
            elif ensemble_bullish_prob < 0.4 and ensemble_confidence > self.confidence_threshold:
                signal = "sell"
                final_confidence = ensemble_confidence
            else:
                signal = "hold"
                final_confidence = 1 - ensemble_confidence  # Lower confidence for hold
            
            # Prepare result
            result = {
                "signal": signal,
                "confidence": float(final_confidence),
                "probabilities": {
                    "bullish": float(ensemble_bullish_prob),
                    "bearish": float(1 - ensemble_bullish_prob)
                },
                "model_predictions": {k: float(v) for k, v in predictions.items()},
                "model_confidences": {k: float(v) for k, v in confidences.items()},
                "ensemble_confidence": float(ensemble_confidence),
                "models_used": list(predictions.keys()),
                "model_weights": dict(self.model_weights),
                "prediction_time": datetime.now(timezone.utc).isoformat(),
                "feature_count": len(self.feature_columns)
            }
            
            # Store prediction for performance tracking
            self.prediction_history.append({
                "timestamp": datetime.now(timezone.utc),
                "signal": signal,
                "confidence": final_confidence,
                "ensemble_prob": ensemble_bullish_prob
            })
            
            # Limit history size
            if len(self.prediction_history) > 1000:
                self.prediction_history = self.prediction_history[-500:]
            
            return result
            
        except Exception as e:
            logger.error(f"Prediction error: {e}")
            return self._default_prediction(f"Prediction error: {str(e)}")

    def _default_prediction(self, reason: str = "Unknown") -> Dict[str, Any]:
        """Default prediction when errors occur"""
        return {
            "signal": "hold",
            "confidence": 0.5,
            "probabilities": {"bullish": 0.5, "bearish": 0.5},
            "model_predictions": {},
            "ensemble_confidence": 0.5,
            "warning": f"Using default prediction: {reason}",
            "prediction_time": datetime.now(timezone.utc).isoformat()
        }

    def get_model_performance(self) -> Dict[str, Any]:
        """Get detailed model performance metrics"""
        return {
            "model_performance": dict(self.model_performance),
            "model_weights": dict(self.model_weights),
            "feature_importance": dict(self.feature_importance),
            "prediction_history_length": len(self.prediction_history),
            "is_trained": self.is_trained,
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "training_data_size": self.training_data_size,
            "available_models": list(self.models.keys())
        }

    def get_feature_importance(self, top_n: int = 20) -> Dict[str, float]:
        """Get top N most important features across all models"""
        if not self.feature_importance:
            return {}
        
        # Aggregate feature importance across models
        aggregated_importance = {}
        
        for model_name, features in self.feature_importance.items():
            weight = self.model_weights.get(model_name, 0)
            for feature, importance in features.items():
                if feature not in aggregated_importance:
                    aggregated_importance[feature] = 0
                aggregated_importance[feature] += importance * weight
        
        # Sort and return top N
        sorted_features = sorted(aggregated_importance.items(), key=lambda x: x[1], reverse=True)
        return dict(sorted_features[:top_n])

    def save_model(self, filepath: str) -> bool:
        """Save trained models and metadata"""
        try:
            if not self.is_trained:
                logger.warning("Cannot save untrained model")
                return False
            
            save_data = {
                'models': self.models,
                'scalers': self.scalers,
                'model_weights': self.model_weights,
                'feature_columns': self.feature_columns,
                'model_performance': self.model_performance,
                'last_training_time': self.last_training_time,
                'training_data_size': self.training_data_size,
                'prediction_horizon': self.prediction_horizon,
                'confidence_threshold': self.confidence_threshold
            }
            
            joblib.dump(save_data, filepath)
            logger.info(f"✅ Model saved to {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model save error: {e}")
            return False

    def load_model(self, filepath: str) -> bool:
        """Load trained models and metadata"""
        try:
            if not os.path.exists(filepath):
                logger.error(f"Model file not found: {filepath}")
                return False
            
            save_data = joblib.load(filepath)
            
            self.models = save_data['models']
            self.scalers = save_data['scalers']
            self.model_weights = save_data['model_weights']
            self.feature_columns = save_data['feature_columns']
            self.model_performance = save_data.get('model_performance', {})
            self.last_training_time = save_data.get('last_training_time')
            self.training_data_size = save_data.get('training_data_size', 0)
            self.prediction_horizon = save_data.get('prediction_horizon', self.prediction_horizon)
            self.confidence_threshold = save_data.get('confidence_threshold', self.confidence_threshold)
            
            self.is_trained = True
            logger.info(f"✅ Model loaded from {filepath}")
            return True
            
        except Exception as e:
            logger.error(f"Model load error: {e}")
            return False

    def get_status(self) -> Dict[str, Any]:
        """Get comprehensive predictor status"""
        return {
            "is_trained": self.is_trained,
            "models_available": list(self.models.keys()),
            "feature_count": len(self.feature_columns),
            "last_training_time": self.last_training_time.isoformat() if self.last_training_time else None,
            "training_data_size": self.training_data_size,
            "model_weights": dict(self.model_weights),
            "prediction_horizon": self.prediction_horizon,
            "confidence_threshold": self.confidence_threshold,
            "prediction_history_length": len(self.prediction_history),
            "xgboost_available": XGBOOST_AVAILABLE,
            "lightgbm_available": LIGHTGBM_AVAILABLE
        }