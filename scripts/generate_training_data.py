#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - ML TRAINING DATA GENERATOR
💎 Ultra Gelişmiş Feature Engineering ve Target Labeling Sistemi

Bu script raw historical verilerden ML modelleri için feature-rich, 
labeled datasets oluşturur. utils/advanced_ml_predictor.py ile tam uyumludur.

FEATURES:
✅ Command-line interface with argparse
✅ Exact feature matching with advanced_ml_predictor.py
✅ Comprehensive technical indicators
✅ Advanced pattern recognition
✅ Market microstructure features
✅ Time-based features
✅ Target labeling with configurable horizons
✅ Clean, ready-to-train datasets

USAGE:
python scripts/generate_training_data.py \
    --data-path historical_data/BTCUSDT_1h_2023_2024.csv \
    --output-path ml_data/momentum_training_data.csv \
    --target-horizon 24 \
    --target-threshold 0.02

HEDGE FUND LEVEL FEATURE ENGINEERING
"""

import argparse
import logging
import sys
from pathlib import Path
from typing import Tuple, List, Optional
import warnings

import pandas as pd
import numpy as np
import pandas_ta as ta
from scipy import stats
from scipy.signal import find_peaks, argrelextrema

# Suppress warnings for clean output
warnings.filterwarnings('ignore')

# Setup logging
logging.basicConfig(
    level=logging.INFO,
    format='%(asctime)s [%(levelname)s] %(message)s',
    handlers=[
        logging.StreamHandler(sys.stdout)
    ]
)
logger = logging.getLogger("TrainingDataGenerator")


class TrainingDataGenerator:
    """🚀 Ultra Advanced Training Data Generator"""
    
    def __init__(self, target_horizon: int = 24, target_threshold: float = 0.02):
        """
        Initialize the training data generator
        
        Args:
            target_horizon: Number of periods to look ahead for target
            target_threshold: Percentage change threshold for UP/DOWN classification
        """
        self.target_horizon = target_horizon
        self.target_threshold = target_threshold
        
        logger.info(f"🎯 Training Data Generator initialized:")
        logger.info(f"   Target horizon: {target_horizon} periods")
        logger.info(f"   Target threshold: ±{target_threshold*100:.1f}%")

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """📊 Load historical data from CSV file"""
        try:
            logger.info(f"📊 Loading data from: {data_path}")
            
            # Load CSV data
            df = pd.read_csv(data_path)
            
            # Validate required columns
            required_columns = ['open', 'high', 'low', 'close', 'volume']
            missing_columns = [col for col in required_columns if col not in df.columns]
            
            if missing_columns:
                raise ValueError(f"Missing required columns: {missing_columns}")
            
            # Convert timestamp column if exists
            if 'timestamp' in df.columns:
                df['timestamp'] = pd.to_datetime(df['timestamp'])
                df.set_index('timestamp', inplace=True)
            else:
                # Create index if no timestamp
                df.index = pd.to_datetime(df.index)
            
            # Ensure numeric data types
            for col in required_columns:
                df[col] = pd.to_numeric(df[col], errors='coerce')
            
            # Remove any invalid data
            df = df.dropna(subset=required_columns)
            
            logger.info(f"✅ Data loaded successfully:")
            logger.info(f"   Shape: {df.shape}")
            logger.info(f"   Date range: {df.index.min()} to {df.index.max()}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def calculate_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """🧠 Calculate comprehensive features matching advanced_ml_predictor.py"""
        try:
            logger.info("🧠 Calculating comprehensive features...")
            
            # Create a copy to avoid modifying original data
            data = df.copy()
            
            # ==================== BASIC FEATURES ====================
            
            # Returns and price changes
            data['returns'] = data['close'].pct_change()
            data['log_returns'] = np.log(data['close'] / data['close'].shift(1))
            data['price_change'] = data['close'] - data['close'].shift(1)
            data['price_change_pct'] = data['price_change'] / data['close'].shift(1)
            
            # High-low spread and body size
            data['hl_range'] = data['high'] - data['low']
            data['hl_range_pct'] = data['hl_range'] / data['close']
            data['body_size'] = abs(data['close'] - data['open'])
            data['body_size_pct'] = data['body_size'] / data['close']
            data['upper_shadow'] = data['high'] - np.maximum(data['close'], data['open'])
            data['lower_shadow'] = np.minimum(data['close'], data['open']) - data['low']
            
            # Volume features
            data['volume_change'] = data['volume'].pct_change()
            data['volume_ratio'] = data['volume'] / data['volume'].rolling(20).mean()
            data['price_volume'] = data['close'] * data['volume']
            
            # Volume-weighted average price (VWAP)
            data['vwap'] = (data['close'] * data['volume']).rolling(20).sum() / data['volume'].rolling(20).sum()
            data['vwap_ratio'] = data['close'] / data['vwap'] - 1
            
            # ==================== TECHNICAL INDICATORS ====================
            
            # RSI with multiple periods
            data['rsi_14'] = ta.rsi(data['close'], length=14)
            data['rsi_7'] = ta.rsi(data['close'], length=7)
            data['rsi_21'] = ta.rsi(data['close'], length=21)
            data['rsi_oversold'] = (data['rsi_14'] < 30).astype(int)
            data['rsi_overbought'] = (data['rsi_14'] > 70).astype(int)
            
            # Bollinger Bands with multiple periods
            for period in [20, 50]:
                bb = ta.bbands(data['close'], length=period, std=2)
                data[f'bb_upper_{period}'] = bb[f'BBU_{period}_2.0']
                data[f'bb_lower_{period}'] = bb[f'BBL_{period}_2.0']
                data[f'bb_middle_{period}'] = bb[f'BBM_{period}_2.0']
                data[f'bb_position_{period}'] = (data['close'] - data[f'bb_lower_{period}']) / (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}'])
                data[f'bb_squeeze_{period}'] = (data[f'bb_upper_{period}'] - data[f'bb_lower_{period}']) / data[f'bb_middle_{period}']
            
            # MACD
            macd = ta.macd(data['close'], fast=12, slow=26, signal=9)
            data['macd_line'] = macd['MACD_12_26_9']
            data['macd_signal'] = macd['MACDs_12_26_9'] 
            data['macd_histogram'] = macd['MACDh_12_26_9']
            data['macd_bullish'] = (data['macd_line'] > data['macd_signal']).astype(int)
            
            # Stochastic Oscillator
            stoch = ta.stoch(data['high'], data['low'], data['close'], k=14, d=3, smooth_k=3)
            data['stoch_k'] = stoch['STOCHk_14_3_3']
            data['stoch_d'] = stoch['STOCHd_14_3_3']
            data['stoch_oversold'] = (data['stoch_k'] < 20).astype(int)
            data['stoch_overbought'] = (data['stoch_k'] > 80).astype(int)
            
            # ATR (Average True Range)
            data['atr_14'] = ta.atr(data['high'], data['low'], data['close'], length=14)
            data['atr_ratio'] = data['atr_14'] / data['close']
            
            # ADX (Average Directional Index)
            adx = ta.adx(data['high'], data['low'], data['close'], length=14)
            data['adx'] = adx['ADX_14']
            data['di_plus'] = adx['DMP_14']
            data['di_minus'] = adx['DMN_14']
            
            # Volatility (20-period rolling standard deviation)
            data['volatility_20'] = data['returns'].rolling(20).std()
            data['volatility_50'] = data['returns'].rolling(50).std()
            
            # Rate of Change (ROC)
            data['roc_10'] = ta.roc(data['close'], length=10)
            data['roc_20'] = ta.roc(data['close'], length=20)
            
            # ==================== LAGGED RETURNS ====================
            
            for lag in [1, 2, 3, 5]:
                data[f'returns_lag_{lag}'] = data['returns'].shift(lag)
                data[f'close_lag_{lag}'] = data['close'].shift(lag)
                
            # ==================== TIME FEATURES ====================
            
            # Hour and day features
            data['hour_of_day'] = data.index.hour
            data['day_of_week'] = data.index.dayofweek
            data['is_weekend'] = (data.index.dayofweek >= 5).astype(int)
            
            # Cyclical encoding for time features
            data['hour_sin'] = np.sin(2 * np.pi * data['hour_of_day'] / 24)
            data['hour_cos'] = np.cos(2 * np.pi * data['hour_of_day'] / 24)
            data['day_sin'] = np.sin(2 * np.pi * data['day_of_week'] / 7)
            data['day_cos'] = np.cos(2 * np.pi * data['day_of_week'] / 7)
            
            # ==================== TREND FEATURES ====================
            
            # Trend strength and direction for multiple periods
            for period in [5, 10, 20, 50]:
                data[f'trend_{period}'] = (data['close'] > data['close'].shift(period)).astype(int)
                data[f'trend_strength_{period}'] = abs(data['close'] - data['close'].shift(period)) / data['close'].shift(period)
            
            # Linear regression slope (trend direction)
            for period in [10, 20, 50]:
                data[f'lr_slope_{period}'] = data['close'].rolling(period).apply(
                    lambda x: np.polyfit(range(len(x)), x, 1)[0] if len(x) == period else np.nan
                )
            
            # ==================== PATTERN FEATURES ====================
            
            # Candlestick patterns (simplified)
            data['doji'] = (abs(data['close'] - data['open']) / (data['high'] - data['low']) < 0.1).astype(int)
            data['hammer'] = ((data['close'] > data['open']) & 
                           ((data['open'] - data['low']) > 2 * (data['close'] - data['open'])) &
                           ((data['high'] - data['close']) < 0.1 * (data['close'] - data['open']))).astype(int)
            
            # Support/Resistance levels
            data['near_high_20'] = (data['close'] / data['high'].rolling(20).max() > 0.98).astype(int)
            data['near_low_20'] = (data['close'] / data['low'].rolling(20).min() < 1.02).astype(int)
            
            # ==================== ADVANCED FEATURES ====================
            
            # Fractal dimension (complexity measure)
            data['fractal_dimension'] = data['close'].rolling(20).apply(self._calculate_fractal_dimension)
            
            # Entropy (uncertainty measure)
            data['price_entropy'] = data['returns'].rolling(20).apply(self._calculate_entropy)
            
            # Autocorrelation with lagged values
            for lag in [1, 3, 5, 10]:
                data[f'autocorr_{lag}'] = data['returns'].rolling(50).apply(
                    lambda x: x.autocorr(lag) if len(x) >= lag + 1 else np.nan
                )
            
            # ==================== MARKET MICROSTRUCTURE ====================
            
            # Spread and efficiency measures
            data['hl_spread'] = (data['high'] - data['low']) / data['close']
            data['oc_spread'] = abs(data['open'] - data['close']) / data['close']
            
            # Market efficiency (random walk test)
            data['efficiency_ratio'] = abs(data['close'] - data['close'].shift(10)) / data['returns'].abs().rolling(10).sum()
            
            # ==================== CLEANUP ====================
            
            # Replace infinite values with NaN
            data = data.replace([np.inf, -np.inf], np.nan)
            
            # Forward fill NaN values
            data = data.fillna(method='ffill')
            
            # Fill remaining NaN with 0
            data = data.fillna(0)
            
            logger.info(f"✅ Features calculated successfully:")
            logger.info(f"   Total columns: {data.shape[1]}")
            logger.info(f"   Feature columns: {data.shape[1] - 5}")  # Excluding OHLCV
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error calculating features: {e}")
            raise

    def create_labels(self, data: pd.DataFrame) -> pd.DataFrame:
        """🎯 Create target labels for classification"""
        try:
            logger.info(f"🎯 Creating target labels with {self.target_horizon} period horizon...")
            
            # Calculate future price change
            future_price = data['close'].shift(-self.target_horizon)
            future_pct_change = (future_price - data['close']) / data['close']
            
            # Create binary labels based on threshold
            conditions = [
                future_pct_change > self.target_threshold,   # UP (1)
                future_pct_change < -self.target_threshold   # DOWN (0)
            ]
            choices = [1, 0]
            
            # Create target column (NaN for HOLD/UNCERTAIN)
            data['target'] = np.select(conditions, choices, default=np.nan)
            
            # Statistics
            target_counts = data['target'].value_counts(dropna=False)
            total_rows = len(data)
            
            logger.info(f"✅ Target labels created:")
            logger.info(f"   UP (1): {target_counts.get(1, 0)} ({target_counts.get(1, 0)/total_rows*100:.1f}%)")
            logger.info(f"   DOWN (0): {target_counts.get(0, 0)} ({target_counts.get(0, 0)/total_rows*100:.1f}%)")
            logger.info(f"   UNCERTAIN (NaN): {target_counts.get(np.nan, 0)} ({target_counts.get(np.nan, 0)/total_rows*100:.1f}%)")
            
            return data
            
        except Exception as e:
            logger.error(f"❌ Error creating labels: {e}")
            raise

    def clean_data(self, data: pd.DataFrame) -> pd.DataFrame:
        """🧹 Clean data and prepare final dataset"""
        try:
            logger.info("🧹 Cleaning data and preparing final dataset...")
            
            # Remove rows with NaN in target column
            data_clean = data.dropna(subset=['target'])
            
            # Select feature columns (exclude OHLCV and intermediate calculations)
            exclude_columns = ['open', 'high', 'low', 'close', 'volume']
            exclude_patterns = ['bb_upper', 'bb_lower', 'bb_middle', 'sma_', 'ema_', 'volume_sma_']
            
            feature_cols = []
            for col in data_clean.columns:
                if col not in exclude_columns and col != 'target':
                    if not any(pattern in col for pattern in exclude_patterns):
                        feature_cols.append(col)
            
            # Keep only feature columns and target
            final_columns = feature_cols + ['target']
            data_final = data_clean[final_columns]
            
            # Remove any remaining NaN values
            data_final = data_final.dropna()
            
            logger.info(f"✅ Data cleaned successfully:")
            logger.info(f"   Final shape: {data_final.shape}")
            logger.info(f"   Feature columns: {len(feature_cols)}")
            logger.info(f"   Clean samples: {len(data_final)}")
            
            return data_final
            
        except Exception as e:
            logger.error(f"❌ Error cleaning data: {e}")
            raise

    def save_data(self, data: pd.DataFrame, output_path: Path) -> None:
        """💾 Save processed data to CSV"""
        try:
            logger.info(f"💾 Saving processed data to: {output_path}")
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Save to CSV
            data.to_csv(output_path, index=True)
            
            logger.info(f"✅ Data saved successfully:")
            logger.info(f"   File: {output_path}")
            logger.info(f"   Size: {output_path.stat().st_size / 1024 / 1024:.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Error saving data: {e}")
            raise

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

    def generate_training_data(self, data_path: Path, output_path: Path) -> None:
        """🚀 Main function to generate complete training dataset"""
        try:
            logger.info("🚀 Starting training data generation process...")
            
            # Load data
            data = self.load_data(data_path)
            
            # Calculate features
            data_with_features = self.calculate_features(data)
            
            # Create labels
            data_with_labels = self.create_labels(data_with_features)
            
            # Clean data
            final_data = self.clean_data(data_with_labels)
            
            # Save data
            self.save_data(final_data, output_path)
            
            logger.info("🎉 Training data generation completed successfully!")
            
        except Exception as e:
            logger.error(f"❌ Training data generation failed: {e}")
            raise


def main():
    """🚀 Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="🚀 PROJE PHOENIX - ML Training Data Generator",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  python scripts/generate_training_data.py \\
    --data-path historical_data/BTCUSDT_1h_2023_2024.csv \\
    --output-path ml_data/momentum_training_data.csv

  python scripts/generate_training_data.py \\
    --data-path data/binance_btc_1h.csv \\
    --output-path training_data/btc_features.csv \\
    --target-horizon 48 \\
    --target-threshold 0.03
        """
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        required=True,
        help='Path to input historical data CSV file'
    )
    
    parser.add_argument(
        '--output-path', 
        type=str, 
        required=True,
        help='Path to save generated labeled CSV file'
    )
    
    parser.add_argument(
        '--target-horizon', 
        type=int, 
        default=24,
        help='Number of periods to look ahead for target (default: 24)'
    )
    
    parser.add_argument(
        '--target-threshold', 
        type=float, 
        default=0.02,
        help='Percentage change threshold for UP/DOWN signals (default: 0.02)'
    )
    
    args = parser.parse_args()
    
    # Convert paths to Path objects
    data_path = Path(args.data_path)
    output_path = Path(args.output_path)
    
    # Validate input file exists
    if not data_path.exists():
        logger.error(f"❌ Input data file not found: {data_path}")
        sys.exit(1)
    
    # Create generator
    generator = TrainingDataGenerator(
        target_horizon=args.target_horizon,
        target_threshold=args.target_threshold
    )
    
    try:
        # Generate training data
        generator.generate_training_data(data_path, output_path)
        
        logger.info("🎉 All done! Training data generated successfully.")
        
    except Exception as e:
        logger.error(f"❌ Process failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()