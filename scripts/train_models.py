#!/usr/bin/env python3
"""
🚀 PROJE PHOENIX - ML MODEL TRAINING PIPELINE
💎 Professional Machine Learning Model Training System

Bu script generate_training_data.py tarafından oluşturulan verilerle 
ML modellerini train eder ve deploy için hazırlar.

FEATURES:
✅ XGBoost ve LightGBM desteği
✅ Comprehensive model evaluation
✅ Time-series aware data splitting
✅ Robust hyperparameters
✅ Professional model persistence
✅ Advanced performance metrics
✅ Production-ready architecture

USAGE:
python scripts/train_models.py \
    --data-path ml_data/momentum_training_data.csv \
    --model-output-path ml_models/momentum_model.pkl \
    --model-type xgboost \
    --test-size 0.2

HEDGE FUND LEVEL MODEL TRAINING
"""

import argparse
import logging
import sys
import warnings
from pathlib import Path
from typing import Dict, Any, Tuple, Optional
import json

import pandas as pd
import numpy as np
import joblib

# Scikit-learn imports
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import RobustScaler
from sklearn.metrics import (
    accuracy_score, f1_score, roc_auc_score, 
    classification_report, confusion_matrix,
    precision_score, recall_score
)

# ML Models
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
logger = logging.getLogger("ModelTrainer")


class ModelTrainer:
    """🚀 Professional ML Model Training System"""
    
    def __init__(self, model_type: str = "xgboost", test_size: float = 0.2):
        """
        Initialize the model trainer
        
        Args:
            model_type: Type of model to train ('xgboost' or 'lightgbm')
            test_size: Proportion of data for testing
        """
        self.model_type = model_type.lower()
        self.test_size = test_size
        self.model = None
        self.scaler = None
        self.feature_columns = []
        self.model_metadata = {}
        
        # Validate model type
        if self.model_type == "xgboost" and not XGBOOST_AVAILABLE:
            raise ValueError("XGBoost is not installed. Please install it with: pip install xgboost")
        
        if self.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
            raise ValueError("LightGBM is not installed. Please install it with: pip install lightgbm")
        
        logger.info(f"🚀 ModelTrainer initialized:")
        logger.info(f"   Model type: {self.model_type}")
        logger.info(f"   Test size: {self.test_size}")

    def load_data(self, data_path: Path) -> pd.DataFrame:
        """📊 Load training data from CSV file"""
        try:
            logger.info(f"📊 Loading training data from: {data_path}")
            
            # Validate file exists
            if not data_path.exists():
                raise FileNotFoundError(f"Training data file not found: {data_path}")
            
            # Load CSV data
            df = pd.read_csv(data_path)
            
            # Validate target column exists
            if 'target' not in df.columns:
                raise ValueError("Missing 'target' column in training data")
            
            # Basic data validation
            if len(df) < 100:
                raise ValueError(f"Insufficient training data: {len(df)} samples (minimum 100 required)")
            
            # Check for missing values in target
            target_na_count = df['target'].isna().sum()
            if target_na_count > 0:
                logger.warning(f"Found {target_na_count} missing target values, removing...")
                df = df.dropna(subset=['target'])
            
            # Validate target values (should be 0 or 1 for binary classification)
            unique_targets = df['target'].unique()
            if not all(target in [0, 1] for target in unique_targets):
                raise ValueError(f"Invalid target values. Expected 0 or 1, found: {unique_targets}")
            
            logger.info(f"✅ Data loaded successfully:")
            logger.info(f"   Shape: {df.shape}")
            logger.info(f"   Target distribution: {df['target'].value_counts().to_dict()}")
            logger.info(f"   Features: {df.shape[1] - 1}")
            
            return df
            
        except Exception as e:
            logger.error(f"❌ Error loading data: {e}")
            raise

    def prepare_data(self, df: pd.DataFrame) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """🔧 Prepare features and target for training"""
        try:
            logger.info("🔧 Preparing features and target...")
            
            # Separate features from target
            target_col = 'target'
            feature_cols = [col for col in df.columns if col != target_col]
            
            X = df[feature_cols].values
            y = df[target_col].values
            
            # Store feature column names
            self.feature_columns = feature_cols
            
            # Check for infinite or extremely large values
            if np.isinf(X).any():
                logger.warning("Found infinite values in features, replacing with NaN...")
                X = np.where(np.isinf(X), np.nan, X)
            
            # Handle missing values (should be minimal after generate_training_data.py)
            nan_count = np.isnan(X).sum()
            if nan_count > 0:
                logger.warning(f"Found {nan_count} NaN values in features, filling with median...")
                from sklearn.impute import SimpleImputer
                imputer = SimpleImputer(strategy='median')
                X = imputer.fit_transform(X)
            
            # Train-test split with time-series awareness
            # Using shuffle=False to maintain temporal order
            X_train, X_test, y_train, y_test = train_test_split(
                X, y, 
                test_size=self.test_size,
                stratify=y,  # Maintain class distribution
                shuffle=False,  # Important for time series data
                random_state=42
            )
            
            logger.info(f"✅ Data prepared successfully:")
            logger.info(f"   Training samples: {len(X_train)}")
            logger.info(f"   Testing samples: {len(X_test)}")
            logger.info(f"   Features: {X_train.shape[1]}")
            logger.info(f"   Train target distribution: {np.bincount(y_train.astype(int))}")
            logger.info(f"   Test target distribution: {np.bincount(y_test.astype(int))}")
            
            return X_train, X_test, y_train, y_test
            
        except Exception as e:
            logger.error(f"❌ Error preparing data: {e}")
            raise

    def initialize_model(self) -> Any:
        """🧠 Initialize ML model with robust hyperparameters"""
        try:
            logger.info(f"🧠 Initializing {self.model_type.upper()} model...")
            
            if self.model_type == "xgboost":
                model = xgb.XGBClassifier(
                    # Core parameters
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    
                    # Regularization
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    
                    # Sampling
                    subsample=0.8,
                    colsample_bytree=0.8,
                    
                    # Performance
                    random_state=42,
                    n_jobs=-1,
                    
                    # Stability
                    eval_metric='logloss',
                    use_label_encoder=False,
                    
                    # Class balance
                    scale_pos_weight=1  # Will adjust based on class distribution
                )
                
            elif self.model_type == "lightgbm":
                model = lgb.LGBMClassifier(
                    # Core parameters
                    n_estimators=100,
                    learning_rate=0.1,
                    max_depth=5,
                    num_leaves=31,
                    
                    # Regularization
                    reg_alpha=0.1,
                    reg_lambda=0.1,
                    
                    # Sampling
                    subsample=0.8,
                    colsample_bytree=0.8,
                    
                    # Performance
                    random_state=42,
                    n_jobs=-1,
                    
                    # Stability
                    objective='binary',
                    metric='binary_logloss',
                    verbose=-1,
                    
                    # Class balance
                    is_unbalance=True
                )
                
            else:
                raise ValueError(f"Unsupported model type: {self.model_type}")
            
            # Initialize scaler
            self.scaler = RobustScaler()
            
            logger.info(f"✅ {self.model_type.upper()} model initialized with robust hyperparameters")
            
            return model
            
        except Exception as e:
            logger.error(f"❌ Error initializing model: {e}")
            raise

    def train_model(self, X_train: np.ndarray, y_train: np.ndarray) -> Any:
        """🎓 Train the ML model"""
        try:
            logger.info(f"🎓 Training {self.model_type.upper()} model...")
            
            # Initialize model
            self.model = self.initialize_model()
            
            # Adjust class weights for imbalanced data
            unique, counts = np.unique(y_train, return_counts=True)
            class_weights = dict(zip(unique, counts))
            
            if len(unique) == 2:
                # Calculate scale_pos_weight for XGBoost or class_weight for others
                neg_count = class_weights.get(0, 1)
                pos_count = class_weights.get(1, 1)
                scale_pos_weight = neg_count / pos_count
                
                if self.model_type == "xgboost":
                    self.model.set_params(scale_pos_weight=scale_pos_weight)
                
                logger.info(f"📊 Class distribution - Negative: {neg_count}, Positive: {pos_count}")
                logger.info(f"⚖️ Scale pos weight: {scale_pos_weight:.3f}")
            
            # Scale features
            logger.info("🔧 Scaling features...")
            X_train_scaled = self.scaler.fit_transform(X_train)
            
            # Train model
            logger.info("🚀 Starting model training...")
            self.model.fit(X_train_scaled, y_train)
            
            # Store metadata
            self.model_metadata = {
                'model_type': self.model_type,
                'n_features': X_train.shape[1],
                'n_samples': X_train.shape[0],
                'feature_columns': self.feature_columns,
                'training_timestamp': pd.Timestamp.now().isoformat(),
                'class_distribution': class_weights,
                'scale_pos_weight': scale_pos_weight if len(unique) == 2 else None
            }
            
            logger.info(f"✅ Model training completed successfully!")
            
            return self.model
            
        except Exception as e:
            logger.error(f"❌ Error training model: {e}")
            raise

    def evaluate_model(self, X_test: np.ndarray, y_test: np.ndarray) -> Dict[str, Any]:
        """📊 Evaluate model performance with comprehensive metrics"""
        try:
            logger.info("📊 Evaluating model performance...")
            
            # Scale test features
            X_test_scaled = self.scaler.transform(X_test)
            
            # Make predictions
            y_pred = self.model.predict(X_test_scaled)
            y_pred_proba = self.model.predict_proba(X_test_scaled)
            
            # Calculate metrics
            accuracy = accuracy_score(y_test, y_pred)
            f1 = f1_score(y_test, y_pred, average='weighted')
            precision = precision_score(y_test, y_pred, average='weighted')
            recall = recall_score(y_test, y_pred, average='weighted')
            
            # ROC AUC (handle edge cases)
            try:
                if len(np.unique(y_test)) > 1:
                    roc_auc = roc_auc_score(y_test, y_pred_proba[:, 1])
                else:
                    roc_auc = 0.5  # Default for single class
            except Exception:
                roc_auc = 0.5
            
            # Classification report
            class_report = classification_report(y_test, y_pred, output_dict=True)
            
            # Confusion matrix
            conf_matrix = confusion_matrix(y_test, y_pred)
            
            # Feature importance (if available)
            feature_importance = None
            if hasattr(self.model, 'feature_importances_'):
                feature_importance = dict(zip(
                    self.feature_columns, 
                    self.model.feature_importances_
                ))
                # Sort by importance
                feature_importance = dict(sorted(
                    feature_importance.items(), 
                    key=lambda x: x[1], 
                    reverse=True
                ))
            
            # Compile results
            results = {
                'accuracy': accuracy,
                'f1_score': f1,
                'precision': precision,
                'recall': recall,
                'roc_auc': roc_auc,
                'classification_report': class_report,
                'confusion_matrix': conf_matrix.tolist(),
                'feature_importance': feature_importance,
                'predictions_sample': {
                    'y_true': y_test[:10].tolist(),
                    'y_pred': y_pred[:10].tolist(),
                    'y_pred_proba': y_pred_proba[:10, 1].tolist()
                }
            }
            
            # Log key metrics
            logger.info(f"✅ Model evaluation completed:")
            logger.info(f"   Accuracy: {accuracy:.4f}")
            logger.info(f"   F1 Score: {f1:.4f}")
            logger.info(f"   Precision: {precision:.4f}")
            logger.info(f"   Recall: {recall:.4f}")
            logger.info(f"   ROC AUC: {roc_auc:.4f}")
            
            # Log confusion matrix
            logger.info(f"📊 Confusion Matrix:")
            logger.info(f"   True Negatives: {conf_matrix[0,0]}")
            logger.info(f"   False Positives: {conf_matrix[0,1]}")
            logger.info(f"   False Negatives: {conf_matrix[1,0]}")
            logger.info(f"   True Positives: {conf_matrix[1,1]}")
            
            # Log top features if available
            if feature_importance:
                logger.info("🔝 Top 5 Most Important Features:")
                for i, (feature, importance) in enumerate(list(feature_importance.items())[:5]):
                    logger.info(f"   {i+1}. {feature}: {importance:.4f}")
            
            return results
            
        except Exception as e:
            logger.error(f"❌ Error evaluating model: {e}")
            raise

    def save_model(self, output_path: Path, evaluation_results: Dict[str, Any]) -> None:
        """💾 Save trained model with metadata"""
        try:
            logger.info(f"💾 Saving model to: {output_path}")
            
            # Create output directory if it doesn't exist
            output_path.parent.mkdir(parents=True, exist_ok=True)
            
            # Prepare model package
            model_package = {
                'model': self.model,
                'scaler': self.scaler,
                'feature_columns': self.feature_columns,
                'model_metadata': self.model_metadata,
                'evaluation_results': evaluation_results,
                'model_type': self.model_type,
                'version': '1.0.0',
                'save_timestamp': pd.Timestamp.now().isoformat()
            }
            
            # Save using joblib for sklearn compatibility
            joblib.dump(model_package, output_path)
            
            # Save evaluation results as JSON for easy reading
            json_path = output_path.with_suffix('.json')
            evaluation_summary = {
                'model_type': self.model_type,
                'accuracy': evaluation_results['accuracy'],
                'f1_score': evaluation_results['f1_score'],
                'precision': evaluation_results['precision'],
                'recall': evaluation_results['recall'],
                'roc_auc': evaluation_results['roc_auc'],
                'feature_count': len(self.feature_columns),
                'training_samples': self.model_metadata['n_samples'],
                'save_timestamp': model_package['save_timestamp']
            }
            
            with open(json_path, 'w') as f:
                json.dump(evaluation_summary, f, indent=2)
            
            # Log save information
            file_size_mb = output_path.stat().st_size / 1024 / 1024
            logger.info(f"✅ Model saved successfully:")
            logger.info(f"   Model file: {output_path}")
            logger.info(f"   Summary file: {json_path}")
            logger.info(f"   File size: {file_size_mb:.2f} MB")
            
        except Exception as e:
            logger.error(f"❌ Error saving model: {e}")
            raise

    def train_complete_pipeline(self, data_path: Path, output_path: Path) -> Dict[str, Any]:
        """🚀 Complete model training pipeline"""
        try:
            logger.info("🚀 Starting complete model training pipeline...")
            
            # Load data
            df = self.load_data(data_path)
            
            # Prepare data
            X_train, X_test, y_train, y_test = self.prepare_data(df)
            
            # Train model
            self.train_model(X_train, y_train)
            
            # Evaluate model
            evaluation_results = self.evaluate_model(X_test, y_test)
            
            # Save model
            self.save_model(output_path, evaluation_results)
            
            logger.info("🎉 Model training pipeline completed successfully!")
            
            return evaluation_results
            
        except Exception as e:
            logger.error(f"❌ Model training pipeline failed: {e}")
            raise


def validate_dependencies():
    """🔍 Validate required dependencies"""
    missing_deps = []
    
    # Check required packages
    try:
        import pandas
    except ImportError:
        missing_deps.append("pandas")
    
    try:
        import numpy
    except ImportError:
        missing_deps.append("numpy")
    
    try:
        import sklearn
    except ImportError:
        missing_deps.append("scikit-learn")
    
    try:
        import joblib
    except ImportError:
        missing_deps.append("joblib")
    
    if missing_deps:
        logger.error(f"❌ Missing required dependencies: {missing_deps}")
        logger.error("Please install with: pip install " + " ".join(missing_deps))
        sys.exit(1)
    
    # Check optional but commonly requested packages
    if not XGBOOST_AVAILABLE:
        logger.warning("⚠️ XGBoost not available. Install with: pip install xgboost")
    
    if not LIGHTGBM_AVAILABLE:
        logger.warning("⚠️ LightGBM not available. Install with: pip install lightgbm")


def main():
    """🚀 Main function with command-line interface"""
    parser = argparse.ArgumentParser(
        description="🚀 PROJE PHOENIX - ML Model Training Pipeline",
        formatter_class=argparse.RawDescriptionHelpFormatter,
        epilog="""
Examples:
  # Train XGBoost model (default)
  python scripts/train_models.py \\
    --data-path ml_data/momentum_training_data.csv \\
    --model-output-path ml_models/momentum_model.pkl

  # Train LightGBM model with custom test size
  python scripts/train_models.py \\
    --data-path ml_data/features.csv \\
    --model-output-path ml_models/lightgbm_model.pkl \\
    --model-type lightgbm \\
    --test-size 0.3
        """
    )
    
    parser.add_argument(
        '--data-path', 
        type=str, 
        required=True,
        help='Path to input training data CSV file'
    )
    
    parser.add_argument(
        '--model-output-path', 
        type=str, 
        required=True,
        help='Path to save the trained model file'
    )
    
    parser.add_argument(
        '--model-type', 
        type=str, 
        choices=['xgboost', 'lightgbm'],
        default='xgboost',
        help='Type of model to train (default: xgboost)'
    )
    
    parser.add_argument(
        '--test-size', 
        type=float, 
        default=0.2,
        help='Proportion of dataset for test split (default: 0.2)'
    )
    
    args = parser.parse_args()
    
    # Validate dependencies
    validate_dependencies()
    
    # Convert paths to Path objects
    data_path = Path(args.data_path)
    output_path = Path(args.model_output_path)
    
    # Validate input file exists
    if not data_path.exists():
        logger.error(f"❌ Training data file not found: {data_path}")
        sys.exit(1)
    
    # Validate model type availability
    if args.model_type == "xgboost" and not XGBOOST_AVAILABLE:
        logger.error("❌ XGBoost not available. Please install with: pip install xgboost")
        sys.exit(1)
    
    if args.model_type == "lightgbm" and not LIGHTGBM_AVAILABLE:
        logger.error("❌ LightGBM not available. Please install with: pip install lightgbm")
        sys.exit(1)
    
    # Validate test size
    if not 0 < args.test_size < 1:
        logger.error(f"❌ Invalid test size: {args.test_size}. Must be between 0 and 1.")
        sys.exit(1)
    
    try:
        # Create trainer
        trainer = ModelTrainer(
            model_type=args.model_type,
            test_size=args.test_size
        )
        
        # Run complete training pipeline
        evaluation_results = trainer.train_complete_pipeline(data_path, output_path)
        
        # Final summary
        logger.info("="*60)
        logger.info("🎉 TRAINING COMPLETED SUCCESSFULLY!")
        logger.info("="*60)
        logger.info(f"📊 Final Performance Summary:")
        logger.info(f"   Model Type: {args.model_type.upper()}")
        logger.info(f"   Accuracy: {evaluation_results['accuracy']:.4f}")
        logger.info(f"   F1 Score: {evaluation_results['f1_score']:.4f}")
        logger.info(f"   ROC AUC: {evaluation_results['roc_auc']:.4f}")
        logger.info(f"   Model saved to: {output_path}")
        logger.info("="*60)
        
    except Exception as e:
        logger.error(f"❌ Training failed: {e}")
        sys.exit(1)


if __name__ == "__main__":
    main()