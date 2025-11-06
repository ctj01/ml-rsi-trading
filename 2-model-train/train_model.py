"""
ML model training script for trading signals
Author: Cristian
Description: Trains classification model (BUY/SELL/NEUTRAL) with MLflow tracking
"""

import warnings
warnings.filterwarnings('ignore')

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split, cross_val_score
from sklearn.metrics import classification_report, confusion_matrix, accuracy_score
import psycopg2
from datetime import datetime
import os

# Configuration
POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'trading_db'),
    'user': os.getenv('POSTGRES_USER', 'trading_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'trading_password_123')
}

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')
EXPERIMENT_NAME = 'RSI_Trading_Strategy'


def connect_db():
    """Connects to PostgreSQL"""
    return psycopg2.connect(**POSTGRES_CONFIG)


def extract_features():
    """
    Extracts features from database to train the model
    """
    print("Extracting features from PostgreSQL...")
    
    conn = connect_db()
    
    # Query to get weekly data with features
    query = """
        WITH features AS (
            SELECT 
                ticker,
                week_start,
                close,
                volume,
                rsi,
                signal,
                -- Feature 1: Normalized RSI
                rsi / 100.0 as rsi_normalized,
                
                -- Feature 2: Volume ratio (vs 4-week average)
                volume / AVG(volume) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start 
                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                ) as volume_ratio,
                
                -- Feature 3: Price vs MA(10 weeks)
                close / AVG(close) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) as price_vs_ma10,
                
                -- Feature 4: Volatility (std of last 4 weeks)
                STDDEV(close) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start 
                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                ) / AVG(close) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start 
                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                ) as volatility,
                
                -- Feature 5: RSI momentum (RSI change vs previous week)
                rsi - LAG(rsi, 1) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start
                ) as rsi_momentum
                
            FROM rsi_weekly
            WHERE rsi IS NOT NULL
        )
        SELECT 
            ticker,
            week_start,
            rsi_normalized,
            COALESCE(volume_ratio, 1.0) as volume_ratio,
            COALESCE(price_vs_ma10, 1.0) as price_vs_ma10,
            COALESCE(volatility, 0.02) as volatility,
            COALESCE(rsi_momentum, 0.0) as rsi_momentum,
            signal
        FROM features
        WHERE volume_ratio IS NOT NULL  -- Filter first weeks without sufficient data
        ORDER BY ticker, week_start
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    print(f"Extracted {len(df)} observations from {df['ticker'].nunique()} tickers")
    print(f"   Class distribution:")
    print(df['signal'].value_counts())
    
    return df


def prepare_data(df):
    """
    Prepares features (X) and target (y) for the model
    """
    print("\nPreparing data for training...")
    
    # Features
    feature_columns = ['rsi_normalized', 'volume_ratio', 'price_vs_ma10', 'volatility', 'rsi_momentum']
    X = df[feature_columns]
    
    # Target (convert to numbers: 0=BUY, 1=NEUTRAL, 2=SELL)
    signal_mapping = {'BUY': 0, 'NEUTRAL': 1, 'SELL': 2}
    y = df['signal'].map(signal_mapping)
    
    # Train-test split (80-20)
    X_train, X_test, y_train, y_test = train_test_split(
        X, y, test_size=0.2, random_state=42, stratify=y
    )
    
    print(f"Train: {len(X_train)} observations")
    print(f"Test:  {len(X_test)} observations")
    
    return X_train, X_test, y_train, y_test, feature_columns


def train_model(X_train, X_test, y_train, y_test, feature_columns):
    """
    Trains Random Forest with MLflow tracking
    """
    print("\nTraining Random Forest Classifier...")
    
    # Configure MLflow
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    mlflow.set_experiment(EXPERIMENT_NAME)
    
    # Hyperparameters
    params = {
        'n_estimators': 100,
        'max_depth': 10,
        'min_samples_split': 10,
        'min_samples_leaf': 5,
        'random_state': 42,
        'class_weight': 'balanced',  # Important: balance classes
        'n_jobs': -1
    }
    
    with mlflow.start_run(run_name=f"RandomForest_{datetime.now().strftime('%Y%m%d_%H%M')}"):
        
        # Log parameters
        mlflow.log_params(params)
        mlflow.log_param('train_size', len(X_train))
        mlflow.log_param('test_size', len(X_test))
        mlflow.log_param('features', ', '.join(feature_columns))
        
        # Train model
        model = RandomForestClassifier(**params)
        model.fit(X_train, y_train)
        
        # Predictions
        y_pred_train = model.predict(X_train)
        y_pred_test = model.predict(X_test)
        
        # Training metrics
        train_accuracy = accuracy_score(y_train, y_pred_train)
        test_accuracy = accuracy_score(y_test, y_pred_test)
        
        mlflow.log_metric('train_accuracy', train_accuracy)
        mlflow.log_metric('test_accuracy', test_accuracy)
        
        print(f"\nResults:")
        print(f"   Train Accuracy: {train_accuracy:.4f}")
        print(f"   Test Accuracy:  {test_accuracy:.4f}")
        
        # Cross-validation (5-fold)
        cv_scores = cross_val_score(model, X_train, y_train, cv=5, scoring='accuracy')
        mlflow.log_metric('cv_mean_accuracy', cv_scores.mean())
        mlflow.log_metric('cv_std_accuracy', cv_scores.std())
        
        print(f"   CV Accuracy:    {cv_scores.mean():.4f} (+/- {cv_scores.std():.4f})")
        
        # Classification report - only include classes present in the data
        unique_classes = sorted(set(y_train) | set(y_test))
        signal_names = {0: 'BUY', 1: 'NEUTRAL', 2: 'SELL'}
        target_names = [signal_names[c] for c in unique_classes]
        
        print(f"\nClassification Report (Test):")
        report = classification_report(
            y_test, y_pred_test, 
            labels=unique_classes,
            target_names=target_names,
            output_dict=True
        )
        print(classification_report(y_test, y_pred_test, labels=unique_classes, target_names=target_names))
        
        # Log metrics per class
        for idx, class_num in enumerate(unique_classes):
            label_name = signal_names[class_num]
            mlflow.log_metric(f'{label_name}_precision', report[target_names[idx]]['precision'])
            mlflow.log_metric(f'{label_name}_recall', report[target_names[idx]]['recall'])
            mlflow.log_metric(f'{label_name}_f1-score', report[target_names[idx]]['f1-score'])
        
        # Confusion matrix
        cm = confusion_matrix(y_test, y_pred_test, labels=unique_classes)
        print(f"\nConfusion Matrix:")
        print(f"                Predicted")
        header = "              " + "  ".join([f"{name:>8s}" for name in target_names])
        print(header)
        print(f"Actual")
        for idx, class_num in enumerate(unique_classes):
            row_label = f"       {signal_names[class_num]:<8s}"
            row_values = "  ".join([f"{cm[idx,j]:>8d}" for j in range(len(unique_classes))])
            print(f"{row_label}{row_values}")
        
        # Feature importance
        importances = pd.DataFrame({
            'feature': feature_columns,
            'importance': model.feature_importances_
        }).sort_values('importance', ascending=False)
        
        print(f"\nFeature Importance:")
        for _, row in importances.iterrows():
            mlflow.log_metric(f'importance_{row["feature"]}', row['importance'])
            print(f"   {row['feature']:20s}: {row['importance']:.4f}")
        
        # Save model
        mlflow.sklearn.log_model(
            model, 
            "model",
            registered_model_name="RSI_Trading_Model"
        )
        
        print(f"\nModel saved in MLflow")
        print(f"   Run ID: {mlflow.active_run().info.run_id}")
        
        return model, mlflow.active_run().info.run_id


def main():
    """
    Complete training pipeline
    """
    print("=" * 60)
    print("RSI Trading Model Training Pipeline")
    print("=" * 60)
    
    try:
        # 1. Extract features
        df = extract_features()
        
        # 2. Prepare data
        X_train, X_test, y_train, y_test, feature_columns = prepare_data(df)
        
        # 3. Train model
        model, run_id = train_model(X_train, X_test, y_train, y_test, feature_columns)
        
        print("\n" + "=" * 60)
        print("Training completed successfully")
        print(f"View results: {MLFLOW_TRACKING_URI}")
        print("=" * 60)
        
    except Exception as e:
        print(f"\n❌ Error: {str(e)}")
        import traceback
        traceback.print_exc()
        raise


if __name__ == '__main__':
    main()
