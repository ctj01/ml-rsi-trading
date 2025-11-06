"""
Inference script for trained ML model predictions.
Author: ctj01
"""

import pandas as pd
import numpy as np
import mlflow
import mlflow.sklearn
from mlflow import MlflowClient
import psycopg2
import os
import json
from datetime import datetime

# Database configuration

POSTGRES_CONFIG = {
    'host': os.getenv('POSTGRES_HOST', 'localhost'),
    'port': os.getenv('POSTGRES_PORT', '5432'),
    'database': os.getenv('POSTGRES_DB', 'trading_db'),
    'user': os.getenv('POSTGRES_USER', 'trading_user'),
    'password': os.getenv('POSTGRES_PASSWORD', 'trading_password_123')
}
MODEL_NAME = 'RSI_Trading_Model'

MLFLOW_TRACKING_URI = os.getenv('MLFLOW_TRACKING_URI', 'http://localhost:5000')

def load_model(model_name):
    """
    Load the latest version of the specified MLflow model.
    """
    mlflow.set_tracking_uri(MLFLOW_TRACKING_URI)
    
    try:
        # Try to load from model registry (Production stage)
        model = mlflow.sklearn.load_model(f"models:/{model_name}/Production")
        print(f"✓ Loaded model from Production: {model_name}")
        return model
    except Exception as e:
        print(f"No Production model found: {e}")
    
    try:
        # Try to load the latest version from registry
        client = mlflow.MlflowClient()
        versions = client.search_model_versions(f"name='{model_name}'")
        if versions:
            latest_version = sorted(versions, key=lambda x: int(x.version), reverse=True)[0]
            model = mlflow.sklearn.load_model(f"models:/{model_name}/{latest_version.version}")
            print(f"✓ Loaded model version {latest_version.version}: {model_name}")
            return model
    except Exception as e:
        print(f"Could not load from registry: {e}")
    
    try:
        # Try to load from the latest run in the experiment
        client = mlflow.MlflowClient()
        experiment = client.get_experiment_by_name('RSI_Trading_Strategy')
        if experiment:
            runs = client.search_runs(
                experiment_ids=[experiment.experiment_id],
                order_by=["start_time DESC"],
                max_results=1
            )
            if runs:
                run_id = runs[0].info.run_id
                model = mlflow.sklearn.load_model(f"runs:/{run_id}/model")
                print(f"✓ Loaded model from latest run: {run_id}")
                return model
    except Exception as e:
        print(f"Could not load from latest run: {e}")
    
    print("❌ Failed to load model from all sources")
    return None

def get_latest_data(ticker='MSFT', weeks=1):
    """
    Fetch the latest weekly data and engineered features for the given ticker.
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    
    query = f"""
        WITH features AS (
            SELECT 
                ticker,
                week_start,
                close,
                volume,
                rsi,
                signal,
                rsi / 100.0 as rsi_normalized,
                volume / AVG(volume) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start 
                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                ) as volume_ratio,
                close / AVG(close) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start 
                    ROWS BETWEEN 10 PRECEDING AND 1 PRECEDING
                ) as price_vs_ma10,
                STDDEV(close) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start 
                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                ) / AVG(close) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start 
                    ROWS BETWEEN 4 PRECEDING AND 1 PRECEDING
                ) as volatility,
                rsi - LAG(rsi, 1) OVER (
                    PARTITION BY ticker 
                    ORDER BY week_start
                ) as rsi_momentum
            FROM rsi_weekly
            WHERE ticker = '{ticker}'
              AND rsi IS NOT NULL
        )
        SELECT 
            ticker,
            week_start,
            close,
            rsi,
            rsi_normalized,
            COALESCE(volume_ratio, 1.0) as volume_ratio,
            COALESCE(price_vs_ma10, 1.0) as price_vs_ma10,
            COALESCE(volatility, 0.02) as volatility,
            COALESCE(rsi_momentum, 0.0) as rsi_momentum,
            signal as actual_signal
        FROM features
        WHERE volume_ratio IS NOT NULL
        ORDER BY week_start DESC
        LIMIT {weeks}
    """
    
    df = pd.read_sql(query, conn)
    conn.close()
    
    return df

def predict_signals(model, data):
    """
    Predict trading signals using the loaded model and prepared data.
    """
    # Features
    feature_columns = ['rsi_normalized', 'volume_ratio', 'price_vs_ma10', 'volatility', 'rsi_momentum']
    X = data[feature_columns]
    
    # predictions
    predictions = model.predict(X)
    probabilities = model.predict_proba(X)
    
    # Get the classes the model was trained on
    classes = model.classes_
    signal_mapping = {0: 'BUY', 1: 'NEUTRAL', 2: 'SELL'}
    
    # Map numeric predictions to signal labels
    data['predicted_signal'] = [signal_mapping[p] for p in predictions]
    
    # Assign probabilities based on available classes
    prob_dict = {signal_mapping[cls]: probabilities[:, idx] for idx, cls in enumerate(classes)}
    
    # Set all probabilities (default to 0 if class not present)
    data['prob_BUY'] = prob_dict.get('BUY', np.zeros(len(data)))
    data['prob_NEUTRAL'] = prob_dict.get('NEUTRAL', np.zeros(len(data)))
    data['prob_SELL'] = prob_dict.get('SELL', np.zeros(len(data)))
    
    # Confidence is the max probability
    data['confidence'] = probabilities.max(axis=1)
    
    return data

def save_predictions(predictions):
    """
    Save the predictions to the database.
    """
    conn = psycopg2.connect(**POSTGRES_CONFIG)
    cursor = conn.cursor()
    
    for _, row in predictions.iterrows():
        # Features as JSON
        features = {
            'rsi': float(row['rsi']),
            'volume_ratio': float(row['volume_ratio']),
            'price_vs_ma10': float(row['price_vs_ma10']),
            'volatility': float(row['volatility']),
            'rsi_momentum': float(row['rsi_momentum'])
        }
        
        # Probabilities as JSON
        probabilities = {
            'BUY': float(row['prob_BUY']),
            'NEUTRAL': float(row['prob_NEUTRAL']),
            'SELL': float(row['prob_SELL'])
        }
        
        query = """
            INSERT INTO model_predictions 
            (ticker, prediction_date, features, prediction, probabilities, model_name, model_version)
            VALUES (%s, %s, %s, %s, %s, %s, %s)
        """
        
        cursor.execute(query, (
            row['ticker'],
            row['week_start'],
            json.dumps(features),  # Proper JSON conversion
            row['predicted_signal'],
            json.dumps(probabilities),  # Proper JSON conversion
            'RandomForest',
            '1.0'
        ))
    
    conn.commit()
    cursor.close()
    conn.close()

if __name__ == "__main__":
    # Load model
    model = load_model(MODEL_NAME)
    if model is None:
        print("Failed to load model. Exiting.")
        exit(1)
    
    # Get latest data
    ticker = 'MSFT'
    data = get_latest_data(ticker=ticker, weeks=1)
    if data.empty:
        print(f"No data found for ticker {ticker}. Exiting.")
        exit(1)
    
    # Predict signals
    predictions = predict_signals(model, data)
    print("Predictions:")
    print(predictions[['ticker', 'week_start', 'predicted_signal', 'confidence']])
    
    # Save predictions
    save_predictions(predictions)
    print("Predictions saved to database.")