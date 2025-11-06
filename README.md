# RSI Trading Strategy MLOps Pipeline

A production-ready MLOps pipeline for algorithmic trading using Relative Strength Index (RSI) indicators. This system automates data ingestion, feature engineering, model training, and prediction generation for stock market analysis.

## Table of Contents

- [Overview](#overview)
- [Architecture](#architecture)
- [Features](#features)
- [Prerequisites](#prerequisites)
- [Installation](#installation)
- [Quick Start](#quick-start)
- [Project Structure](#project-structure)
- [Components](#components)
- [Usage](#usage)
- [Configuration](#configuration)
- [Model Performance](#model-performance)
- [Contributing](#contributing)
- [License](#license)

## Overview

This project implements a complete end-to-end machine learning operations pipeline for stock trading signals based on RSI technical indicators. The system automatically:

- Fetches historical market data from Yahoo Finance
- Calculates technical indicators (RSI, moving averages, volatility)
- Trains machine learning models to predict trading signals
- Generates predictions with confidence scores
- Tracks experiments and models using MLflow
- Orchestrates workflows with Apache Airflow

## Architecture

```
┌─────────────────┐
│  Yahoo Finance  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│     Airflow     │  ← Data Pipeline Orchestration
│  ┌───────────┐  │
│  │ Fetch Data│  │
│  └─────┬─────┘  │
│  ┌─────v─────┐  │
│  │Calculate  │  │
│  │   RSI     │  │
│  └─────┬─────┘  │
│  ┌─────v─────┐  │
│  │ Generate  │  │
│  │ Signals   │  │
│  └───────────┘  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│   PostgreSQL    │  ← Data Storage
│  - Market Data  │
│  - RSI Metrics  │
│  - Predictions  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│  ML Training    │
│  ┌───────────┐  │
│  │  Extract  │  │
│  │ Features  │  │
│  └─────┬─────┘  │
│  ┌─────v─────┐  │
│  │   Train   │  │
│  │ RF Model  │  │
│  └─────┬─────┘  │
│  ┌─────v─────┐  │
│  │  Predict  │  │
│  └───────────┘  │
└────────┬────────┘
         │
         v
┌─────────────────┐
│     MLflow      │  ← Experiment Tracking
│  - Models       │
│  - Metrics      │
│  - Artifacts    │
└─────────────────┘
```

## Features

### Data Pipeline
- Automated daily market data collection from Yahoo Finance
- Historical data backfill (2020-present)
- Weekly RSI calculation with configurable periods
- Trading signal generation (BUY/SELL/NEUTRAL)
- Pipeline execution logging and monitoring

### Machine Learning
- Random Forest classification model
- Feature engineering with technical indicators:
  - Normalized RSI values
  - Price vs Moving Average ratios
  - Volume ratios
  - Volatility metrics
  - RSI momentum
- Experiment tracking with MLflow
- Model versioning and registry
- Cross-validation and performance metrics

### Infrastructure
- Containerized deployment with Docker
- Apache Airflow for workflow orchestration
- PostgreSQL for data persistence
- MLflow for model management
- Automated scheduling and retry mechanisms

## Prerequisites

- Docker Desktop (latest version)
- Docker Compose v2.0+
- 4GB+ RAM available for containers
- 10GB+ disk space
- Internet connection for market data access

## Installation

1. Clone the repository:
```bash
git clone https://github.com/ctj01/ml-rsi-trading.git
cd ml-rsi-trading
```

2. Create network for inter-service communication:
```bash
docker network create database_trading_network
```

3. Start the database service:
```bash
cd database
docker-compose up -d
```

4. Start the data pipeline:
```bash
cd ../1-data-pipeline
docker-compose up -d
```

5. Start MLflow server:
```bash
cd ../2-model-train
docker-compose up -d mlflow
```

## Quick Start

### Step 1: Initialize Data

Fetch historical market data:
```bash
cd 1-data-pipeline
docker exec airflow_scheduler airflow dags trigger fetch_market_data --conf '{"backfill": true}'
```

Wait for the DAG to complete (check status in Airflow UI at http://localhost:8080).

### Step 2: Calculate RSI

Compute weekly RSI indicators:
```bash
docker exec airflow_scheduler airflow dags trigger calculate_rsi_weekly
```

### Step 3: Train Model

Train the machine learning model:
```bash
cd ../2-model-train
docker-compose run --rm trainer
```

### Step 4: Generate Predictions

Create predictions for the latest data:
```bash
docker-compose run --rm trainer python predict.py
```

### Step 5: View Results

- **Airflow UI**: http://localhost:8080 (username: airflow, password: airflow)
- **MLflow UI**: http://localhost:5000
- **Database**: `psql -h localhost -p 5432 -U trading_user -d trading_db`

## Project Structure

```
rsi-mlops/
├── database/
│   ├── docker-compose.yml
│   └── init.sql                    # Database schema
├── 1-data-pipeline/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   └── dags/
│       ├── fetch_market_data.py    # Data ingestion DAG
│       └── calculate-rsi.py        # RSI calculation DAG
├── 2-model-train/
│   ├── docker-compose.yml
│   ├── Dockerfile
│   ├── requirements.txt
│   ├── train_model.py              # Model training script
│   └── predict.py                  # Inference script
└── README.md
```

## Components

### 1. Data Pipeline (Airflow)

**fetch_market_data DAG**
- Schedule: Monday-Friday at 18:00 UTC
- Fetches OHLCV data for: MSFT, AAPL, GOOGL, AMZN, NVDA
- Supports backfill mode for historical data
- Automatic retry on failure (3 attempts)

**calculate_rsi_weekly DAG**
- Schedule: Sundays at 20:00 UTC
- Resamples daily data to weekly frequency
- Calculates 14-period RSI using EMA method
- Generates trading signals based on thresholds:
  - BUY: RSI < 37
  - SELL: RSI > 77
  - NEUTRAL: 37 ≤ RSI ≤ 77

### 2. Database Schema

**market_data**
- Daily OHLCV data
- Primary key: (ticker, date)
- ~7,000+ records (5 years)

**rsi_weekly**
- Weekly aggregated data with RSI
- Technical indicators and signals
- ~130+ records per ticker

**model_predictions**
- ML model predictions
- Confidence scores and probabilities
- Model version tracking

**trading_signals**
- Generated trading signals
- RSI-based recommendations
- Historical signal tracking

**pipeline_runs**
- Pipeline execution logs
- Success/failure tracking
- Performance metrics

### 3. Machine Learning Model

**Algorithm**: Random Forest Classifier

**Features**:
- rsi_normalized: RSI value scaled to 0-1
- volume_ratio: Volume vs 4-week average
- price_vs_ma10: Price vs 10-week moving average
- volatility: 4-week price standard deviation
- rsi_momentum: Week-over-week RSI change

**Performance**:
- Training Accuracy: 100%
- Test Accuracy: 100%
- Cross-Validation: 100% (5-fold)

**Feature Importance**:
1. RSI Normalized: 67%
2. Price vs MA10: 22%
3. RSI Momentum: 5%
4. Volume Ratio: 3%
5. Volatility: 3%

## Usage

### Manual DAG Execution

Trigger data pipeline:
```bash
docker exec airflow_scheduler airflow dags trigger fetch_market_data
```

Trigger RSI calculation:
```bash
docker exec airflow_scheduler airflow dags trigger calculate_rsi_weekly
```

### Training New Models

Basic training:
```bash
cd 2-model-train
docker-compose run --rm trainer
```

The trained model will be:
- Logged to MLflow with all metrics
- Registered in the model registry
- Available for inference immediately

### Running Predictions

Generate predictions for all tickers:
```bash
cd 2-model-train
docker-compose run --rm trainer python predict.py
```

Query predictions from database:
```bash
docker exec trading_postgres psql -U trading_user -d trading_db -c \
  "SELECT ticker, prediction_date, prediction, 
   ROUND(confidence::numeric, 4) as confidence 
   FROM model_predictions 
   ORDER BY created_at DESC LIMIT 10;"
```

### Monitoring

View Airflow logs:
```bash
cd 1-data-pipeline
docker-compose logs -f airflow-scheduler
```

View MLflow experiments:
```bash
# Open browser to http://localhost:5000
```

Check database status:
```bash
docker exec trading_postgres psql -U trading_user -d trading_db -c \
  "SELECT 
     (SELECT COUNT(*) FROM market_data) as market_rows,
     (SELECT COUNT(*) FROM rsi_weekly) as rsi_rows,
     (SELECT COUNT(*) FROM model_predictions) as prediction_rows;"
```

## Configuration

### Environment Variables

**Database** (database/.env):
```env
POSTGRES_DB=trading_db
POSTGRES_USER=trading_user
POSTGRES_PASSWORD=trading_password_123
```

**Airflow** (1-data-pipeline/docker-compose.yml):
```yaml
AIRFLOW__CORE__EXECUTOR: LocalExecutor
AIRFLOW__CORE__LOAD_EXAMPLES: 'false'
AIRFLOW_CONN_TRADING_POSTGRES: postgresql://...
```

**MLflow** (2-model-train/docker-compose.yml):
```yaml
MLFLOW_TRACKING_URI: http://mlflow:5000
MLFLOW_BACKEND_STORE_URI: sqlite:///mlflow/mlflow.db
```

### Model Parameters

Edit `2-model-train/train_model.py`:
```python
# Random Forest parameters
n_estimators=100
max_depth=10
min_samples_split=5
random_state=42
```

### RSI Parameters

Edit `1-data-pipeline/dags/calculate-rsi.py`:
```python
RSI_PERIOD = 14              # RSI calculation period
RSI_BUY_THRESHOLD = 37       # Buy signal threshold
RSI_SELL_THRESHOLD = 77      # Sell signal threshold
```

## Model Performance

### Latest Model Metrics

**Classification Report**:
```
              precision    recall  f1-score   support

     NEUTRAL       1.00      1.00      1.00        13
        SELL       1.00      1.00      1.00        13

    accuracy                           1.00        26
   macro avg       1.00      1.00      1.00        26
weighted avg       1.00      1.00      1.00        26
```

**Confusion Matrix**:
```
                Predicted
               NEUTRAL    SELL
Actual
  NEUTRAL         13       0
  SELL             0      13
```

**Note**: Current model shows perfect accuracy due to limited data and clear feature separation. Real-world performance may vary with market conditions.

## API Reference

### Database Tables

Query market data:
```sql
SELECT * FROM market_data 
WHERE ticker = 'MSFT' 
ORDER BY date DESC 
LIMIT 10;
```

Query RSI data:
```sql
SELECT ticker, week_end, rsi, signal 
FROM rsi_weekly 
WHERE ticker = 'AAPL' 
ORDER BY week_end DESC;
```

Query predictions:
```sql
SELECT ticker, prediction_date, prediction, confidence
FROM model_predictions
ORDER BY created_at DESC;
```

### MLflow API

Load model in Python:
```python
import mlflow.sklearn

mlflow.set_tracking_uri("http://localhost:5000")
model = mlflow.sklearn.load_model("models:/RSI_Trading_Model/2")
```

## Troubleshooting

### Common Issues

**Issue**: Airflow DAGs not appearing
```bash
# Restart scheduler
cd 1-data-pipeline
docker-compose restart airflow-scheduler
```

**Issue**: Database connection refused
```bash
# Check if PostgreSQL is running
docker ps | grep postgres
# Restart database
cd database
docker-compose restart
```

**Issue**: MLflow artifacts not found
```bash
# Restart MLflow and retrain model
cd 2-model-train
docker-compose restart mlflow
docker-compose run --rm trainer
```

**Issue**: Yahoo Finance data download fails
- Verify internet connectivity
- Check if yfinance library is up to date
- Try manual trigger with smaller date range

## Development

### Running Tests

```bash
# Test database connection
docker exec trading_postgres psql -U trading_user -d trading_db -c "SELECT 1;"

# Test Airflow DAGs
docker exec airflow_scheduler airflow dags test fetch_market_data

# Test model training (dry run)
cd 2-model-train
docker-compose run --rm trainer python -c "from train_model import *; print('Imports OK')"
```

### Adding New Tickers

Edit `1-data-pipeline/dags/fetch_market_data.py`:
```python
TICKERS = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'NVDA', 'TSLA', 'META']
```

### Extending Features

Add new features in `2-model-train/train_model.py`:
```python
# In extract_features() function, add to SQL query:
-- Feature N: Your custom indicator
custom_indicator_calculation as custom_feature
```

Then add to feature list:
```python
feature_columns = [..., 'custom_feature']
```

## Performance Optimization

### Database
- Indexes already created on (ticker, date)
- Partition tables if data exceeds 10M rows
- Consider TimescaleDB for time-series optimization

### Airflow
- Increase worker count for parallel processing
- Use CeleryExecutor for distributed execution
- Enable XCom cleanup for long-running pipelines

### Model Training
- Use GridSearchCV for hyperparameter tuning
- Implement feature selection (e.g., RFE)
- Consider ensemble methods (XGBoost, LightGBM)

## Roadmap

- [ ] Add real-time streaming data pipeline
- [ ] Implement backtesting framework
- [ ] Add more technical indicators (MACD, Bollinger Bands)
- [ ] Create REST API for predictions
- [ ] Add Grafana dashboards for monitoring
- [ ] Implement automated model retraining
- [ ] Add unit and integration tests
- [ ] Support for cryptocurrency data
- [ ] Paper trading integration
- [ ] Alert system for trading signals

## Contributing

Contributions are welcome! Please follow these steps:

1. Fork the repository
2. Create a feature branch (`git checkout -b feature/amazing-feature`)
3. Commit your changes (`git commit -m 'Add amazing feature'`)
4. Push to the branch (`git push origin feature/amazing-feature`)
5. Open a Pull Request

Please ensure your code:
- Follows PEP 8 style guidelines
- Includes docstrings for functions
- Has appropriate error handling
- Includes relevant tests

## License

This project is licensed under the MIT License - see the [LICENSE](LICENSE) file for details.

## Disclaimer

This software is for educational and research purposes only. It is not intended as financial advice. Trading stocks and securities involves substantial risk of loss. Past performance does not guarantee future results. Always consult with a qualified financial advisor before making investment decisions.

## Acknowledgments

- [Apache Airflow](https://airflow.apache.org/) - Workflow orchestration
- [MLflow](https://mlflow.org/) - ML lifecycle management
- [yfinance](https://github.com/ranaroussi/yfinance) - Yahoo Finance data access
- [scikit-learn](https://scikit-learn.org/) - Machine learning library
- [PostgreSQL](https://www.postgresql.org/) - Database system

## Contact

Project Link: [https://github.com/ctj01/ml-rsi-trading.git](https://github.com/ctj01/ml-rsi-trading.git)

## Citation

If you use this project in your research, please cite:

```bibtex
@software{rsi_mlops,
  title = {RSI Trading Strategy MLOps Pipeline},
  author = {ctj01},
  year = {2025},
  url = {https://github.com/ctj01/ml-rsi-trading.git}
}
```
