-- Trading Database Schema

-- Table: market_data (daily data)
CREATE TABLE IF NOT EXISTS market_data (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    date DATE NOT NULL,
    open DECIMAL(10, 2) NOT NULL,
    high DECIMAL(10, 2) NOT NULL,
    low DECIMAL(10, 2) NOT NULL,
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, date)
);

-- Indexes for fast queries
CREATE INDEX idx_ticker_date ON market_data(ticker, date DESC);
CREATE INDEX idx_date ON market_data(date DESC);

-- Table: rsi_weekly (calculated weekly RSI)
CREATE TABLE IF NOT EXISTS rsi_weekly (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    week_start DATE NOT NULL,
    week_end DATE NOT NULL,
    open DECIMAL(10, 2) NOT NULL,
    high DECIMAL(10, 2) NOT NULL,
    low DECIMAL(10, 2) NOT NULL,
    close DECIMAL(10, 2) NOT NULL,
    volume BIGINT NOT NULL,
    rsi DECIMAL(5, 2),
    signal VARCHAR(10),  -- 'BUY', 'SELL', 'NEUTRAL'
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    updated_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP,
    UNIQUE(ticker, week_start)
);

CREATE INDEX idx_rsi_ticker_date ON rsi_weekly(ticker, week_start DESC);
CREATE INDEX idx_rsi_signal ON rsi_weekly(signal);

-- Table: trading_signals (generated signals)
CREATE TABLE IF NOT EXISTS trading_signals (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    signal_date DATE NOT NULL,
    signal VARCHAR(10) NOT NULL,  -- 'BUY', 'SELL', 'NEUTRAL'
    rsi DECIMAL(5, 2) NOT NULL,
    price DECIMAL(10, 2) NOT NULL,
    confidence DECIMAL(5, 4),  -- ML model confidence (0-1)
    model_version VARCHAR(50),
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_signals_ticker_date ON trading_signals(ticker, signal_date DESC);

-- Table: model_predictions (ML model predictions)
CREATE TABLE IF NOT EXISTS model_predictions (
    id SERIAL PRIMARY KEY,
    ticker VARCHAR(10) NOT NULL,
    prediction_date TIMESTAMP NOT NULL,
    features JSONB NOT NULL,  -- Features used: {rsi, volume_ratio, etc}
    prediction VARCHAR(10) NOT NULL,  -- 'BUY', 'SELL', 'HOLD'
    probabilities JSONB NOT NULL,  -- {buy: 0.7, sell: 0.1, hold: 0.2}
    model_name VARCHAR(100) NOT NULL,
    model_version VARCHAR(50) NOT NULL,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_predictions_ticker_date ON model_predictions(ticker, prediction_date DESC);

-- Table: pipeline_runs (pipeline execution tracking)
CREATE TABLE IF NOT EXISTS pipeline_runs (
    id SERIAL PRIMARY KEY,
    pipeline_name VARCHAR(100) NOT NULL,
    run_date TIMESTAMP NOT NULL,
    status VARCHAR(20) NOT NULL,  -- 'SUCCESS', 'FAILED', 'RUNNING'
    records_processed INTEGER,
    error_message TEXT,
    execution_time_seconds INTEGER,
    created_at TIMESTAMP DEFAULT CURRENT_TIMESTAMP
);

CREATE INDEX idx_pipeline_runs_date ON pipeline_runs(run_date DESC);

-- View: latest_signals (latest signals per ticker)
CREATE OR REPLACE VIEW latest_signals AS
SELECT DISTINCT ON (ticker)
    ticker,
    signal,
    rsi,
    price,
    confidence,
    signal_date
FROM trading_signals
ORDER BY ticker, signal_date DESC;

-- View: weekly_performance (weekly performance)
CREATE OR REPLACE VIEW weekly_performance AS
SELECT
    ticker,
    week_start,
    close,
    rsi,
    signal,
    LAG(close) OVER (PARTITION BY ticker ORDER BY week_start) as prev_close,
    (close - LAG(close) OVER (PARTITION BY ticker ORDER BY week_start)) / 
        LAG(close) OVER (PARTITION BY ticker ORDER BY week_start) * 100 as week_return_pct
FROM rsi_weekly
ORDER BY ticker, week_start DESC;

-- Function: automatically update updated_at
CREATE OR REPLACE FUNCTION update_updated_at_column()
RETURNS TRIGGER AS $$
BEGIN
    NEW.updated_at = CURRENT_TIMESTAMP;
    RETURN NEW;
END;
$$ LANGUAGE plpgsql;

CREATE TRIGGER update_rsi_weekly_updated_at
BEFORE UPDATE ON rsi_weekly
FOR EACH ROW
EXECUTE FUNCTION update_updated_at_column();

-- Initial sample data (optional)
-- INSERT INTO market_data (ticker, date, open, high, low, close, volume)
-- VALUES ('MSFT', '2025-11-01', 420.00, 425.00, 418.00, 423.00, 25000000);

-- Grants (adjust according to user)
GRANT ALL PRIVILEGES ON ALL TABLES IN SCHEMA public TO trading_user;
GRANT ALL PRIVILEGES ON ALL SEQUENCES IN SCHEMA public TO trading_user;
