"""
DAG to calculate the Relative Strength Index (RSI) for stock data.
AUTHOR: ctj01
DESCRIPTION: This DAG fetches stock price data, calculates the RSI using EMA method, and stores the results.
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator  #
from airflow.providers.postgres.hooks.postgres import PostgresHook
import pandas as pd
import numpy as np
import logging

POSTGRES_CONN_ID = 'trading_postgres'  # More standard name
RSI_PERIOD = 14
RSI_BUY_THRESHOLD = 37
RSI_SELL_THRESHOLD = 77

default_args = {
    'owner': 'ctj01',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 2,
    'retry_delay': timedelta(minutes=5),
}

def calculate_rsi(series, period=14):
    """
    Calculate the Relative Strength Index (RSI) using EMA (Wilder's method).
    
    Args:
        series (pd.Series): Price series (typically close prices)
        period (int): RSI period (default 14)
    
    Returns:
        pd.Series: RSI values (0-100)
    """
    delta = series.diff()
    gain = delta.where(delta > 0, 0)
    loss = -delta.where(delta < 0, 0)
    
    # EMA (Exponential Moving Average) - Standard method
    avg_gain = gain.ewm(alpha=1/period, adjust=False).mean()
    avg_loss = loss.ewm(alpha=1/period, adjust=False).mean()
    
    # Avoid division by zero
    rs = avg_gain / avg_loss.replace(0, 1e-10)
    rsi = 100 - (100 / (1 + rs))
    
    return rsi

def fetch_and_resample_data(**context):
    """
    Fetch market data and resample it to weekly frequency.
    
    Returns:
        str: JSON string with weekly data (passed via XCom)
    """
    execution_date = context['execution_date']

    # Get data for the last 6 months (or whatever is available)
    start_date = (execution_date - timedelta(days=180)).strftime('%Y-%m-%d')
    end_date = execution_date.strftime('%Y-%m-%d')
    
    logging.info(f"Processing data from {start_date} to {end_date}")
    
    # Connect to PostgreSQL
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    
    # Load daily data - first try the date range, if empty, get all available data
    query = f"""
        SELECT ticker, date, open, high, low, close, volume
        FROM market_data
        WHERE date >= '{start_date}' AND date <= '{end_date}'
        ORDER BY ticker, date
    """
    
    df = pd.read_sql(query, conn)
    
    # If no data in the date range, try to get all available data
    if df.empty:
        logging.warning(f"No data found in range {start_date} to {end_date}, fetching all available data")
        query = """
            SELECT ticker, date, open, high, low, close, volume
            FROM market_data
            ORDER BY ticker, date
        """
        df = pd.read_sql(query, conn)
    
    conn.close()
    
    if df.empty:
        raise ValueError("No data found in database")
    
    logging.info(f"Loaded {len(df)} daily records for {df['ticker'].nunique()} tickers")

    # Process each ticker
    all_weekly = []
    
    for ticker in df['ticker'].unique():
        try:
            ticker_data = df[df['ticker'] == ticker].copy()
            ticker_data['date'] = pd.to_datetime(ticker_data['date'])
            ticker_data = ticker_data.set_index('date')
            
            # Resample to weekly
            weekly = ticker_data.resample('W').agg({
                'open': 'first',
                'high': 'max',
                'low': 'min',
                'close': 'last',
                'volume': 'sum'
            }).dropna()
            
            # Calculate RSI using EMA method
            weekly['rsi'] = calculate_rsi(weekly['close'], period=RSI_PERIOD)

            # Determine signal
            def get_signal(rsi):
                if pd.isna(rsi):
                    return 'NEUTRAL'
                elif rsi < RSI_BUY_THRESHOLD:
                    return 'BUY'
                elif rsi > RSI_SELL_THRESHOLD:
                    return 'SELL'
                else:
                    return 'NEUTRAL'
            
            weekly['signal'] = weekly['rsi'].apply(get_signal)
            weekly['ticker'] = ticker

            # Reset index
            weekly = weekly.reset_index()
            weekly['week_start'] = weekly['date']
            weekly['week_end'] = weekly['date']
            
            # Select relevant columns
            weekly = weekly[['ticker', 'week_start', 'week_end', 'open', 'high', 'low', 'close', 'volume', 'rsi', 'signal']]
            
            all_weekly.append(weekly)
            logging.info(f"{ticker}: {len(weekly)} weekly candles")
            
        except Exception as e:
            logging.error(f"Error processing {ticker}: {str(e)}")
            continue
    
    if not all_weekly:
        raise ValueError("No weekly data calculated")
    
    # Combine all tickers
    combined_weekly = pd.concat(all_weekly, ignore_index=True)
    logging.info(f"Total weekly records: {len(combined_weekly)}")
    
    return combined_weekly.to_json(orient='records', date_format='iso')

def store_weekly_data(**context):
    """
    Store the calculated weekly RSI data back into the database.
    """
    ti = context['task_instance']
    data_json = ti.xcom_pull(task_ids='calculate_weekly_rsi')
    
    if not data_json:
        raise ValueError("No data received from previous task")
    
    weekly_data = pd.read_json(data_json, orient='records')
    weekly_data['week_start'] = pd.to_datetime(weekly_data['week_start']).dt.date
    weekly_data['week_end'] = pd.to_datetime(weekly_data['week_end']).dt.date

    logging.info(f"Saving {len(weekly_data)} weekly records to database...")

    # Connect to PostgreSQL
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # Insert data with conflict handling
    insert_query = """
        INSERT INTO rsi_weekly 
        (ticker, week_start, week_end, open, high, low, close, volume, rsi, signal)
        VALUES (%s, %s, %s, %s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, week_start)
        DO UPDATE SET
            week_end = EXCLUDED.week_end,
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            rsi = EXCLUDED.rsi,
            signal = EXCLUDED.signal,
            updated_at = CURRENT_TIMESTAMP
    """
    
    rows_inserted = 0
    for _, row in weekly_data.iterrows():
        try:
            cursor.execute(insert_query, (
                row['ticker'],
                row['week_start'],
                row['week_end'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume']),
                float(row['rsi']) if not pd.isna(row['rsi']) else None,  # ✅ Handle NaN
                row['signal']
            ))
            rows_inserted += 1
        except Exception as e:
            logging.error(f"Error inserting row: {str(e)}")
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logging.info(f"nserted/Updated {rows_inserted} weekly RSI records")
    return rows_inserted

def generate_trading_signals(**context):
    """
    Generate trading signals based on the latest RSI values.
    """
    execution_date = context['execution_date']  # Removed duplicate line
    
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # Get latest weekly data with signals
    query = """
        SELECT ticker, week_start, rsi, close, signal
        FROM rsi_weekly
        WHERE week_start = (SELECT MAX(week_start) FROM rsi_weekly)
        AND signal IN ('BUY', 'SELL')
    """
    
    df = pd.read_sql(query, conn)
    
    if df.empty:
        logging.info("No BUY/SELL signals generated this week")
        conn.close()
        return 0
    
    # Insert signals into trading_signals table
    insert_query = """
        INSERT INTO trading_signals 
        (ticker, signal_date, signal, rsi, price, confidence)
        VALUES (%s, %s, %s, %s, %s, %s)
    """
    
    rows_inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(insert_query, (
                row['ticker'],
                execution_date.date(),
                row['signal'],
                float(row['rsi']),
                float(row['close']),
                0.80  # Placeholder confidence (will be replaced by ML model later)
            ))
            rows_inserted += 1
            logging.info(f"Signal: {row['ticker']} - {row['signal']} (RSI: {row['rsi']:.2f})")
        except Exception as e:
            logging.error(f"Error inserting signal: {str(e)}")
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logging.info(f"Generated {rows_inserted} trading signals")
    return rows_inserted

# Define the DAG
with DAG(
    'calculate_rsi_weekly',
    default_args=default_args,
    description='DAG to calculate weekly RSI and generate trading signals',
    schedule_interval='0 20 * * 0',  #  Sundays at 8 PM (after weekly close)
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['rsi', 'trading', 'weekly'],
) as dag:

    calculate_weekly_rsi = PythonOperator(
        task_id='calculate_weekly_rsi',
        python_callable=fetch_and_resample_data,
        provide_context=True,
    )

    store_weekly_rsi = PythonOperator(
        task_id='store_weekly_rsi',
        python_callable=store_weekly_data,
        provide_context=True,
    )

    generate_signals = PythonOperator(
        task_id='generate_trading_signals',
        python_callable=generate_trading_signals,
        provide_context=True,
    )

    # Define task dependencies (topological sort order)
    calculate_weekly_rsi >> store_weekly_rsi >> generate_signals
