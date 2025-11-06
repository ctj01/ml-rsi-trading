"""
DAG: Download daily financial market data
Author: Cristian
Description: Downloads data from Yahoo Finance and saves it to PostgreSQL
"""

from datetime import datetime, timedelta
from airflow import DAG
from airflow.operators.python import PythonOperator
from airflow.providers.postgres.hooks.postgres import PostgresHook
import yfinance as yf
import pandas as pd
import logging
import requests

# Configuration
TICKERS = ['MSFT', 'AAPL', 'GOOGL', 'AMZN', 'NVDA']
POSTGRES_CONN_ID = 'trading_postgres'

# Default args for the DAG
default_args = {
    'owner': 'cristian',
    'depends_on_past': False,
    'email_on_failure': False,
    'email_on_retry': False,
    'retries': 3,
    'retry_delay': timedelta(minutes=5),
}

def fetch_market_data(**context):
    """
    Downloads market data for the configured tickers
    """
    execution_date = context['execution_date']
    
    # Check if this is a backfill run (use dag_run.conf to pass parameters)
    dag_run = context.get('dag_run')
    backfill = dag_run.conf.get('backfill', False) if dag_run and dag_run.conf else False
    
    if backfill:
        # For backfill, get data from 2020
        start_date = '2020-01-01'
        end_date = execution_date.strftime('%Y-%m-%d')
        logging.info(f"🔄 BACKFILL MODE: Fetching historical data from {start_date} to {end_date}")
    else:
        # For regular runs, get last 7 days
        start_date = (execution_date - timedelta(days=7)).strftime('%Y-%m-%d')
        end_date = execution_date.strftime('%Y-%m-%d')
        logging.info(f"Fetching data from {start_date} to {end_date}")
    
    all_data = []
    
    for ticker in TICKERS:
        try:
            logging.info(f"Downloading {ticker}...")
            
            # Use yfinance download - newer versions handle rate limiting better
            df = yf.download(ticker, start=start_date, end=end_date, progress=False)
            
            if df.empty:
                logging.warning(f"No data for {ticker}")
                continue
            
            # Clean MultiIndex if it exists
            if isinstance(df.columns, pd.MultiIndex):
                df.columns = df.columns.droplevel(1)
            
            # Reset index to have date as a column
            df = df.reset_index()
            df['ticker'] = ticker
            
            # Rename columns
            df = df.rename(columns={
                'Date': 'date',
                'Open': 'open',
                'High': 'high',
                'Low': 'low',
                'Close': 'close',
                'Volume': 'volume'
            })
            
            # Select only necessary columns
            df = df[['ticker', 'date', 'open', 'high', 'low', 'close', 'volume']]
            
            all_data.append(df)
            logging.info(f"✅ {ticker}: {len(df)} rows")
            
        except Exception as e:
            logging.error(f"Error downloading {ticker}: {str(e)}")
            continue
    
    if not all_data:
        raise ValueError("No data downloaded for any ticker")
    
    # Combine all data
    combined_df = pd.concat(all_data, ignore_index=True)
    logging.info(f"Total rows to insert: {len(combined_df)}")
    
    # Save to XCom for the next task
    return combined_df.to_json(orient='records', date_format='iso')


def save_to_database(**context):
    """
    Saves the downloaded data to PostgreSQL
    """
    # Get data from XCom
    ti = context['task_instance']
    data_json = ti.xcom_pull(task_ids='fetch_market_data')
    
    if not data_json:
        raise ValueError("No data received from previous task")
    
    # Convert JSON to DataFrame
    df = pd.read_json(data_json, orient='records')
    df['date'] = pd.to_datetime(df['date']).dt.date
    
    logging.info(f"Saving {len(df)} rows to database...")
    
    # Connect to PostgreSQL
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    # Insert data (ON CONFLICT to avoid duplicates)
    insert_query = """
        INSERT INTO market_data (ticker, date, open, high, low, close, volume)
        VALUES (%s, %s, %s, %s, %s, %s, %s)
        ON CONFLICT (ticker, date) 
        DO UPDATE SET
            open = EXCLUDED.open,
            high = EXCLUDED.high,
            low = EXCLUDED.low,
            close = EXCLUDED.close,
            volume = EXCLUDED.volume,
            created_at = CURRENT_TIMESTAMP
    """
    
    rows_inserted = 0
    for _, row in df.iterrows():
        try:
            cursor.execute(insert_query, (
                row['ticker'],
                row['date'],
                float(row['open']),
                float(row['high']),
                float(row['low']),
                float(row['close']),
                int(row['volume'])
            ))
            rows_inserted += 1
        except Exception as e:
            logging.error(f"Error inserting row: {str(e)}")
            continue
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logging.info(f"✅ Inserted/Updated {rows_inserted} rows")
    return rows_inserted


def log_pipeline_run(**context):
    """
    Logs the pipeline execution
    """
    ti = context['task_instance']
    execution_date = context['execution_date']
    
    # Get number of processed records
    records_processed = ti.xcom_pull(task_ids='save_to_database')
    
    # Convert execution_date to a proper datetime object
    if hasattr(execution_date, '__wrapped__'):
        # If it's a Proxy object, get the underlying value
        run_date = execution_date.__wrapped__.date()
    else:
        # Otherwise just convert to date
        run_date = execution_date.date()
    
    # Connect to PostgreSQL
    pg_hook = PostgresHook(postgres_conn_id=POSTGRES_CONN_ID)
    conn = pg_hook.get_conn()
    cursor = conn.cursor()
    
    insert_query = """
        INSERT INTO pipeline_runs 
        (pipeline_name, run_date, status, records_processed, execution_time_seconds)
        VALUES (%s, %s, %s, %s, %s)
    """
    
    # Calculate execution time (simplified)
    execution_time = 60  # Placeholder, ideally calculate the actual time
    
    cursor.execute(insert_query, (
        'fetch_market_data',
        run_date,
        'SUCCESS',
        records_processed,
        execution_time
    ))
    
    conn.commit()
    cursor.close()
    conn.close()
    
    logging.info(f"✅ Pipeline run logged: {records_processed} records")


# Define the DAG
with DAG(
    'fetch_market_data',
    default_args=default_args,
    description='Download daily market data from Yahoo Finance',
    schedule_interval='0 18 * * 1-5',  # Monday to Friday at 6 PM
    start_date=datetime(2025, 1, 1),
    catchup=False,
    tags=['trading', 'data-ingestion'],
) as dag:
    
    # Task 1: Download data
    task_fetch = PythonOperator(
        task_id='fetch_market_data',
        python_callable=fetch_market_data,
        provide_context=True,
    )
    
    # Task 2: Save to DB
    task_save = PythonOperator(
        task_id='save_to_database',
        python_callable=save_to_database,
        provide_context=True,
    )
    
    # Task 3: Log pipeline
    task_log = PythonOperator(
        task_id='log_pipeline_run',
        python_callable=log_pipeline_run,
        provide_context=True,
    )
    
    # Define execution order
    task_fetch >> task_save >> task_log
