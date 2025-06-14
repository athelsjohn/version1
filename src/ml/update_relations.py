#!/usr/bin/env python3
import pandas as pd
from datetime import datetime
import iniconfig
import logging
from logging.handlers import RotatingFileHandler
import os
import sys

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] UPDATE: %(message)s",
    handlers=[
        RotatingFileHandler("update_relations.log", maxBytes=1e6, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def handle_critical_error(message, exit_code = 1):
    logger.critical(message)
    sys.exit(exit_code)


def update_relations_main(config_path = "/home/athel/Desktop/Litmus7/order/config.ini"):
    # Load config
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config file not found: {config_path}")
        config = iniconfig.IniConfig(config_path)
        PATHS = config['file_paths']
        required_paths = ['merged_data', 'new_merged_data','order_info_norm','order_line_norm','product_norm','customer_df']
        for path_key in required_paths:
            if path_key not in PATHS or not PATHS[path_key]:
                raise ValueError(f"Missing config entry: {path_key}")
        MERGED_DATA_PATH = PATHS['merged_data']
        NEW_MERGED_DATA_PATH = PATHS['new_merged_data']
        ORDER_INFO_NORM = PATHS['order_info_norm']
        ORDER_LINE_NORM = PATHS['order_line_norm']
        PRODUCT_NORM = PATHS['product_norm']
        CUSTOMER_DF_PATH = PATHS['customer_df']
        logger.info("Configuration loaded successfully")
    except (FileNotFoundError, PermissionError) as e:
        handle_critical_error(f"Config initialization failed: {str(e)}")
    except KeyError as e:
        handle_critical_error(f"Missing config section: {str(e)}")
    except Exception as e:
        handle_critical_error(f"Unexpected config error: {str(e)}")

    # Load main data
    try:
        df = pd.read_csv(MERGED_DATA_PATH, parse_dates=['Date'])
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        logger.info(f"Loaded {len(df)} records from main dataset")
    except FileNotFoundError:
        logger.warning("Main dataset not found, starting with empty DataFrame")
        df = pd.DataFrame()
    except pd.errors.EmptyDataError:
        logger.warning("Main dataset is empty or corrupt")
    except Exception as e:
        handle_critical_error(f"Data loading failed: {str(e)}")

    # Ingest new data if present
    if not df.empty and os.path.exists(NEW_MERGED_DATA_PATH) and os.path.getsize(NEW_MERGED_DATA_PATH) > 0:
        try:
            new_df = pd.read_csv(NEW_MERGED_DATA_PATH, parse_dates=['Date'])
            new_df['Date'] = pd.to_datetime(new_df['Date']).dt.date
            required_columns = ['Order ID', 'Customer ID', 'Product ID', 'Quantity', 'Price per Unit']
            missing = [col for col in required_columns if col not in new_df.columns]
            if missing:
                raise ValueError(f"New data missing columns: {missing}")
            existing_ids = df[['Order ID', 'Product ID', 'SKU ID']].drop_duplicates()
            new_df = new_df.merge(existing_ids, on=['Order ID', 'Product ID', 'SKU ID'], how='left', indicator=True)
            new_df = new_df[new_df['_merge'] == 'left_only'].drop('_merge', axis=1)
            if not new_df.empty:
                df = pd.concat([df, new_df], ignore_index=True)
                logger.info(f"Added {len(new_df)} new records")
            else:
                logger.info("No new records to add")
        except pd.errors.EmptyDataError:
            logger.warning("New data file is empty - skipping ingestion")
        except Exception as e:
            logger.error(f"New data ingestion failed: {str(e)}")

    if not df.empty:
        try:
            numeric_cols = ['Quantity', 'Price per Unit']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            invalid_numeric = df[numeric_cols].isnull().any(axis = 1)
            if invalid_numeric.any():
                logger.warning(f"Dropping {invalid_numeric.sum()} rows with invalid numeric data")
                df = df[~invalid_numericx]
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).copy()
            df['Sales'] = df['Quantity'] * df['Price per Unit']
            today = pd.Timestamp(datetime.now().date())
            df['Recency'] = (today - df['Date']).dt.days
            df = df.sort_values(['Customer ID', 'Date'])
            df['Order Gap'] = df.groupby('Customer ID')['Date'].diff().dt.days.fillna(0)
            logger.info("Data preprocessing completed")
        except Exception as e:
            handle_critical_error(f"Data validation failed: {str(e)}")

    # Normalized tables
        try:
            order_info = df[['Order ID', 'Customer ID', 'Warehouse ID', 'Customer Age',
                        'Customer Gender', 'Date', 'Recency', 'Order Gap']].drop_duplicates()
            order_info.to_csv(ORDER_INFO_NORM, index=False)
            logger.info(f"Order info saved to {ORDER_INFO_NORM}")

            order_line = df[['Order ID', 'Product ID', 'SKU ID', 'Category',
                        'Quantity', 'Price per Unit', 'Sales']].drop_duplicates()
            order_line.to_csv(ORDER_LINE_NORM, index=False)
            logger.info(f"Order line saved to {ORDER_LINE_NORM}")

            product = df[['Product ID', 'SKU ID', 'Category', 'Price per Unit']].drop_duplicates()
            product.to_csv(PRODUCT_NORM, index=False)
            logger.info(f"Product data saved to {PRODUCT_NORM}")
        except Exception as e:
            logger.exception("Normalization failed")
            raise

    # Customer features
        try:
            logger.info("Creating customer dataframe...")
            customer_df = df.groupby('Customer ID').agg(
                total_spend=('Sales', 'sum'),
                purchase_frequency=('Order ID', 'nunique'),
                avg_basket_size=('Quantity', 'mean'),
                cat_diversity=('Category', 'nunique'),
                recency=('Date', lambda x: (datetime.now().date() - x.max().date()).days),
                gap=('Order Gap', 'mean'),
                age=('Customer Age', 'first')
            ).reset_index()
            customer_df.to_csv(CUSTOMER_DF_PATH, index=False)
            logger.info(f"Customer dataframe saved to {CUSTOMER_DF_PATH}")
        except Exception as e:
            logger.exception("Customer dataframe creation failed")
            raise

        # Save merged data
        try:
            df.to_csv(MERGED_DATA_PATH, index=False)
            logger.info("Merged data saved")
        except Exception as e:
            logger.exception("Failed to save merged data")
            raise

    logger.info("Hourly relations update completed successfully.")

if __name__ == "__main__":
    update_relations_main()