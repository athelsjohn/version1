#!/usr/bin/env python3
import pandas as pd
from datetime import datetime
import iniconfig
import logging
from logging.handlers import RotatingFileHandler
import os

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


def update_relations_main(config_path = "/home/athel/Desktop/Litmus7/order/config.ini"):
    # Load config
    try:
        config = iniconfig.IniConfig(config_path)
        PATHS = config['file_paths']
        MERGED_DATA_PATH = PATHS['merged_data']
        NEW_MERGED_DATA_PATH = PATHS['new_merged_data']
        ORDER_INFO_NORM = PATHS['order_info_norm']
        ORDER_LINE_NORM = PATHS['order_line_norm']
        PRODUCT_NORM = PATHS['product_norm']
        CUSTOMER_DF_PATH = PATHS['customer_df']
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.exception("Failed to load configuration")
        raise SystemExit(f"Error loading config: {str(e)}")

    # Load main data
    try:
        df = pd.read_csv(MERGED_DATA_PATH, parse_dates=['Date'])
        df['Date'] = pd.to_datetime(df['Date']).dt.date
        logger.info(f"Loaded {len(df)} records from main dataset")
    except FileNotFoundError:
        logger.warning("Main dataset not found, starting with empty DataFrame")
        df = pd.DataFrame()

    # Ingest new data if present
    if not df.empty and os.path.exists(NEW_MERGED_DATA_PATH) and os.path.getsize(NEW_MERGED_DATA_PATH) > 0:
        try:
            logger.info("Checking for new data...")
            new_df = pd.read_csv(NEW_MERGED_DATA_PATH, parse_dates=['Date'])
            new_df['Date'] = pd.to_datetime(new_df['Date']).dt.date
            existing_ids = df[['Order ID', 'Product ID', 'SKU ID']].drop_duplicates()
            new_df = new_df.merge(existing_ids, on=['Order ID', 'Product ID', 'SKU ID'], how='left', indicator=True)
            new_df = new_df[new_df['_merge'] == 'left_only'].drop('_merge', axis=1)
            if not new_df.empty:
                df = pd.concat([df, new_df], ignore_index=True)
                logger.info(f"Added {len(new_df)} new records")
            else:
                logger.info("No new records to add")
        except Exception as e:
            logger.exception("New data ingestion failed")

    if not df.empty:
        try:
            logger.info("Preprocessing data...")
            numeric_cols = ['Quantity', 'Price per Unit']
            df[numeric_cols] = df[numeric_cols].apply(pd.to_numeric, errors='coerce')
            df['Date'] = pd.to_datetime(df['Date'], errors='coerce')
            df = df.dropna(subset=['Date']).copy()
            df['Sales'] = df['Quantity'] * df['Price per Unit']
            today = pd.Timestamp(datetime.now().date())
            df['Recency'] = (today - df['Date']).dt.days
            df = df.sort_values(['Customer ID', 'Date'])
            df['Order Gap'] = df.groupby('Customer ID')['Date'].diff().dt.days.fillna(0)
            logger.info("Data preprocessing completed")
        except Exception as e:
            logger.exception("Error during preprocessing")
            raise

    # Normalized tables
        try:
            logger.info("Creating normalized tables...")
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