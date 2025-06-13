from fastapi import FastAPI, HTTPException
from pydantic import BaseModel, field_validator, ConfigDict, Field
import pandas as pd
import pickle
import os
import re
import iniconfig
from datetime import datetime
import logging
from logging.handlers import RotatingFileHandler

# Logging configuration
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] API: %(message)s",
    handlers=[
        RotatingFileHandler("api.log", maxBytes=1e6, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

# Load configuration
config = iniconfig.IniConfig("/home/athel/Desktop/Litmus7/order/config.ini")
CLUSTER_NUMBER = int(config['general']['cluster_number'])
PATHS = config["file_paths"]
MERGED_DATA_PATH = PATHS.get("merged_data")
CUSTOMER_DF_PATH = PATHS.get("customer_df")
KMEANS_MODEL_PATH = PATHS.get("kmeans_model")
PT_MODEL_PATH = PATHS.get("pt_model")
PCA_MODEL_PATH = PATHS.get("pca_model")
CF_MODEL_TEMPLATE = PATHS.get("cf_model_template")

app = FastAPI()

def load_models_and_data():
    global model_kmeans, pt, pca, cf_models, customer_df, df
    try:
        logger.info("Loading machine learning models...")
        with open(KMEANS_MODEL_PATH, 'rb') as f:
            model_kmeans = pickle.load(f)
        with open(PT_MODEL_PATH, 'rb') as f:
            pt = pickle.load(f)
        with open(PCA_MODEL_PATH, 'rb') as f:
            pca = pickle.load(f)
        logger.info("Loading collaborative filtering models...")
        cf_models = {}
        for i in range(CLUSTER_NUMBER):
            with open(CF_MODEL_TEMPLATE.format(cluster_id=i), 'rb') as f:
                cf_models[i] = pickle.load(f)
        logger.info("Loading customer data...")
        customer_df = pd.read_csv(CUSTOMER_DF_PATH)
        df = pd.read_csv(MERGED_DATA_PATH)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.date
        logger.info("Initial data loading completed successfully")
    except Exception as e:
        logger.exception("Failed to initialize models and data")
        raise RuntimeError(f"Initialization error: {str(e)}")

load_models_and_data()

class OrderLine(BaseModel):
    model_config = ConfigDict(
        populate_by_name=True
    )
    Order_ID: int = Field(alias="Order ID")
    Customer_ID: str = Field(alias="Customer ID")
    Warehouse_ID: str = Field(alias="Warehouse ID")
    Customer_Age: int = Field(alias="Customer Age")
    Customer_Gender: str = Field(alias="Customer Gender")
    Date: str = Field(alias="Date")
    Product_ID: str = Field(alias="Product ID")
    SKU_ID: str = Field(alias="SKU ID")
    Category: str = Field(alias="Category")
    Quantity: int = Field(alias="Quantity")
    Price_per_Unit: float = Field(alias="Price per Unit")

    @field_validator('Warehouse_ID')
    @staticmethod
    def warehouse_id_format(v):
        if not re.fullmatch(r'WH\d+', v):
            raise ValueError("Warehouse_ID must start with 'WH' followed by digits.")
        return v

    @field_validator('Customer_ID')
    @staticmethod
    def customer_id_format(v):
        if not re.fullmatch(r'CUST\d+', v):
            raise ValueError("Customer_ID must start with 'CUST' followed by digits.")
        return v

    @field_validator('Product_ID')
    @staticmethod
    def product_id_format(v):
        if not re.fullmatch(r'Product_\d+', v):
            raise ValueError("Product_ID must start with 'Product_' followed by digits.")
        return v

    @field_validator('SKU_ID')
    @staticmethod
    def sku_id_format(v):
        if not re.fullmatch(r'SKU_\d+', v):
            raise ValueError("SKU_ID must start with 'SKU_' followed by digits.")
        return v

@app.post("/orders")
def add_order(order: OrderLine):
    """Add a new order to MergedData.csv"""
    try:
        logger.info(f"Received new order request: {order.model_dump()}")
        # Use by_alias to ensure columns match CSV ("Order ID", etc.)
        new_row = pd.DataFrame([order.model_dump(by_alias=True)])

        new_row['Date'] = pd.to_datetime(new_row['Date']).dt.date

        # Load existing data or create empty DataFrame
        if os.path.exists(MERGED_DATA_PATH):
            merged_df = pd.read_csv(MERGED_DATA_PATH)
            if 'Date' in merged_df.columns:
                merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.date
        else:
            merged_df = pd.DataFrame(columns=new_row.columns)
            logger.info("Created new merged dataframe")

        # Check for duplicates
        key_cols = ['Order ID', 'Product ID', 'SKU ID']
        is_duplicate = merged_df.merge(new_row[key_cols], on=key_cols).shape[0] > 0
        if is_duplicate:
            logger.warning(f"Duplicate order detected: {order.model_dump()}")
            raise HTTPException(status_code=400, detail="Duplicate order line detected.")

        # Calculate derived fields
        new_order_date = new_row['Date'].iloc[0]
        customer_id = new_row['Customer ID'].iloc[0]
        new_row['Sales'] = new_row['Quantity'] * new_row['Price per Unit']
        today = datetime.now().date()
        new_row['Recency'] = (today - new_order_date).days

        customer_orders = merged_df[merged_df['Customer ID'] == customer_id]
        if not customer_orders.empty:
            last_order_date = customer_orders['Date'].max()
            last_order_date = pd.to_datetime(last_order_date).date() if not isinstance(last_order_date, datetime) else last_order_date
            new_row['Order Gap'] = (new_order_date - last_order_date).days
        else:
            new_row['Order Gap'] = 0
            logger.info(f"First order for customer {customer_id}")

        # Append and save
        merged_df = pd.concat([merged_df, new_row], ignore_index=True)
        merged_df['Date'] = pd.to_datetime(merged_df['Date']).dt.date
        merged_df.to_csv(MERGED_DATA_PATH, index=False)

        logger.info(f"Successfully added order ID {order.Order_ID}")
        return {"message": "Order added successfully."}

    except HTTPException as he:
        raise
    except Exception as e:
        logger.exception(f"Failed to add order: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))

@app.post("/users")
def predict_next_product(customer_id: str):
    """Get product recommendations for a customer"""
    try:
        logger.info(f"Starting recommendation process for {customer_id}")
        cust_row = customer_df[customer_df['Customer ID'] == customer_id]
        if cust_row.empty:
            logger.warning(f"Customer not found: {customer_id}")
            raise ValueError("Customer not found")
        features = cust_row[['total_spend', 'purchase_frequency', 'avg_basket_size',
                            'cat_diversity', 'recency', 'gap', 'age']].values
        features_pca = pca.transform(pt.transform(features))
        cluster = int(model_kmeans.predict(features_pca)[0])
        logger.debug(f"Customer {customer_id} assigned to cluster {cluster}")
        all_products = set(df['Product ID'].unique())
        cf_model = cf_models[cluster]
        predictions = []
        for pid in all_products:
            pred = cf_model.predict(customer_id, pid)
            predictions.append((pid, pred.est))
        top_products = sorted(predictions, key=lambda x: x[1], reverse=True)[:5]
        logger.info(f"Generated recommendations for {customer_id}: {[pid for pid, _ in top_products]}")
        return {"recommended_products": [pid for pid, _ in top_products]}
    except Exception as e:
        logger.exception(f"Recommendation failed for {customer_id}: {str(e)}")
        raise HTTPException(status_code=404, detail=str(e))

@app.get("/orders")
def order_exists(order_id: int, product_id: str, sku_id: str):
    """Check if an order line exists in MergedData.csv"""
    try:
        logger.info(f"Checking existence: Order {order_id}, Product {product_id}, SKU {sku_id}")
        if not os.path.exists(MERGED_DATA_PATH):
            return {"exists": False}
        df = pd.read_csv(MERGED_DATA_PATH)
        exists = ((df['Order ID'] == order_id) &
                  (df['Product ID'] == product_id) &
                  (df['SKU ID'] == sku_id)).any()
        logger.info(f"Existence check result: {exists}")
        return {"exists": bool(exists)}
    except Exception as e:
        logger.exception(f"Existence check failed: {str(e)}")
        raise HTTPException(status_code=500, detail=str(e))
