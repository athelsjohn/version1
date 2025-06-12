#!/usr/bin/env python3
import pandas as pd
import pickle
from datetime import datetime
from sklearn.cluster import KMeans
from sklearn.preprocessing import PowerTransformer
from sklearn.decomposition import PCA
import iniconfig
from surprise import Dataset, Reader, SVD
import logging
from logging.handlers import RotatingFileHandler

# Logging setup
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] RETRAIN: %(message)s",
    handlers=[
        RotatingFileHandler("retrain_models.log", maxBytes=1e6, backupCount=3),
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

def retrain_models_main(config_path = "/home/athel/Desktop/Litmus7/order/config.ini"):
    # Load config
    try:
        config = iniconfig.IniConfig(config_path)
        CLUSTER_NUMBER = int(config['general']['cluster_number'])
        PATHS = config['file_paths']
        CUSTOMER_DF_PATH = PATHS['customer_df']
        MERGED_DATA_PATH = PATHS['merged_data']
        KMEANS_MODEL_PATH = PATHS['kmeans_model']
        PT_MODEL_PATH = PATHS['pt_model']
        PCA_MODEL_PATH = PATHS['pca_model']
        CF_MODEL_TEMPLATE = PATHS['cf_model_template']
        logger.info("Configuration loaded successfully")
    except Exception as e:
        logger.exception("Failed to load configuration")
        raise SystemExit(f"Error loading config: {str(e)}")

    try:
        customer_df = pd.read_csv(CUSTOMER_DF_PATH)
        df = pd.read_csv(MERGED_DATA_PATH)
        if 'Date' in df.columns:
            df['Date'] = pd.to_datetime(df['Date']).dt.date
        logger.info(f"Loaded {len(customer_df)} customers and {len(df)} orders")
    except Exception as e:
        logger.exception("Failed to load data")
        raise

    # Train cluster models
    try:
        logger.info("Training clustering models...")
        features = ['total_spend','purchase_frequency','avg_basket_size','cat_diversity','recency','gap','age']
        X = customer_df[features].values

        pt = PowerTransformer(method='yeo-johnson', standardize=True)
        n_components = min(2, X.shape[0] - 1)  # Ensure n_components <= n_samples - 1
        pca = PCA(n_components=n_components)
        X_pca = pca.fit_transform(pt.fit_transform(X))

        kmeans = KMeans(n_clusters=CLUSTER_NUMBER, random_state=42)
        customer_df['KMeans Cluster'] = kmeans.fit_predict(X_pca)

        cluster_cols = [col for col in df.columns if 'KMeans Cluster' in col]
        df = df.drop(columns=cluster_cols, errors='ignore')

        df = pd.merge(df, customer_df[['Customer ID', 'KMeans Cluster']], on='Customer ID', how='left', validate='many_to_one')

        # Save models
        for obj, path in [(pt, PT_MODEL_PATH), (pca, PCA_MODEL_PATH), (kmeans, KMEANS_MODEL_PATH)]:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
        logger.info("Models saved successfully")
    except Exception as e:
        logger.exception("Model training failed")
        raise

    # Train collaborative filtering models
    try:
        logger.info("Training collaborative filtering models...")
        for cluster_id in customer_df['KMeans Cluster'].unique():
            cluster_users = customer_df[customer_df['KMeans Cluster'] == cluster_id]['Customer ID']
            cluster_data = df[df['Customer ID'].isin(cluster_users)]

            data_long = cluster_data[['Customer ID', 'Product ID', 'Quantity']]
            reader = Reader(rating_scale=(data_long['Quantity'].min(), data_long['Quantity'].max()))
            algo = SVD()
            algo.fit(Dataset.load_from_df(data_long, reader).build_full_trainset())

            with open(CF_MODEL_TEMPLATE.format(cluster_id=cluster_id), 'wb') as f:
                pickle.dump(algo, f)
            logger.info(f"CF model for cluster {cluster_id} saved")
    except Exception as e:
        logger.exception("CF model training failed")
        raise

    # Save updated customer_df (with cluster info)
    try:
        customer_df.to_csv(CUSTOMER_DF_PATH, index=False)
        df.to_csv(MERGED_DATA_PATH, index=False)
        logger.info("Retraining completed successfully")
    except Exception as e:
        logger.exception("Final save failed")
        raise

if __name__ == "__main__":
    retrain_models_main()