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
import os
import sys


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

def handle_critical_error(message, exit_code = 1):
    logger.critical(message)
    sys.exit(exit_code)

def retrain_models_main(config_path = "/home/athel/Desktop/Litmus7/order/config.ini"):
    # Load config
    try:
        if not os.path.exists(config_path):
            raise FileNotFoundError(f"Config File not found: {config_path}")
        config = iniconfig.IniConfig(config_path)
        PATHS = config['file_paths']
        required_paths = ['customer_df','merged_data', 'kmeans_model', 'pt_model', 'pca_model', 'cf_model_template']
        for path_key in required_paths:
            if path_key not in PATHS or not PATHS[path_key]:
                raise ValueError(f"Missing Config Entry: {path_key}")
        for path in [PATHS['kmeans_model'], PATHS['pt_model'], PATHS['pca_model']]:
            dir_path = os.path.dirname(path)
            if dir_path:
                os.makedirs(dir_path, exist_ok = True)
        CLUSTER_NUMBER = int(config['general']['cluster_number'])
        CUSTOMER_DF_PATH = PATHS['customer_df']
        MERGED_DATA_PATH = PATHS['merged_data']
        KMEANS_MODEL_PATH = PATHS['kmeans_model']
        PT_MODEL_PATH = PATHS['pt_model']
        PCA_MODEL_PATH = PATHS['pca_model']
        CF_MODEL_TEMPLATE = PATHS['cf_model_template']
        logger.info("Configuration loaded successfully")
    except (FileNotFoundError, PermissionError) as e:
        handle_critical_error(f"Config initialization failed: {str(e)}")
    except KeyError as e:
        handle_critical_error(f"Missing config section: {str(e)}")
    except Exception as e:
        handle_critical_error(f"Unexpected config error: {str(e)}")

    try:
        customer_df = pd.read_csv(CUSTOMER_DF_PATH, dtype = {'Customer ID': str})
        df = pd.read_csv(MERGED_DATA_PATH, parse_dates=['Date'])
        required_cols = ['Customer ID', 'Date']
        for df_name, dataframe in [('customer_df', customer_df), ('merged_data', df)]:
            missing = [col for col in required_cols if col not in dataframe.columns]
            if missing:
                raise ValueError(f"{df_name} missing columns: {', '.join(missing)}")
        logger.info(f"Loaded {len(customer_df)} customers and {len(df)} orders")
    except pd.errors.EmptyDataError:
        handle_critical_error("Input CSV file is empty or corrupt")
    except pd.errors.ParserError:
        handle_critical_error("CSV parsing error - check file format")
    except Exception as e:
        handle_critical_error(f"Data loading failed: {str(e)}")

    # Train cluster models
    try:
        features = ['total_spend','purchase_frequency','avg_basket_size','cat_diversity','recency','gap','age']
        missing_features = [f for f in features if f not in customer_df.columns]
        if missing_features:
            raise ValueError(f"Missing Customer Features: {missing_features}")
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
        logger.info("Models trained and merged successfully")
    except ValueError as e:
        handle_critical_error(f"Feature validation failed: {str(e)}")
    except MemoryError:
        handle_critical_error("Insufficient memory for model training")
    except Exception as e:
        handle_critical_error(f"Model training failed: {str(e)}")

    try:
        for obj, path in [(pt, PT_MODEL_PATH), (pca, PCA_MODEL_PATH), (kmeans, KMEANS_MODEL_PATH)]:
            with open(path, 'wb') as f:
                pickle.dump(obj, f)
            with open(path, 'rb') as f:
                pickle.load(f)
        logger.info("Models saved and verified successfully")
    except (pickle.PickleError, EOFError):
        handle_critical_error("Model serialization failed - corrupt output file")
    except Exception as e:
        handle_critical_error(f"Model saving failed: {str(e)}")

    # Train collaborative filtering models
    try:
        for cluster_id in customer_df['KMeans Cluster'].unique():
            cluster_users = customer_df[customer_df['KMeans Cluster'] == cluster_id]['Customer ID']
            cluster_data = df[df['Customer ID'].isin(cluster_users)]
            data_long = cluster_data[['Customer ID', 'Product ID', 'Quantity']]
            reader = Reader(rating_scale=(data_long['Quantity'].min(), data_long['Quantity'].max()))
            algo = SVD()
            algo.fit(Dataset.load_from_df(data_long, reader).build_full_trainset())
            with open(CF_MODEL_TEMPLATE.format(cluster_id=cluster_id), 'wb') as f:
                pickle.dump(algo, f)
            with open(CF_MODEL_TEMPLATE.format(cluster_id=cluster_id), 'rb') as f:
                pickle.load(f)
            logger.info(f"CF model for cluster {cluster_id} saved and verified")
    except (pickle.PickleError, EOFError):
        handle_critical_error("CF model serialization failed")
    except Exception as e:
        handle_critical_error(f"CF model training failed: {str(e)}")
    # Save updated customer_df (with cluster info)
    try:
        customer_df.to_csv(CUSTOMER_DF_PATH, index=False)
        df.to_csv(MERGED_DATA_PATH, index=False)
        logger.info("Retraining completed successfully")
    except Exception as e:
        handle_critical_error(f"Final Save failed: {str(e)}")

if __name__ == "__main__":
    retrain_models_main()