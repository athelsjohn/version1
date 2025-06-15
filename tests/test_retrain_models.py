import os
import pandas as pd
from datetime import datetime
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ml.retrain_models import retrain_models_main

def test_retrain_models_creates_models(tmp_path):
    """Tests the model retraining pipeline end-to-end.
    
    Verifies that the retrain_models_main function:
    1. Processes input customer and order data
    2. Generates clustering models (KMeans, PowerTransformer, PCA)
    3. Creates collaborative filtering models per cluster
    4. Persists all models to disk
    
    Args:
        tmp_path (pathlib.Path): pytest fixture providing temporary directory path
    
    Steps:
        1. Creates temporary config file with test paths
        2. Generates sample customer and order data
        3. Executes retraining pipeline
        4. Verifies output model file creation
    
    Raises:
        pytest.Fail: If models are not generated or unexpected errors occur
    """
    
    config_path = tmp_path / "config.ini"
    customer_df_path = tmp_path / "customer_df.csv"
    merged_data_path = tmp_path / "MergedData.csv"
    kmeans_model_path = tmp_path / "kmeans_model.pkl"
    pt_model_path = tmp_path / "pt.pkl"
    pca_model_path = tmp_path / "pca.pkl"
    cf_model_template = str(tmp_path / "cf_model_cluster_{cluster_id}.pkl")

    config_content = f"""
[general]
cluster_number = 2

[file_paths]
customer_df = {customer_df_path}
merged_data = {merged_data_path}
kmeans_model = {kmeans_model_path}
pt_model = {pt_model_path}
pca_model = {pca_model_path}
cf_model_template = {cf_model_template}
"""
    config_path.write_text(config_content)

    try:
        customer_df = pd.DataFrame({
            'Customer ID': ['CUST99998','CUST99997'],
            'total_spend': [20.0, 30.0],
            'purchase_frequency': [1, 2],
            'avg_basket_size': [2.0, 3.0],
            'cat_diversity': [1, 2],
            'recency': [0, 5],
            'gap': [0.0, 2.0],
            'age': [30, 35],
            'Date': [datetime.now().strftime('%Y-%m-%d'), datetime.now().strftime('%Y-%m-%d')]
        })
        customer_df.to_csv(customer_df_path, index=False)

        merged_data = pd.DataFrame({
            'Order ID': [1],
            'Customer ID': ['CUST1'],
            'Product ID': ['Product_1'],
            'SKU ID': ['SKU_1'],
            'Quantity': [2],
            'Price per Unit': [10.0],
            'Date': [datetime.now().strftime('%Y-%m-%d')]
        })
        merged_data.to_csv(merged_data_path, index=False)

        retrain_models_main(str(config_path))

        assert os.path.exists(kmeans_model_path)
        assert os.path.exists(pt_model_path)
        assert os.path.exists(pca_model_path)
        assert os.path.exists(cf_model_template.format(cluster_id=0))

    except SystemExit as e:
        pytest.fail(f"Retraining failed with exit code {e.code}")
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")