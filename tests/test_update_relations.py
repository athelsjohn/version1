import os
import pandas as pd
from datetime import datetime
import sys
import pytest

sys.path.insert(0, os.path.abspath(os.path.join(os.path.dirname(__file__), '..', 'src')))

from ml.update_relations import update_relations_main

def test_update_relations_creates_outputs(tmp_path):
    # Setup dummy config and data
    config_path = tmp_path / "config.ini"
    merged_data_path = tmp_path / "MergedData.csv"
    new_merged_data_path = tmp_path / "new_MergedData.csv"
    order_info_norm = tmp_path / "order_info_normalized.csv"
    order_line_norm = tmp_path / "order_line_normalized.csv"
    product_norm = tmp_path / "product_normalized.csv"
    customer_df_path = tmp_path / "customer_df.csv"

    config_content = f"""
[general]
cluster_number = 2

[file_paths]
merged_data = {merged_data_path}
new_merged_data = {new_merged_data_path}
customer_df = {customer_df_path}
order_info_norm = {order_info_norm}
order_line_norm = {order_line_norm}
product_norm = {product_norm}
"""
    config_path.write_text(config_content)

    try:
        df = pd.DataFrame({
            'Order ID': [1],
            'Customer ID': ['CUST1'],
            'Warehouse ID': ['WH001'],
            'Customer Age': [30],
            'Customer Gender': ['M'],
            'Date': [datetime.now().strftime('%Y-%m-%d')],
            'Product ID': ['Product_1'],
            'SKU ID': ['SKU_1'],
            'Category': ['Cat1'],
            'Quantity': [2],
            'Price per Unit': [10.0]
        })
        df.to_csv(merged_data_path, index=False)

        update_relations_main(str(config_path))

        assert os.path.exists(order_info_norm)
        assert os.path.exists(order_line_norm)
        assert os.path.exists(product_norm)
        assert os.path.exists(customer_df_path)

    except SystemExit as e:
        pytest.fail(f"Update failed with exit code {e.code}")
    except Exception as e:
        pytest.fail(f"Test failed: {str(e)}")