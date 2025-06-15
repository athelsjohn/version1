from fastapi.testclient import TestClient
from src.main import app
import logging

logging.basicConfig(
    level = logging.INFO,
    format = "%(asctime)s [%(levelname)s] TEST_MAIN_API: %(message)s",
    handlers = [
        logging.StreamHandler()
    ]
)
logger = logging.getLogger(__name__)

client = TestClient(app)

def test_add_order():

    """Tests the order creation endpoint.

    Verifies that the `/orders` POST endpoint correctly processes a new order,
    returns a success status (200), and includes the expected success message.
    Logs the test result and any errors encountered.

    Raises:
        Exception: If the test fails, logs the error and re-raises.
    """

    try:
        order = {
            "Order_ID": 99994,
            "Customer_ID": "CUST2002",
            "Warehouse_ID": "WH001",
            "Customer_Age": 25,
            "Customer_Gender": "Male",
            "Date": "2025-06-09",
            "Product_ID": "Product_123",
            "SKU_ID": "SKU_456",
            "Category": "Electronics",
            "Quantity": 1,
            "Price_per_Unit": 100.0
        }
        response = client.post("/orders", json=order)
        assert response.status_code == 200
        assert "Order added successfully" in response.json()["message"]
        logger.info("test_add_order passed")
    except Exception as e:
        logger.error(f"test_add_order failed: {str(e)}")
        raise

def test_order_exists():

    """Tests the order existence endpoint.

    Verifies that the `/orders` GET endpoint correctly checks if an order exists,
    returns a success status (200), and provides the correct existence status.
    Logs the test result and any errors encountered.

    Raises:
        Exception: If the test fails, logs the error and re-raises.
    """

    try:
        response = client.get("/orders?order_id=99999&product_id=Product_123&sku_id=SKU_456")
        assert response.status_code == 200
        assert response.json()["exists"] is True
        logger.info("test_order_exists passed")
    except Exception as e:
        logger.error(f"test_order_exists failed: {str(e)}")
        raise

def test_user_recommendation():

    """Tests the user recommendation endpoint.

    Verifies that the `/users` POST endpoint responds with a valid status (200 or 404)
    when requesting recommendations for a customer. Logs the test result and any errors.

    Raises:
        Exception: If the test fails, logs the error and re-raises.
    """

    try:
        response = client.post("/users?customer_id=CUSTTEST")
        assert response.status_code in (200, 404)
        logger.info("test_user_recommendation passed")
    except Exception as e:
        logger.error(f"test_user_recommendation failed: {str(e)}")
        raise
