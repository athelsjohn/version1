from fastapi.testclient import TestClient
from src.main import app

client = TestClient(app)

def test_add_order():
    order = {
        "Order_ID": 99995,
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

def test_order_exists():
    response = client.get("/orders?order_id=99999&product_id=Product_123&sku_id=SKU_456")
    assert response.status_code == 200
    assert response.json()["exists"] is True

def test_user_recommendation():
    response = client.post("/users?customer_id=CUSTTEST")
    assert response.status_code in (200, 404)
