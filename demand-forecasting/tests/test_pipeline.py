import pytest
import pandas as pd
import numpy as np
from src.data_pipeline import M5DataPipeline

@pytest.fixture
def pipeline():
    return M5DataPipeline("demand-forecasting/data/raw", "demand-forecasting/data/processed")

def test_feature_engineering(pipeline):
    # Create small mock sales data
    data = {
        "item_id": ["FOODS_3_090"],
        "dept_id": ["FOODS_3"],
        "cat_id": ["FOODS"],
        "store_id": ["CA_1"],
        "state_id": ["CA"],
    }
    # Add days d_1 to d_120
    for i in range(1, 121):
        data[f"d_{i}"] = [np.random.randint(0, 10)]
    
    sales = pd.DataFrame(data)
    sales['id'] = sales['item_id'] + "_" + sales['store_id']
    
    # Mock calendar and prices
    calendar = pd.DataFrame({
        "date": pd.date_range(start="2011-01-29", periods=120),
        "d": [f"d_{i}" for i in range(1, 121)],
        "wm_yr_wk": [11101] * 120,
        "event_name_1": [None] * 120,
        "snap_CA": [0] * 120,
        "snap_TX": [0] * 120,
        "snap_WI": [0] * 120
    })
    
    prices = pd.DataFrame({
        "store_id": ["CA_1"],
        "item_id": ["FOODS_3_090"],
        "wm_yr_wk": [11101],
        "sell_price": [1.50]
    })
    
    # Prepare long format
    id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
    sales_long = sales.melt(id_vars=id_vars, var_name="d", value_name="sales")
    df = sales_long.merge(calendar, on="d", how="left")
    df = df.merge(prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
    
    processed = pipeline.engineer_features(df)
    
    # Assertions
    expected_cols = ["sales_lag_7", "sales_lag_14", "sales_lag_28", "rolling_mean_7", "rolling_mean_28"]
    for col in expected_cols:
        assert col in processed.columns
    assert processed.isnull().sum().sum() == 0
    assert len(processed) > 0
