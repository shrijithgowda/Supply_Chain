import pandas as pd
import numpy as np
from statsmodels.tsa.statespace.sarimax import SARIMAX
from sklearn.metrics import mean_absolute_error, mean_squared_error
import json
import os
from src.data_pipeline import M5DataPipeline

def wmape(y_true, y_pred):
    """
    Weighted Mean Absolute Percentage Error.
    """
    return np.sum(np.abs(y_true - y_pred)) / (np.sum(np.abs(y_true)) + 1e-6)

class BaselineModel:
    def __init__(self, pipeline: M5DataPipeline):
        self.pipeline = pipeline
        self.metrics_file = "baseline_metrics.json"

    def fit_and_evaluate(self, n_skus: int = 50):
        print(f"Loading data for baseline (sampling {n_skus} SKUs)...")
        # Load enough data to get n_skus
        # Each SKU has ~1913 rows in the full dataset.
        # We'll load a subset of the original sales file.
        raw_sales = pd.read_csv(os.path.join(self.pipeline.raw_data_path, "sales_train_validation.csv"), nrows=n_skus)
        
        # Add 'id' column if missing
        if 'id' not in raw_sales.columns:
            raw_sales['id'] = raw_sales['item_id'] + "_" + raw_sales['store_id']

        # Manually melt and prepare for SARIMA
        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        sales_long = raw_sales.melt(id_vars=id_vars, var_name="d", value_name="sales")
        
        # Filter for 28 day test horizon
        # d_1 to d_1913
        # Train: d_1 to d_1885
        # Test: d_1886 to d_1913
        
        results = []
        all_metrics = []
        
        sku_ids = sales_long["id"].unique()
        for sku_id in sku_ids:
            print(f"Fitting SARIMA for {sku_id}...")
            sku_data = sales_long[sales_long["id"] == sku_id].sort_values("d")
            
            # Map d_x to integer for sorting if needed, but melting order is usually fine
            # Actually, let's just use the order.
            
            y = sku_data["sales"].values
            y_train = y[:-28]
            y_test = y[-28:]
            
            try:
                # SARIMA(1,1,1)(1,1,1,7)
                model = SARIMAX(y_train, order=(1, 1, 1), seasonal_order=(1, 1, 1, 7))
                model_fit = model.fit(disp=False)
                y_pred = model_fit.forecast(steps=28)
                y_pred = np.maximum(y_pred, 0) # Ensure no negative sales
                
                mae = mean_absolute_error(y_test, y_pred)
                rmse = np.sqrt(mean_squared_error(y_test, y_pred))
                wmape_val = wmape(y_test, y_pred)
                
                all_metrics.append({
                    "sku_id": sku_id,
                    "mae": mae,
                    "rmse": rmse,
                    "wmape": wmape_val
                })
            except Exception as e:
                print(f"Failed to fit SARIMA for {sku_id}: {e}")
                
        # Aggregate metrics
        if all_metrics:
            metrics_df = pd.DataFrame(all_metrics)
            avg_metrics = {
                "avg_mae": metrics_df["mae"].mean(),
                "avg_rmse": metrics_df["rmse"].mean(),
                "avg_wmape": metrics_df["wmape"].mean()
            }
            
            with open(self.metrics_file, "w") as f:
                json.dump(avg_metrics, f, indent=4)
                
            print(f"Baseline evaluation complete. Avg WMAPE: {avg_metrics['avg_wmape']:.4f}")
            return avg_metrics
        else:
            print("No baseline metrics generated.")
            return None

if __name__ == "__main__":
    pipeline = M5DataPipeline("demand-forecasting/data/raw", "demand-forecasting/data/processed")
    baseline = BaselineModel(pipeline)
    baseline.fit_and_evaluate(n_skus=10) # Using 10 for faster check
