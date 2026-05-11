import pandas as pd
import numpy as np
from typing import Tuple, List
import os
from sklearn.preprocessing import LabelEncoder

class M5DataPipeline:
    """
    Data pipeline for the M5 Forecasting dataset.
    Handles ingestion, cleaning, feature engineering, and splitting.
    """
    
    def __init__(self, raw_data_path: str, processed_data_path: str):
        self.raw_data_path = raw_data_path
        self.processed_data_path = processed_data_path
        self.label_encoders = {}

    def load_data(self, nrows: int = None) -> pd.DataFrame:
        """
        Loads and merges M5 dataset files.
        """
        print("Loading calendar.csv...")
        calendar = pd.read_csv(os.path.join(self.raw_data_path, "calendar.csv"))
        # Add 'd' column if missing
        if 'd' not in calendar.columns:
            calendar['d'] = [f"d_{i+1}" for i in range(len(calendar))]
        
        print("Loading sell_prices.csv...")
        sell_prices = pd.read_csv(os.path.join(self.raw_data_path, "sell_prices.csv"))
        
        print("Loading sales_train_validation.csv...")
        sales = pd.read_csv(os.path.join(self.raw_data_path, "sales_train_validation.csv"), nrows=nrows)
        
        # Add 'id' column if missing
        if 'id' not in sales.columns:
            sales['id'] = sales['item_id'] + "_" + sales['store_id']
            
        # Melt sales to long format
        print("Melting sales data...")
        id_vars = ["id", "item_id", "dept_id", "cat_id", "store_id", "state_id"]
        sales = sales.melt(id_vars=id_vars, var_name="d", value_name="sales")
        
        # Merge with calendar
        print("Merging with calendar...")
        sales = sales.merge(calendar, on="d", how="left")
        
        # Merge with sell_prices
        print("Merging with sell_prices...")
        sales = sales.merge(sell_prices, on=["store_id", "item_id", "wm_yr_wk"], how="left")
        
        return sales

    def engineer_features(self, df: pd.DataFrame) -> pd.DataFrame:
        """
        Performs feature engineering as per the requirements.
        """
        print("Engineering features...")
        
        # 1. Static features
        static_cols = ["item_id", "dept_id", "cat_id", "store_id", "state_id"]
        for col in static_cols:
            df[col] = df[col].astype(str)
            
        # 2. Time-varying known features
        df["date"] = pd.to_datetime(df["date"])
        df["day_of_week"] = df["date"].dt.dayofweek.astype(str)
        df["week_of_year"] = df["date"].dt.isocalendar().week.astype(int)
        df["month"] = df["date"].dt.month.astype(str)
        df["is_weekend"] = (df["date"].dt.dayofweek >= 5).astype(int)
        
        # is_holiday from event_name_1
        df["is_holiday"] = df["event_name_1"].notna().astype(int).astype(str)
        
        for col in ["snap_CA", "snap_TX", "snap_WI"]:
            df[col] = df[col].astype(str)
            
        # Price features
        df["price_change_pct"] = df.groupby(["store_id", "item_id"])["sell_price"].pct_change()
        
        # 3. Time-varying unknown features (Lags and Rolling stats)
        # Sort by date and item/store for correct shift/rolling
        df = df.sort_values(["id", "date"])
        
        lags = [7, 14, 28]
        for lag in lags:
            df[f"sales_lag_{lag}"] = df.groupby("id")["sales"].shift(lag)
            
        rolling_windows = [7, 28]
        for window in rolling_windows:
            df[f"rolling_mean_{window}"] = df.groupby("id")["sales"].transform(lambda x: x.shift(28).rolling(window).mean())
            df[f"rolling_std_{window}"] = df.groupby("id")["sales"].transform(lambda x: x.shift(28).rolling(window).std())
            
        df["rolling_max_28"] = df.groupby("id")["sales"].transform(lambda x: x.shift(28).rolling(28).max())
        df["rolling_min_28"] = df.groupby("id")["sales"].transform(lambda x: x.shift(28).rolling(28).min())
        
        df["sales_velocity"] = df["rolling_mean_7"] / (df["rolling_mean_28"] + 1e-6)
        
        # Handle NaNs from lags/rolling
        df["price_change_pct"] = df["price_change_pct"].fillna(0)
        df["sales_velocity"] = df["sales_velocity"].fillna(1) # Default velocity is 1
        
        # Fill other rolling features with 0 if they are still NaN
        rolling_cols = [c for c in df.columns if "rolling_" in c]
        df[rolling_cols] = df[rolling_cols].fillna(0)
        
        df = df.dropna(subset=["sales_lag_28", "sell_price"])
        
        # Final fill for any residual NaNs
        df = df.fillna(0)
        
        # Time index for pytorch-forecasting
        df["time_idx"] = (df["date"] - df["date"].min()).dt.days
        
        return df

    def split_data(self, df: pd.DataFrame) -> Tuple[pd.DataFrame, pd.DataFrame, pd.DataFrame]:
        """
        Splits data into train, validation, and test sets.
        Validation and test sets must include history for the encoder.
        """
        max_time = df["time_idx"].max()
        test_horizon = 28
        encoder_history = 90
        
        test = df[df["time_idx"] > max_time - test_horizon - encoder_history]
        val = df[(df["time_idx"] > max_time - 2*test_horizon - encoder_history) & (df["time_idx"] <= max_time - test_horizon)]
        train = df[df["time_idx"] <= max_time - 2*test_horizon]
        
        return train, val, test

    def create_timeseries_dataset(self, train: pd.DataFrame, max_encoder_length: int = 90, max_prediction_length: int = 28) -> "TimeSeriesDataSet":
        """
        Creates a TimeSeriesDataSet object for TFT.
        """
        from pytorch_forecasting import TimeSeriesDataSet
        from pytorch_forecasting.data import GroupNormalizer

        dataset = TimeSeriesDataSet(
            train,
            time_idx="time_idx",
            target="sales",
            group_ids=["id"],
            min_encoder_length=max_encoder_length // 2,
            max_encoder_length=max_encoder_length,
            min_prediction_length=1,
            max_prediction_length=max_prediction_length,
            static_categoricals=["item_id", "dept_id", "cat_id", "store_id", "state_id"],
            time_varying_known_categoricals=["day_of_week", "month", "is_holiday", "snap_CA", "snap_TX", "snap_WI"],
            time_varying_known_reals=["time_idx", "sell_price", "price_change_pct"],
            time_varying_unknown_categoricals=[],
            time_varying_unknown_reals=[
                "sales",
                "sales_lag_7", "sales_lag_14", "sales_lag_28",
                "rolling_mean_7", "rolling_mean_28",
                "rolling_std_7", "rolling_std_28",
                "rolling_max_28", "rolling_min_28",
                "sales_velocity"
            ],
            target_normalizer=GroupNormalizer(
                groups=["id"], transformation="log1p"
            ),
            add_relative_time_idx=True,
            add_target_scales=True,
            add_encoder_length=True,
        )
        return dataset

if __name__ == "__main__":
    # Example usage / Test
    pipeline = M5DataPipeline("demand-forecasting/data/raw", "demand-forecasting/data/processed")
    # For testing, load a small subset (e.g., 50 SKUs)
    raw_df = pipeline.load_data(nrows=50)
    processed_df = pipeline.engineer_features(raw_df)
    train, val, test = pipeline.split_data(processed_df)
    print(f"Train size: {len(train)}, Val size: {len(val)}, Test size: {len(test)}")
    print("Data Pipeline check complete.")
