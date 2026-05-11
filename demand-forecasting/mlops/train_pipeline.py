import os
import torch
import pandas as pd
from src.data_pipeline import M5DataPipeline
from src.tft_model import TFTModel
from src.baseline_model import BaselineModel
import json
from datetime import datetime

def run_pipeline():
    print(f"Starting training pipeline at {datetime.now()}")
    
    # Paths
    raw_path = "demand-forecasting/data/raw"
    processed_path = "demand-forecasting/data/processed"
    model_save_path = "demand-forecasting/models/tft_v1.pt"
    os.makedirs("demand-forecasting/models", exist_ok=True)
    os.makedirs("demand-forecasting/data/processed", exist_ok=True)
    
    # 1. Data Pipeline
    pipeline = M5DataPipeline(raw_path, processed_path)
    # Using a larger subset for a real-ish training run (e.g. 500 SKUs)
    # nrows=100 keeps startup fast — each row = 1 SKU, each SKU has ~1900 day records
    raw_df = pipeline.load_data(nrows=100)
    processed_df = pipeline.engineer_features(raw_df)
    
    # Ensure each SKU has enough history (at least 150 days)
    sku_counts = processed_df.groupby("id").size()
    valid_skus = sku_counts[sku_counts >= 150].index
    processed_df = processed_df[processed_df["id"].isin(valid_skus)]
    
    train, val, test = pipeline.split_data(processed_df)
    print(f"Dataset Split - Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
    print(f"Unique SKUs: {processed_df['id'].nunique()}")
    
    # 2. Baseline Model
    baseline = BaselineModel(pipeline)
    baseline_metrics = baseline.fit_and_evaluate(n_skus=20)
    
    # 3. TFT Training
    ts_train = pipeline.create_timeseries_dataset(train)
    # Use smaller loaders for faster execution in this demonstration
    train_loader = ts_train.to_dataloader(train=True, batch_size=64, num_workers=0)
    from pytorch_forecasting import TimeSeriesDataSet
    ts_val = TimeSeriesDataSet.from_dataset(ts_train, val)
    val_loader = ts_val.to_dataloader(train=False, batch_size=64, num_workers=0)
    
    tft = TFTModel(ts_train)
    tft.build_model(hidden_size=32, attention_head_size=2)  # compact but effective
    
    print("Starting TFT training...")
    tft.train(train_loader, val_loader, max_epochs=3)
    
    # 4. Evaluation
    print("Evaluating TFT on test set...")
    ts_test = TimeSeriesDataSet.from_dataset(ts_train, test)
    test_loader = ts_test.to_dataloader(train=False, batch_size=64, num_workers=0)
    tft_metrics = tft.evaluate(test_loader)
    
    # 5. Promotion logic
    baseline_wmape = baseline_metrics.get("avg_wmape", 1.0)
    improvement = (baseline_wmape - tft_metrics["wmape"]) / (baseline_wmape + 1e-6)
    
    print(f"Baseline WMAPE: {baseline_wmape:.4f}")
    print(f"TFT WMAPE: {tft_metrics['wmape']:.4f}")
    print(f"Improvement: {improvement*100:.2f}%")
    
    if improvement >= 0.15: # 15% improvement threshold
        print("Model meets production criteria. Promoting to production.")
        # Save model
        torch.save(tft.model.state_dict(), model_save_path)
        print(f"Model saved to {model_save_path}")
    else:
        print("Model does not meet improvement threshold. Promotion aborted.")

if __name__ == "__main__":
    run_pipeline()
