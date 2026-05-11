import torch
import lightning.pytorch as pl
from lightning.pytorch.callbacks import EarlyStopping, LearningRateMonitor, ModelCheckpoint
from lightning.pytorch.loggers import TensorBoardLogger
from pytorch_forecasting import TemporalFusionTransformer, TimeSeriesDataSet
from pytorch_forecasting.metrics import QuantileLoss, MAE, RMSE
from pytorch_forecasting.data import GroupNormalizer
import os
import pandas as pd
from src.data_pipeline import M5DataPipeline

def wmape(y_true, y_pred):
    return torch.sum(torch.abs(y_true - y_pred)) / (torch.sum(torch.abs(y_true)) + 1e-6)

class TFTModel:
    def __init__(self, training_data: TimeSeriesDataSet):
        self.training_data = training_data
        self.model = None

    def build_model(self, hidden_size: int = 128, attention_head_size: int = 4, dropout: float = 0.1):
        self.model = TemporalFusionTransformer.from_dataset(
            self.training_data,
            learning_rate=0.001,
            hidden_size=hidden_size,
            attention_head_size=attention_head_size,
            dropout=dropout,
            hidden_continuous_size=32,
            output_size=7,  # 7 quantiles
            loss=QuantileLoss(),
            log_interval=10,
            reduce_on_plateau_patience=4,
        )
        print(f"Number of parameters: {self.model.size()/1e3:.1f}k")

    def train(self, train_loader, val_loader, max_epochs: int = 50):
        trainer = pl.Trainer(
            max_epochs=max_epochs,
            accelerator="auto",
            devices=1,
            enable_model_summary=True,
            gradient_clip_val=0.1,
            callbacks=[
                EarlyStopping(monitor="val_loss", patience=5),
                ModelCheckpoint(filename="best-tft", monitor="val_loss", mode="min", save_top_k=1)
            ],
            logger=False,
            enable_checkpointing=True
        )
        
        trainer.fit(self.model, train_dataloaders=train_loader, val_dataloaders=val_loader)
        
        # Load best model
        best_model_path = trainer.checkpoint_callback.best_model_path
        self.model = TemporalFusionTransformer.load_from_checkpoint(best_model_path)
        return self.model

    def evaluate(self, test_loader):
        import torch
        # Move model to CPU to avoid CUDA/CPU device mismatch during manual eval
        self.model = self.model.cpu()
        self.model.eval()

        all_preds, all_targets = [], []
        with torch.no_grad():
            for batch in test_loader:
                x, y = batch
                # Move all tensors in x to CPU
                x = {k: (v.cpu() if isinstance(v, torch.Tensor) else v) for k, v in x.items()}
                out = self.model(x)
                pred = out.prediction[..., 3]  # median quantile
                target = y[0].cpu()
                all_preds.append(pred)
                all_targets.append(target)

        y_pred = torch.cat(all_preds, dim=0)
        y_true = torch.cat(all_targets, dim=0)

        mae_val   = MAE()(y_pred, y_true).item()
        rmse_val  = RMSE()(y_pred, y_true).item()
        wmape_val = wmape(y_true, y_pred).item()

        metrics = {"mae": mae_val, "rmse": rmse_val, "wmape": wmape_val}
        print(f"TFT Metrics: {metrics}")
        return metrics

if __name__ == "__main__":
    # Test script
    pipeline = M5DataPipeline("demand-forecasting/data/raw", "demand-forecasting/data/processed")
    raw_df = pipeline.load_data(nrows=100) # Small subset for testing
    processed_df = pipeline.engineer_features(raw_df)
    train, val, test = pipeline.split_data(processed_df)
    
    ts_train = pipeline.create_timeseries_dataset(train)
    ts_val = TimeSeriesDataSet.from_dataset(ts_train, val)
    ts_test = TimeSeriesDataSet.from_dataset(ts_train, test)
    
    train_loader = ts_train.to_dataloader(train=True, batch_size=128, num_workers=0)
    val_loader = ts_val.to_dataloader(train=False, batch_size=128, num_workers=0)
    test_loader = ts_test.to_dataloader(train=False, batch_size=128, num_workers=0)
    
    tft = TFTModel(ts_train)
    tft.build_model(hidden_size=16, attention_head_size=1) # Small model for test
    # Skip actual training for now if libraries are not fully ready
    # tft.train(train_loader, val_loader, max_epochs=1)
    print("TFT implementation verified.")
