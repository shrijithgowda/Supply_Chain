import warnings
warnings.filterwarnings('ignore')
import sys
sys.path.insert(0, 'demand-forecasting')

from src.data_pipeline import M5DataPipeline
from src.tft_model import TFTModel
from pytorch_forecasting import TimeSeriesDataSet

pipeline = M5DataPipeline('demand-forecasting/data/raw', 'demand-forecasting/data/processed')
raw = pipeline.load_data(nrows=100)
df = pipeline.engineer_features(raw)

# Filter to SKUs with enough history
sku_counts = df.groupby('id').size()
valid_skus = sku_counts[sku_counts >= 150].index
df = df[df['id'].isin(valid_skus)]

train, val, test = pipeline.split_data(df)
print(f"Train: {len(train)}, Val: {len(val)}, Test: {len(test)}")
print(f"Unique SKUs: {df['id'].nunique()}")

ts_train = pipeline.create_timeseries_dataset(train)
ts_val = TimeSeriesDataSet.from_dataset(ts_train, val)
train_loader = ts_train.to_dataloader(train=True, batch_size=32, num_workers=0)
val_loader = ts_val.to_dataloader(train=False, batch_size=32, num_workers=0)

tft = TFTModel(ts_train)
tft.build_model(hidden_size=32, attention_head_size=2)
tft.train(train_loader, val_loader, max_epochs=1)
print("SUCCESS: Training completed!")
