from pydantic import BaseModel
from typing import List, Optional

class ForecastRequest(BaseModel):
    item_id: str
    store_id: str
    horizon_days: int = 28
    include_intervals: bool = True

class ForecastResponse(BaseModel):
    item_id: str
    store_id: str
    forecast_dates: List[str]
    point_forecast: List[float]
    lower_bound_80: Optional[List[float]] = None
    upper_bound_80: Optional[List[float]] = None
    top_drivers: List[str]

class HealthResponse(BaseModel):
    status: str
    model_version: str
    last_training_date: str
    current_wmape: float
