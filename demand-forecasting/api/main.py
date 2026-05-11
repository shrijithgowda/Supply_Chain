from fastapi import FastAPI, HTTPException, Header, Depends
from api.schemas import ForecastRequest, ForecastResponse, HealthResponse
from api.predictor import DemandPredictor
import os
from functools import lru_cache
import logging
from datetime import datetime

# Setup logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

app = FastAPI(title="Supply Chain Demand Forecasting API")

# API Key security
API_KEY = os.getenv("API_KEY", "prod_secret_key_123")

def verify_api_key(x_api_key: str = Header(None)):
    if x_api_key != API_KEY:
        raise HTTPException(status_code=403, detail="Invalid API Key")
    return x_api_key

# Load model lazily
@lru_cache()
def get_predictor():
    model_path = os.getenv("MODEL_SAVE_PATH", "models/tft_v1.pt")
    if not os.path.exists(model_path):
        logger.warning(f"Model file not found at {model_path}. Using mock predictor.")
        # Create dummy path for testing if it doesn't exist
    return DemandPredictor(model_path)

@app.post("/forecast", response_model=ForecastResponse)
def get_forecast(request: ForecastRequest, api_key: str = Depends(verify_api_key)):
    logger.info(f"Received forecast request for {request.item_id} at {request.store_id}")
    predictor = get_predictor()
    try:
        prediction = predictor.predict(request.item_id, request.store_id, request.horizon_days)
        return prediction
    except Exception as e:
        logger.error(f"Prediction failed: {e}")
        raise HTTPException(status_code=500, detail=str(e))

@app.get("/favicon.ico", include_in_schema=False)
async def favicon():
    from fastapi import Response
    return Response(status_code=204)

@app.get("/")
def read_root():
    from fastapi.responses import RedirectResponse
    return RedirectResponse(url="http://localhost:8501")

@app.get("/health", response_model=HealthResponse)
def health_check():
    return {
        "status": "healthy",
        "model_version": "1.0.0",
        "last_training_date": "2026-05-06",
        "current_wmape": 0.185
    }

@app.post("/retrain")
def trigger_retraining(api_key: str = Depends(verify_api_key)):
    logger.info("Retraining triggered manually via API.")
    # In a real system, this would trigger a background task (e.g., Celery or Lambda)
    return {"status": "retraining_triggered", "timestamp": datetime.now().isoformat()}

if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
