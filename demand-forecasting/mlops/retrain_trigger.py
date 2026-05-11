import os
import time
from datetime import datetime

def lambda_handler(event, context):
    """
    AWS Lambda handler to trigger retraining.
    """
    print("Evaluating retraining conditions...")
    
    # Condition A: New data arrived (simplified check)
    new_data_path = os.getenv("RAW_DATA_PATH", "demand-forecasting/data/raw/sales_train_validation.csv")
    if os.path.exists(new_data_path):
        last_modified = os.path.getmtime(new_data_path)
        if (time.time() - last_modified) < 86400: # New data in last 24 hours
            print("New data detected. Triggering retrain.")
            return True

    # Condition B: Accuracy drop (mock check)
    current_wmape = 0.22 # Example: current accuracy dropped
    threshold = 0.20
    if current_wmape > threshold:
        print(f"Accuracy ({current_wmape}) below threshold ({threshold}). Triggering retrain.")
        return True

    print("No retraining criteria met.")
    return False

if __name__ == "__main__":
    lambda_handler(None, None)
