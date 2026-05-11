import numpy as np
import pandas as pd
import logging

def calculate_psi(expected, actual, buckets=10):
    """
    Calculate Population Stability Index (PSI).
    """
    def scale_to_buckets(data, buckets):
        breakpoints = np.arange(0, buckets + 1) / buckets * 100
        return np.percentile(data, breakpoints)

    def bucketize(data, breakpoints):
        return np.histogram(data, bins=breakpoints)[0]

    breakpoints = scale_to_buckets(expected, buckets)
    expected_percents = bucketize(expected, breakpoints) / len(expected)
    actual_percents = bucketize(actual, breakpoints) / len(actual)

    # Prevent division by zero
    expected_percents = np.clip(expected_percents, 1e-6, None)
    actual_percents = np.clip(actual_percents, 1e-6, None)

    psi_val = np.sum((actual_percents - expected_percents) * np.log(actual_percents / expected_percents))
    return psi_val

def check_drift(historical_df, new_df, features):
    """
    Check for data drift using PSI on top features.
    """
    drift_detected = False
    for feature in features:
        if feature in historical_df.columns and feature in new_df.columns:
            psi = calculate_psi(historical_df[feature].dropna(), new_df[feature].dropna())
            print(f"Feature: {feature}, PSI: {psi:.4f}")
            if psi > 0.2:
                print(f"WARNING: DRIFT_DETECTED for feature {feature} (PSI > 0.2)")
                drift_detected = True
    return drift_detected

if __name__ == "__main__":
    # Mock check
    hist = pd.DataFrame({"sales": np.random.normal(10, 2, 1000)})
    new = pd.DataFrame({"sales": np.random.normal(12, 3, 1000)}) # Slight drift
    check_drift(hist, new, ["sales"])
