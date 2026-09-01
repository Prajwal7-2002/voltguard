"""Automatic model retraining daemon script.

Checks the DriftDetector periodically against a buffer of 
newly accumulated data. If data drift drops below thresholds, 
automatically kicks off `train_model()`.
"""

import time
import logging
from pathlib import Path

from voltguard.diagnostics.drift import DriftDetector
from voltguard.diagnostics.trainer import train_model

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger("AutoRetrain")

def run_retrain_loop(buffer_path: str = "data/recent_drift_buffer.csv"):
    """Monitors the buffer and triggers retraining if drifted."""
    # Check if we have enough accumulated data to even bother
    if not Path(buffer_path).exists():
        logger.info(f"No recent data buffer at {buffer_path} to check drift against.")
        return

    detector = DriftDetector("latest")
    
    import pandas as pd
    logger.info(f"Loading incoming data chunk from {buffer_path}")
    df_incoming = pd.read_csv(buffer_path)
    
    if len(df_incoming) < 50:
        logger.info("Not enough data in buffer to statistically measure drift. Passing.")
        return
        
    drift_report = detector.calculate_drift_score(df_incoming)
    logger.info(f"Drift check complete. Score: {drift_report['drift_score']:.2f}")
    
    if drift_report["threshold_exceeded"]:
        logger.warning(f"🚨 DRIFT DETECTED (Score {drift_report['drift_score']:.2f}). Triggering Auto-Retrain!")
        
        # Merge buffer into main training dataset if keeping context
        # (In reality, we might rotate them or re-balance, here we just invoke training on the fresh accumulated dataset context or full dataset)
        
        # Trigger retraining
        logger.info("Invoking training pipeline...")
        # Assume data/train.csv was somehow appended with buffer_path by an ingest worker
        train_model(data_path="data/train.csv")
        logger.info("✅ Auto-Retraining completed. System is utilizing new model.")
    else:
        logger.info("Data distribution is nominal. No retraining required.")

if __name__ == "__main__":
    run_retrain_loop()
