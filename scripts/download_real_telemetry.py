"""
Script to orchestrate the manual ingestion of the PMSM Paderborn University Dataset.
"""

from pathlib import Path


def setup_real_data():
    print("=" * 60)
    print("🚀 VOLTGUARD: REAL PMSM DATASET INGESTION REQUIRED")
    print("=" * 60)
    print("Because the Paderborn University Electric Motor dataset is an academic")
    print("artifact, it cannot be curled via terminal without an API key.")
    print("\n[ACTION REQUIRED]")
    print("1. Go to: https://www.kaggle.com/datasets/wkirgsn/electric-motor-temperature")
    print("2. Download the 'measures_v2.csv' file.")
    print("3. Place it inside the 'data/' folder and rename it to:")
    print("   'comprehensive_fault_training_data.csv'")
    print("\nOnce placed, VoltGuard's backend (engine.py) will automatically")
    print("ingest the true 2Hz motor physics instead of the synthetic simulator.")

    target_path = Path("data/comprehensive_fault_training_data.csv")
    if target_path.exists():
        print(f"\n✅ SUCCESS: Dataset found at {target_path}. Ready for training.")
    else:
        print(f"\n❌ PENDING: Awaiting manual transfer to {target_path}.")


if __name__ == "__main__":
    setup_real_data()
