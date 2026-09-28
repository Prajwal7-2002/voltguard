"""Train the VoltGuard fault model on the PMSM dataset.

Thin CLI over ``voltguard.diagnostics.trainer.train_model``: evaluates on
held-out drive profiles and saves a new versioned model under ``models/``.

    python scripts/train.py [--data PATH] [--version v2] [--results results]
"""

import argparse
import logging

import matplotlib

matplotlib.use("Agg")

from voltguard.diagnostics.trainer import train_model  # noqa: E402

DEFAULT_DATA = "data/comprehensive_fault_training_data.csv"


def main() -> None:
    parser = argparse.ArgumentParser(description=__doc__.splitlines()[0])
    parser.add_argument("--data", default=DEFAULT_DATA, help="Training CSV with profile_id")
    parser.add_argument("--version", default=None, help="Version name (default: next vN)")
    parser.add_argument("--results", default="results", help="Directory for evaluation plots")
    args = parser.parse_args()

    logging.basicConfig(level=logging.INFO, format="%(asctime)s | %(levelname)-8s | %(message)s")
    train_model(args.data, version=args.version, results_dir=args.results)


if __name__ == "__main__":
    main()
