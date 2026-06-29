"""
Entry point: train the BiLSTM spam-detection model.

Usage
-----
    python train.py
    python train.py --epochs 15 --batch_size 64
"""

import argparse
import sys
import os

# Allow imports from project root
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

import config
from src.train_model import train
from src.evaluate import (
    evaluate_model, plot_confusion_matrix,
    plot_training_history, plot_roc_curve
)
from src.utils import get_logger
import numpy as np

logger = get_logger("train_entry")


def parse_args():
    p = argparse.ArgumentParser(description="Train BiLSTM Spam Detector")
    p.add_argument("--epochs",      type=int,   default=config.EPOCHS)
    p.add_argument("--batch_size",  type=int,   default=config.BATCH_SIZE)
    p.add_argument("--test_size",   type=float, default=config.TEST_SIZE)
    p.add_argument("--no_plots",    action="store_true",
                   help="Skip saving evaluation plots")
    return p.parse_args()


def main():
    args = parse_args()
    logger.info("=== Training started ===")
    logger.info("epochs=%d  batch_size=%d  test_size=%.2f",
                args.epochs, args.batch_size, args.test_size)

    history, X_test, y_test = train(
        epochs=args.epochs,
        batch_size=args.batch_size,
        test_size=args.test_size,
    )

    # ── Evaluate on held-out test set ─────────────────────────────────────────
    from tensorflow.keras.models import load_model
    model   = load_model(config.MODEL_PATH)
    results = evaluate_model(model, X_test, y_test)

    if not args.no_plots:
        y_pred = np.array(results["y_pred"])
        y_prob = np.array(results["y_prob"])
        plot_confusion_matrix(y_test, y_pred)
        plot_training_history(history)
        plot_roc_curve(y_test, y_prob)

    logger.info("=== Training pipeline complete ===")
    logger.info("Accuracy : %.4f  |  ROC-AUC : %.4f",
                results["accuracy"], results["roc_auc"])


if __name__ == "__main__":
    main()
