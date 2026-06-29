"""
Model evaluation: classification report, confusion matrix, and plots.
"""

import os
import json

import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import (
    classification_report, confusion_matrix,
    roc_auc_score, roc_curve
)
from tensorflow.keras.models import load_model

import config
from src.utils import get_logger

logger = get_logger("evaluate")


# ── Metrics ────────────────────────────────────────────────────────────────────

def evaluate_model(model, X_test: np.ndarray, y_test: np.ndarray,
                   threshold: float = 0.5) -> dict:
    """
    Compute and log key classification metrics.

    Returns
    -------
    dict with keys: accuracy, roc_auc, report (dict)
    """
    y_prob  = model.predict(X_test, verbose=0).flatten()
    y_pred  = (y_prob >= threshold).astype(int)

    report  = classification_report(y_test, y_pred,
                                    target_names=["ham", "spam"],
                                    output_dict=True)
    roc_auc = roc_auc_score(y_test, y_prob)
    accuracy = report["accuracy"]

    logger.info("Accuracy : %.4f", accuracy)
    logger.info("ROC-AUC  : %.4f", roc_auc)
    logger.info("\n%s", classification_report(y_test, y_pred,
                                              target_names=["ham", "spam"]))

    results = {"accuracy": accuracy, "roc_auc": roc_auc, "report": report,
               "y_prob": y_prob.tolist(), "y_pred": y_pred.tolist()}

    out = os.path.join(config.REPORTS_DIR, "evaluation_results.json")
    with open(out, "w") as f:
        json.dump({"accuracy": accuracy, "roc_auc": roc_auc,
                   "report": report}, f, indent=2)
    logger.info("Evaluation results saved → %s", out)
    return results


# ── Plots ──────────────────────────────────────────────────────────────────────

def plot_confusion_matrix(y_true: np.ndarray, y_pred: np.ndarray,
                          save: bool = True) -> None:
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(6, 5))
    sns.heatmap(cm, annot=True, fmt="d", cmap="Blues",
                xticklabels=["ham", "spam"],
                yticklabels=["ham", "spam"], ax=ax)
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title("Confusion Matrix – BiLSTM Spam Detector")
    plt.tight_layout()
    if save:
        path = os.path.join(config.PLOTS_DIR, "confusion_matrix.png")
        fig.savefig(path, dpi=150)
        logger.info("Confusion matrix saved → %s", path)
    plt.show()


def plot_training_history(history, save: bool = True) -> None:
    hist = history.history
    epochs = range(1, len(hist["loss"]) + 1)

    fig, axes = plt.subplots(1, 2, figsize=(12, 4))

    # Loss
    axes[0].plot(epochs, hist["loss"],     label="Train Loss")
    axes[0].plot(epochs, hist["val_loss"], label="Val Loss")
    axes[0].set_title("Loss"); axes[0].legend()

    # Accuracy
    axes[1].plot(epochs, hist["accuracy"],     label="Train Acc")
    axes[1].plot(epochs, hist["val_accuracy"], label="Val Acc")
    axes[1].set_title("Accuracy"); axes[1].legend()

    plt.suptitle("Training History – BiLSTM", y=1.02)
    plt.tight_layout()
    if save:
        path = os.path.join(config.PLOTS_DIR, "training_history.png")
        fig.savefig(path, dpi=150, bbox_inches="tight")
        logger.info("Training history plot saved → %s", path)
    plt.show()


def plot_roc_curve(y_true: np.ndarray, y_prob: np.ndarray,
                   save: bool = True) -> None:
    fpr, tpr, _ = roc_curve(y_true, y_prob)
    auc = roc_auc_score(y_true, y_prob)

    fig, ax = plt.subplots(figsize=(6, 5))
    ax.plot(fpr, tpr, label=f"AUC = {auc:.4f}", color="darkorange", lw=2)
    ax.plot([0, 1], [0, 1], "k--", lw=1)
    ax.set_xlabel("False Positive Rate"); ax.set_ylabel("True Positive Rate")
    ax.set_title("ROC Curve – BiLSTM Spam Detector")
    ax.legend(loc="lower right")
    plt.tight_layout()
    if save:
        path = os.path.join(config.PLOTS_DIR, "roc_curve.png")
        fig.savefig(path, dpi=150)
        logger.info("ROC curve saved → %s", path)
    plt.show()
