"""
Training loop: fits the BiLSTM model and saves weights + history.
"""

import os
import json

import numpy as np
from sklearn.model_selection import train_test_split
from tensorflow.keras.callbacks import (
    ModelCheckpoint, EarlyStopping, ReduceLROnPlateau, TensorBoard
)

import config
from src.utils import get_logger, ensure_dirs
from src.preprocessing import load_and_preprocess
from src.tokenizer_utils import (
    build_tokenizer, save_tokenizer, texts_to_padded_sequences, get_vocab_size
)
from src.model import build_bilstm_model

logger = get_logger("train_model")


def train(
    epochs: int       = config.EPOCHS,
    batch_size: int   = config.BATCH_SIZE,
    val_split: float  = config.VALIDATION_SPLIT,
    test_size: float  = config.TEST_SIZE,
    random_state: int = config.RANDOM_STATE,
):
    """
    End-to-end training pipeline.

    Returns
    -------
    history : keras History object
    X_test  : np.ndarray  — held-out test sequences
    y_test  : np.ndarray  — held-out labels
    """
    ensure_dirs()

    # ── 1. Load & preprocess ─────────────────────────────────────────────────
    df = load_and_preprocess()
    X = df["cleaned_text"].values
    y = df["label"].values

    # ── 2. Train / test split ─────────────────────────────────────────────────
    X_train_raw, X_test_raw, y_train, y_test = train_test_split(
        X, y, test_size=test_size,
        random_state=random_state, stratify=y
    )
    logger.info("Train=%d  Test=%d", len(X_train_raw), len(X_test_raw))

    # ── 3. Tokenise & pad ─────────────────────────────────────────────────────
    tokenizer = build_tokenizer(X_train_raw.tolist())
    save_tokenizer(tokenizer)

    X_train = texts_to_padded_sequences(tokenizer, X_train_raw.tolist())
    X_test  = texts_to_padded_sequences(tokenizer, X_test_raw.tolist())

    # ── 4. Build model ────────────────────────────────────────────────────────
    vocab_size = get_vocab_size(tokenizer)
    model = build_bilstm_model(vocab_size=vocab_size)

    # ── 5. Callbacks ──────────────────────────────────────────────────────────
    callbacks = [
        ModelCheckpoint(
            filepath=config.MODEL_PATH,
            monitor="val_accuracy",
            save_best_only=True,
            verbose=1,
        ),
        EarlyStopping(
            monitor="val_loss",
            patience=3,
            restore_best_weights=True,
            verbose=1,
        ),
        ReduceLROnPlateau(
            monitor="val_loss",
            factor=0.5,
            patience=2,
            min_lr=1e-6,
            verbose=1,
        ),
        TensorBoard(log_dir=config.LOG_DIR),
    ]

    # ── 6. Fit ────────────────────────────────────────────────────────────────
    logger.info("Starting training …")
    history = model.fit(
        X_train, y_train,
        epochs=epochs,
        batch_size=batch_size,
        validation_split=val_split,
        callbacks=callbacks,
        verbose=1,
    )

    # ── 7. Persist history ────────────────────────────────────────────────────
    history_path = os.path.join(config.REPORTS_DIR, "training_history.json")
    with open(history_path, "w") as f:
        json.dump({k: [float(v) for v in vals]
                   for k, vals in history.history.items()}, f, indent=2)
    logger.info("Training history saved → %s", history_path)

    logger.info("Training complete. Best model saved → %s", config.MODEL_PATH)
    return history, X_test, y_test
