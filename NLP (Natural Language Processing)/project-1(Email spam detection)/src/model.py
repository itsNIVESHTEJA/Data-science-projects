"""
BiLSTM model architecture for binary spam classification.

Architecture
------------
Embedding → SpatialDropout1D → Bidirectional(LSTM) → Dropout → Dense → Output
"""

from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import (
    Embedding, SpatialDropout1D,
    Bidirectional, LSTM,
    Dropout, Dense
)
from tensorflow.keras.optimizers import Adam

import config
from src.utils import get_logger

logger = get_logger("model")


def build_bilstm_model(vocab_size: int,
                       embedding_dim: int  = config.EMBEDDING_DIM,
                       bilstm_units: int   = config.BILSTM_UNITS,
                       dropout_rate: float = config.DROPOUT_RATE,
                       dense_units: int    = config.DENSE_UNITS,
                       max_seq_len: int    = config.MAX_SEQ_LEN,
                       learning_rate: float = config.LEARNING_RATE
                       ) -> Sequential:
    """
    Build and compile the BiLSTM spam-detection model.

    Parameters
    ----------
    vocab_size     : int    Vocabulary size (tokenizer word index + 1).
    embedding_dim  : int    Dimensionality of the embedding layer.
    bilstm_units   : int    LSTM units per direction.
    dropout_rate   : float  Dropout probability.
    dense_units    : int    Units in the hidden Dense layer.
    max_seq_len    : int    Input sequence length.
    learning_rate  : float  Adam optimizer learning rate.

    Returns
    -------
    keras.Sequential  — compiled model ready for training.
    """
    logger.info(
        "Building BiLSTM model  vocab=%d  emb=%d  lstm_units=%d  lr=%.4f",
        vocab_size, embedding_dim, bilstm_units, learning_rate
    )

    model = Sequential([
        # ── Embedding ────────────────────────────────────────────────────────
        Embedding(input_dim=vocab_size,
                  output_dim=embedding_dim,
                  input_length=max_seq_len,
                  name="embedding"),

        # ── Regularisation after embedding ───────────────────────────────────
        SpatialDropout1D(rate=0.2, name="spatial_dropout"),

        # ── Bidirectional LSTM ───────────────────────────────────────────────
        Bidirectional(
            LSTM(units=bilstm_units, return_sequences=False),
            name="bilstm"
        ),

        # ── Fully-connected head ─────────────────────────────────────────────
        Dropout(rate=dropout_rate, name="dropout"),
        Dense(units=dense_units, activation="relu", name="dense"),
        Dense(units=1, activation="sigmoid", name="output"),
    ], name="BiLSTM_SpamDetector")

    model.compile(
        optimizer=Adam(learning_rate=learning_rate),
        loss="binary_crossentropy",
        metrics=["accuracy"],
    )

    model.summary(print_fn=logger.info)
    return model
