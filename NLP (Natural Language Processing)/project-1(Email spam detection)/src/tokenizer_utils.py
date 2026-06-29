"""
Tokenizer utilities: fit, save, load, and apply the Keras tokenizer
for converting cleaned text into padded integer sequences.
"""

import pickle

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
import numpy as np

import config
from src.utils import get_logger

logger = get_logger("tokenizer_utils")


# ── Tokenizer lifecycle ────────────────────────────────────────────────────────

def build_tokenizer(texts: list[str]) -> Tokenizer:
    """
    Fit a new Keras Tokenizer on the provided texts.

    Parameters
    ----------
    texts : list[str]
        Cleaned training texts.

    Returns
    -------
    Tokenizer
        Fitted tokenizer.
    """
    logger.info("Fitting tokenizer on %d texts  (vocab_size=%d, oov='%s') …",
                len(texts), config.MAX_VOCAB_SIZE, config.OOV_TOKEN)
    tok = Tokenizer(num_words=config.MAX_VOCAB_SIZE,
                    oov_token=config.OOV_TOKEN)
    tok.fit_on_texts(texts)
    vocab_size = len(tok.word_index) + 1
    logger.info("Vocabulary size (incl. OOV+PAD): %d", vocab_size)
    return tok


def save_tokenizer(tokenizer: Tokenizer,
                   path: str = config.TOKENIZER_PATH) -> None:
    """Persist the fitted tokenizer to disk via pickle."""
    with open(path, "wb") as f:
        pickle.dump(tokenizer, f)
    logger.info("Tokenizer saved → %s", path)


def load_tokenizer(path: str = config.TOKENIZER_PATH) -> Tokenizer:
    """Load a previously saved tokenizer from disk."""
    with open(path, "rb") as f:
        tok = pickle.load(f)
    logger.info("Tokenizer loaded ← %s", path)
    return tok


# ── Sequence helpers ───────────────────────────────────────────────────────────

def texts_to_padded_sequences(tokenizer: Tokenizer,
                               texts: list[str],
                               maxlen: int = config.MAX_SEQ_LEN) -> np.ndarray:
    """
    Convert a list of cleaned texts to a padded numpy array.

    Parameters
    ----------
    tokenizer : Tokenizer
        A *fitted* Keras Tokenizer.
    texts : list[str]
        Raw or cleaned texts.
    maxlen : int
        Sequence length to pad / truncate to.

    Returns
    -------
    np.ndarray  shape (len(texts), maxlen)
    """
    seqs = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(seqs, maxlen=maxlen,
                           padding="post", truncating="post")
    logger.debug("Sequences shape: %s", padded.shape)
    return padded


def get_vocab_size(tokenizer: Tokenizer) -> int:
    """Return the effective vocabulary size capped at MAX_VOCAB_SIZE."""
    return min(len(tokenizer.word_index) + 1, config.MAX_VOCAB_SIZE)
