"""
Predictor: loads the saved model + tokenizer and exposes a predict() method
for single messages or batches.
"""

import numpy as np
from tensorflow.keras.models import load_model

import config
from src.preprocessing import clean_text
from src.tokenizer_utils import load_tokenizer, texts_to_padded_sequences
from src.utils import get_logger

logger = get_logger("predictor")


class SpamPredictor:
    """
    Singleton-style inference wrapper.

    Usage
    -----
    >>> predictor = SpamPredictor()
    >>> result = predictor.predict("Congratulations! You won a FREE iPhone!")
    >>> print(result)
    {'label': 'spam', 'confidence': 0.9987, 'is_spam': True}
    """

    def __init__(self,
                 model_path: str     = config.MODEL_PATH,
                 tokenizer_path: str = config.TOKENIZER_PATH,
                 threshold: float    = 0.5):
        logger.info("Loading model ← %s", model_path)
        self.model     = load_model(model_path)
        self.tokenizer = load_tokenizer(tokenizer_path)
        self.threshold = threshold
        logger.info("SpamPredictor ready.")

    # ── Public methods ─────────────────────────────────────────────────────────

    def predict(self, text: str) -> dict:
        """
        Classify a single email / SMS message.

        Parameters
        ----------
        text : str
            Raw (uncleaned) message text.

        Returns
        -------
        dict
            {'label': 'spam'|'ham', 'confidence': float, 'is_spam': bool}
        """
        cleaned = clean_text(text)
        seq     = texts_to_padded_sequences(self.tokenizer, [cleaned])
        prob    = float(self.model.predict(seq, verbose=0)[0][0])
        is_spam = prob >= self.threshold
        result  = {
            "label"     : "spam" if is_spam else "ham",
            "confidence": round(prob if is_spam else 1 - prob, 4),
            "is_spam"   : is_spam,
            "raw_prob"  : round(prob, 4),
        }
        logger.debug("Input: %r  →  %s (%.4f)", text[:60], result["label"], prob)
        return result

    def predict_batch(self, texts: list[str]) -> list[dict]:
        """Classify a list of messages efficiently in one forward pass."""
        cleaned = [clean_text(t) for t in texts]
        seqs    = texts_to_padded_sequences(self.tokenizer, cleaned)
        probs   = self.model.predict(seqs, verbose=0).flatten()
        return [
            {
                "label"     : "spam" if p >= self.threshold else "ham",
                "confidence": round(float(p) if p >= self.threshold
                                    else 1 - float(p), 4),
                "is_spam"   : bool(p >= self.threshold),
                "raw_prob"  : round(float(p), 4),
            }
            for p in probs
        ]
