"""
Data loading and NLP text-cleaning pipeline.

Steps
-----
1. Load the raw CSV (spam.csv format: v1 = label, v2 = text).
2. Drop duplicates and null rows.
3. Encode labels  →  spam = 1, ham = 0.
4. Clean each message:
   - lower-case
   - remove URLs, email addresses, phone numbers
   - remove punctuation and digits
   - strip extra whitespace
   - remove English stopwords (NLTK)
   - simple stemming (PorterStemmer)
"""

import re
import string

import pandas as pd
import nltk
from nltk.corpus import stopwords
from nltk.stem import PorterStemmer

import config
from src.utils import get_logger

logger = get_logger("preprocessing")

# Download NLTK data once
for pkg in ("stopwords", "punkt"):
    try:
        nltk.data.find(f"tokenizers/{pkg}" if pkg == "punkt"
                       else f"corpora/{pkg}")
    except LookupError:
        nltk.download(pkg, quiet=True)

_STOP_WORDS = set(stopwords.words("english"))
_STEMMER    = PorterStemmer()


# ── Public API ─────────────────────────────────────────────────────────────────

def load_raw_data(path: str = config.DATASET_PATH) -> pd.DataFrame:
    """Load the raw spam CSV and return a clean two-column DataFrame."""
    logger.info("Loading dataset from %s", path)
    df = pd.read_csv(path, encoding="latin-1", usecols=[0, 1],
                     names=[config.LABEL_COL, config.TEXT_COL], header=0)
    df.drop_duplicates(inplace=True)
    df.dropna(inplace=True)
    logger.info("Loaded %d rows  |  spam=%d  ham=%d",
                len(df),
                (df[config.LABEL_COL] == "spam").sum(),
                (df[config.LABEL_COL] == "ham").sum())
    return df


def encode_labels(df: pd.DataFrame) -> pd.DataFrame:
    """Map 'spam' → 1 and 'ham' → 0 in a new 'label' column."""
    df = df.copy()
    df["label"] = df[config.LABEL_COL].map({"spam": 1, "ham": 0})
    return df


def clean_text(text: str) -> str:
    """
    Apply the full NLP cleaning pipeline to a single string.

    Parameters
    ----------
    text : str
        Raw email / SMS body.

    Returns
    -------
    str
        Cleaned, stemmed, stopword-free text.
    """
    text = text.lower()

    # Remove URLs
    text = re.sub(r"http\S+|www\.\S+", " ", text)
    # Remove email addresses
    text = re.sub(r"\S+@\S+", " ", text)
    # Remove phone numbers (basic pattern)
    text = re.sub(r"\b\d[\d\s\-().]{6,}\d\b", " ", text)
    # Remove punctuation
    text = text.translate(str.maketrans("", "", string.punctuation))
    # Remove digits
    text = re.sub(r"\d+", " ", text)
    # Tokenise
    tokens = text.split()
    # Remove stopwords and stem
    tokens = [_STEMMER.stem(t) for t in tokens if t not in _STOP_WORDS]
    return " ".join(tokens)


def preprocess_dataframe(df: pd.DataFrame) -> pd.DataFrame:
    """
    Apply encode_labels + clean_text to the full DataFrame.

    Returns a DataFrame with columns: v1, v2, label, cleaned_text.
    """
    logger.info("Encoding labels …")
    df = encode_labels(df)
    logger.info("Cleaning text (this may take a moment) …")
    df["cleaned_text"] = df[config.TEXT_COL].apply(clean_text)
    logger.info("Preprocessing complete.")
    return df


def load_and_preprocess() -> pd.DataFrame:
    """Convenience: load raw data then preprocess in one call."""
    return preprocess_dataframe(load_raw_data())
