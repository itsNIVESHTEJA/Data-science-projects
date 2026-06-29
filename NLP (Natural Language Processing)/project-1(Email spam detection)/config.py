"""
Central configuration for Spam Email Detection Using BiLSTM.
All hyperparameters, paths, and settings are defined here.
"""

import os

# ── Paths ──────────────────────────────────────────────────────────────────────
BASE_DIR        = os.path.dirname(os.path.abspath(__file__))
DATASET_PATH    = os.path.join(BASE_DIR, "dataset", "spam.csv")
MODEL_DIR       = os.path.join(BASE_DIR, "models")
TOKENIZER_PATH  = os.path.join(MODEL_DIR, "tokenizer.pkl")
MODEL_PATH      = os.path.join(MODEL_DIR, "bilstm_spam_model.h5")
LOG_DIR         = os.path.join(BASE_DIR, "logs")
PLOTS_DIR       = os.path.join(BASE_DIR, "saved_plots")
REPORTS_DIR     = os.path.join(BASE_DIR, "reports")

# ── Data ───────────────────────────────────────────────────────────────────────
LABEL_COL       = "v1"
TEXT_COL        = "v2"
TEST_SIZE       = 0.2
RANDOM_STATE    = 42

# ── Text Preprocessing ─────────────────────────────────────────────────────────
MAX_VOCAB_SIZE  = 10_000   # Maximum vocabulary size for tokenizer
MAX_SEQ_LEN     = 150      # Pad / truncate sequences to this length
OOV_TOKEN       = "<OOV>"  # Out-of-vocabulary token

# ── Model Architecture ─────────────────────────────────────────────────────────
EMBEDDING_DIM   = 64       # Word embedding dimensions
BILSTM_UNITS    = 64       # Units per LSTM direction (total hidden = 2 × this)
DROPOUT_RATE    = 0.3      # Dropout after BiLSTM layer
DENSE_UNITS     = 32       # Units in the intermediate Dense layer

# ── Training ───────────────────────────────────────────────────────────────────
BATCH_SIZE      = 32
EPOCHS          = 10
LEARNING_RATE   = 1e-3
VALIDATION_SPLIT = 0.1     # Fraction of training data used for validation

# ── Flask App ──────────────────────────────────────────────────────────────────
FLASK_HOST      = "0.0.0.0"
FLASK_PORT      = 5000
FLASK_DEBUG     = False
