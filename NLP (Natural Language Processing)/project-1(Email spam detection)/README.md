# Email Spam Detection Using BiLSTM

> **97% accuracy** — Bidirectional LSTM model for robust spam / ham email classification, built with TensorFlow/Keras and served via a Flask web app.

---
![Uploading Screenshot 2026-06-30 031212.png…]()


## Project Overview

This project implements an end-to-end NLP pipeline to detect spam emails and SMS messages using a **Bidirectional LSTM (BiLSTM)** neural network. The model learns contextual word representations from both directions of a message, achieving strong performance across precision, recall, and F1-score.

**Key achievements:**
- 97% classification accuracy on the UCI SMS Spam Collection dataset
- Full NLP pipeline: tokenisation, stopword removal, stemming, and sequence padding
- Evaluated with precision, recall, F1-score, ROC-AUC, and confusion matrix
- Served as a real-time web application via Flask

---

## Project Structure

```
Spam_Email_Detection_Using_BiLSTM/
│
├── dataset/                    ← Raw data (spam.csv)
├── notebooks/
│   ├── 01_EDA.ipynb            ← Exploratory data analysis
│   ├── 02_Preprocessing.ipynb  ← NLP pipeline walkthrough
│   ├── 03_Model_Training.ipynb ← BiLSTM training
│   └── 04_Model_Evaluation.ipynb ← Metrics & plots
│
├── src/
│   ├── preprocessing.py        ← Data loading + text cleaning
│   ├── tokenizer_utils.py      ← Keras tokenizer: fit / save / load
│   ├── model.py                ← BiLSTM architecture
│   ├── train_model.py          ← Training loop + callbacks
│   ├── evaluate.py             ← Metrics, confusion matrix, ROC curve
│   ├── predictor.py            ← Inference class (single + batch)
│   └── utils.py                ← Logger, directory helpers
│
├── models/                     ← Saved model (.h5) and tokenizer (.pkl)
├── templates/index.html        ← Flask web UI
├── static/css/style.css        ← Dark-themed styling
├── static/js/script.js         ← Async predict + result display
│
├── saved_plots/                ← EDA + evaluation plots (PNG)
├── logs/                       ← Training logs + TensorBoard events
├── reports/                    ← JSON metrics, training history
├── tests/                      ← pytest unit tests
│
├── app.py                      ← Flask application
├── train.py                    ← CLI: train the model
├── predict.py                  ← CLI: predict single / batch / REPL
├── config.py                   ← All hyperparameters and paths
├── requirements.txt
├── Dockerfile
└── README.md
```

---

## Quick Start

### 1. Install dependencies

```bash
python -m venv venv
source venv/bin/activate        # Windows: venv\Scripts\activate
pip install -r requirements.txt
```

### 2. Train the model

```bash
python train.py
# Optional flags:
python train.py --epochs 15 --batch_size 64
```

This will:
- Load and preprocess `dataset/spam.csv`
- Fit the tokenizer and save it to `models/tokenizer.pkl`
- Train the BiLSTM and save the best weights to `models/bilstm_spam_model.h5`
- Save evaluation plots to `saved_plots/`

### 3. Predict from the command line

```bash
# Single message
python predict.py "Congratulations! You won a FREE iPhone. Click here now."

# From a file (one message per line)
python predict.py --file emails.txt

# Interactive REPL
python predict.py --interactive
```

### 4. Launch the web app

```bash
python app.py
```

Open `http://localhost:5000` in your browser.

### 5. Run tests

```bash
pytest tests/ -v
```

---

## Model Architecture

```
Input (max_seq_len=150)
    │
    ▼
Embedding (vocab=10000, dim=64)
    │
    ▼
SpatialDropout1D (0.2)
    │
    ▼
Bidirectional LSTM (64 units × 2 directions = 128)
    │
    ▼
Dropout (0.3)
    │
    ▼
Dense (32, ReLU)
    │
    ▼
Dense (1, Sigmoid)  →  P(spam)
```

**Training callbacks:** ModelCheckpoint · EarlyStopping · ReduceLROnPlateau · TensorBoard

---

## NLP Pipeline

```
Raw text
    ↓  lowercase
    ↓  remove URLs, emails, phone numbers
    ↓  remove punctuation & digits
    ↓  tokenise (split on whitespace)
    ↓  remove NLTK English stopwords
    ↓  Porter Stemmer
Cleaned text → Keras Tokenizer → Padded integer sequences (len=150)
```

---

## Results

| Metric    | Value  |
|-----------|--------|
| Accuracy  | ~97%   |
| Precision | ~98%   |
| Recall    | ~93%   |
| F1-score  | ~95%   |
| ROC-AUC   | ~99%   |

*(Exact values will vary slightly across runs due to random seed.)*

---

## Docker

```bash
# Build
docker build -t spam-detector .

# Train inside container (optional)
docker run --rm spam-detector python train.py

# Serve
docker run -p 5000:5000 spam-detector
```

---

## Dataset

**UCI SMS Spam Collection Dataset** — 5,574 labelled English SMS messages (747 spam, 4,827 ham).

Source: [UCI Machine Learning Repository](https://archive.ics.uci.edu/ml/datasets/sms+spam+collection)

---

## Tech Stack

| Layer        | Library / Framework          |
|--------------|------------------------------|
| Model        | TensorFlow 2 / Keras         |
| NLP          | NLTK (stopwords, stemmer)    |
| Data         | Pandas, NumPy                |
| Evaluation   | scikit-learn, Matplotlib, Seaborn |
| Web App      | Flask                        |
| Notebooks    | Jupyter                      |
| Tests        | pytest                       |
| Container    | Docker                       |

---

## Resume Reference

> *Developed a BiLSTM model in TensorFlow/Keras, achieving 97% accuracy for robust spam email classification tasks overall. Built an NLP pipeline with tokenization, stopword removal, and sequence padding for efficient email text preprocessing. Evaluated performance using precision, recall, F1-score, and confusion matrix to validate robust spam classification models.*

GitHub: [itsNIVESHTEJA/Data-science-projects](https://github.com/itsNIVESHTEJA/Data-science-projects/tree/main/NLP%20(Natural%20Language%20Processing)/project-1(Email%20spam%20detection))
