"""
Flask web app for real-time spam detection.

Routes
------
GET  /          → landing page (index.html)
POST /predict   → JSON { message: str } → JSON { label, confidence, is_spam }
GET  /health    → JSON health check
"""

import sys
import os
sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))

from flask import Flask, request, jsonify, render_template
from src.predictor import SpamPredictor
from src.utils import get_logger
import config

logger = get_logger("app")
app    = Flask(__name__)

# Lazy-load predictor (loaded once on first request, not at import time)
_predictor: SpamPredictor | None = None


def get_predictor() -> SpamPredictor:
    global _predictor
    if _predictor is None:
        _predictor = SpamPredictor()
    return _predictor


# ── Routes ─────────────────────────────────────────────────────────────────────

@app.route("/")
def index():
    return render_template("index.html")


@app.route("/predict", methods=["POST"])
def predict():
    data = request.get_json(silent=True)
    if not data or "message" not in data:
        return jsonify({"error": "Missing 'message' field in JSON body."}), 400

    message = str(data["message"]).strip()
    if not message:
        return jsonify({"error": "Message cannot be empty."}), 400

    try:
        result = get_predictor().predict(message)
        logger.info("Prediction: %s (%.4f) | input=%r",
                    result["label"], result["raw_prob"], message[:60])
        return jsonify(result)
    except Exception as exc:
        logger.exception("Prediction error: %s", exc)
        return jsonify({"error": "Internal server error."}), 500


@app.route("/health")
def health():
    return jsonify({"status": "ok", "model": config.MODEL_PATH})


# ── Main ───────────────────────────────────────────────────────────────────────

if __name__ == "__main__":
    app.run(
        host=config.FLASK_HOST,
        port=config.FLASK_PORT,
        debug=config.FLASK_DEBUG,
    )
