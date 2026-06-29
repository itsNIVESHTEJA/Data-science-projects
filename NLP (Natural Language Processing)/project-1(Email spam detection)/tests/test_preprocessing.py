"""
Unit tests for the preprocessing and tokenizer modules.

Run with:
    pytest tests/ -v
"""

import sys, os
sys.path.insert(0, os.path.join(os.path.dirname(__file__), ".."))

import pytest
from src.preprocessing import clean_text, load_raw_data, preprocess_dataframe
import config


# ── clean_text ─────────────────────────────────────────────────────────────────

class TestCleanText:
    def test_lowercases(self):
        assert clean_text("FREE PRIZE") == clean_text("free prize")

    def test_removes_urls(self):
        result = clean_text("Visit http://spam.com/win now")
        assert "http" not in result
        assert "spam.com" not in result

    def test_removes_punctuation(self):
        result = clean_text("Hello, world!!! How are you???")
        assert "!" not in result
        assert "," not in result

    def test_removes_digits(self):
        result = clean_text("Call 09061701461 to claim")
        assert "09061701461" not in result

    def test_empty_string(self):
        result = clean_text("")
        assert result == ""

    def test_stopwords_removed(self):
        result = clean_text("this is a test message")
        # Common stopwords like 'this', 'is', 'a' should be removed
        assert "this" not in result.split()

    def test_returns_string(self):
        assert isinstance(clean_text("some text"), str)


# ── load_raw_data ──────────────────────────────────────────────────────────────

class TestLoadRawData:
    def test_columns(self):
        df = load_raw_data()
        assert config.LABEL_COL in df.columns
        assert config.TEXT_COL  in df.columns

    def test_no_nulls(self):
        df = load_raw_data()
        assert df[config.LABEL_COL].isna().sum() == 0
        assert df[config.TEXT_COL].isna().sum()  == 0

    def test_label_values(self):
        df = load_raw_data()
        assert set(df[config.LABEL_COL].unique()) == {"spam", "ham"}

    def test_minimum_rows(self):
        df = load_raw_data()
        assert len(df) > 5000


# ── preprocess_dataframe ───────────────────────────────────────────────────────

class TestPreprocessDataframe:
    def test_label_column_created(self):
        df = load_raw_data()
        df = preprocess_dataframe(df)
        assert "label" in df.columns
        assert set(df["label"].unique()).issubset({0, 1})

    def test_cleaned_text_column_created(self):
        df = load_raw_data()
        df = preprocess_dataframe(df)
        assert "cleaned_text" in df.columns
        assert df["cleaned_text"].isna().sum() == 0
