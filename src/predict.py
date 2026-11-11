"""
AeroLens â€“ Prediction Module
==============================
Provides utility functions for making predictions using the trained
sentiment and category models. Used by both the Streamlit app and
for batch CSV processing.

Functions
---------
load_models()           â€“ Load all saved model artifacts.
predict_single(text)    â€“ Predict sentiment, category, and priority for one review.
predict_batch(df)       â€“ Predict for an entire DataFrame of reviews.
assign_priority(s, c)   â€“ Rule-based priority assignment.
"""

import os

import joblib
import numpy as np
import pandas as pd

from preprocessing import clean_text


# ---------------------------------------------------------------------------
# Model loading
# ---------------------------------------------------------------------------

def get_models_dir() -> str:
    """Resolve the models directory relative to this script's location."""
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)
    return os.path.join(project_root, "models")


def load_models(models_dir: str | None = None) -> dict:
    """
    Load all four model artifacts from disk.

    Returns a dict with keys:
        tfidf_sentiment, sentiment_model,
        tfidf_category, category_model
    """
    if models_dir is None:
        models_dir = get_models_dir()

    artifacts = {}
    files = {
        "tfidf_sentiment": "tfidf_sentiment.joblib",
        "sentiment_model": "sentiment_model.joblib",
        "tfidf_category": "tfidf_category.joblib",
        "category_model": "category_model.joblib",
    }

    for key, filename in files.items():
        path = os.path.join(models_dir, filename)
        if not os.path.exists(path):
            raise FileNotFoundError(
                f"Model artifact not found: {path}\n"
                "Run train.py before using the prediction module."
            )
        artifacts[key] = joblib.load(path)

    return artifacts


# ---------------------------------------------------------------------------
# Priority assignment (rule-based â€“ NOT ML)
# ---------------------------------------------------------------------------

HIGH_PRIORITY_CATEGORIES = {
    "Flight Delays and Cancellations",
    "Baggage Handling",
    "Refunds and Cancellations",
    "Customer Support",
}


def assign_priority(sentiment: str, category: str) -> str:
    """
    Rule-based complaint priority.

    This is a transparent business-logic function, NOT a machine-learning
    prediction. It applies the following rules:

    - High:   Negative feedback in high-impact categories
              (delays, baggage, refunds, support).
    - Medium: Other negative feedback, or neutral feedback in
              high-impact categories.
    - Low:    Positive feedback and remaining neutral feedback.
    """
    if sentiment == "Negative" and category in HIGH_PRIORITY_CATEGORIES:
        return "High"
    elif sentiment == "Negative":
        return "Medium"
    elif sentiment == "Neutral" and category in HIGH_PRIORITY_CATEGORIES:
        return "Medium"
    else:
        return "Low"


# ---------------------------------------------------------------------------
# Single-review prediction
# ---------------------------------------------------------------------------

def predict_single(text: str, models: dict) -> dict:
    """
    Predict sentiment, category, confidence scores, and priority for
    a single review string.

    Returns a dict:
        sentiment, sentiment_confidence,
        category, category_confidence,
        priority, cleaned_text
    """
    cleaned = clean_text(text)

    # Sentiment prediction
    X_sent = models["tfidf_sentiment"].transform([cleaned])
    sent_pred = models["sentiment_model"].predict(X_sent)[0]
    sent_proba = models["sentiment_model"].predict_proba(X_sent)[0]
    sent_conf = float(np.max(sent_proba))

    # Category prediction
    X_cat = models["tfidf_category"].transform([cleaned])
    cat_pred = models["category_model"].predict(X_cat)[0]
    cat_proba = models["category_model"].predict_proba(X_cat)[0]
    cat_conf = float(np.max(cat_proba))

    # Priority (rule-based)
    priority = assign_priority(sent_pred, cat_pred)

    return {
        "cleaned_text": cleaned,
        "sentiment": sent_pred,
        "sentiment_confidence": round(sent_conf, 4),
        "category": cat_pred,
        "category_confidence": round(cat_conf, 4),
        "priority": priority,
    }


# ---------------------------------------------------------------------------
# Batch prediction
# ---------------------------------------------------------------------------

def predict_batch(
    df: pd.DataFrame,
    models: dict,
    text_column: str = "review_text",
) -> pd.DataFrame:
    """
    Run predictions on every row of a DataFrame.

    Adds columns: cleaned_text, predicted_sentiment, sentiment_confidence,
    predicted_category, category_confidence, predicted_priority.

    The original DataFrame columns are preserved.
    """
    df = df.copy()

    # Clean text
    df["cleaned_text"] = df[text_column].apply(clean_text)

    # Sentiment
    X_sent = models["tfidf_sentiment"].transform(df["cleaned_text"])
    df["predicted_sentiment"] = models["sentiment_model"].predict(X_sent)
    sent_proba = models["sentiment_model"].predict_proba(X_sent)
    df["sentiment_confidence"] = np.max(sent_proba, axis=1).round(4)

    # Category
    X_cat = models["tfidf_category"].transform(df["cleaned_text"])
    df["predicted_category"] = models["category_model"].predict(X_cat)
    cat_proba = models["category_model"].predict_proba(X_cat)
    df["category_confidence"] = np.max(cat_proba, axis=1).round(4)

    # Priority (rule-based)
    df["predicted_priority"] = df.apply(
        lambda row: assign_priority(row["predicted_sentiment"], row["predicted_category"]),
        axis=1,
    )

    return df


# ---------------------------------------------------------------------------
# Entry point (quick test)
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    print("=" * 60)
    print("AeroLens â€“ Prediction Module Test")
    print("=" * 60)

    models = load_models()
    print("âœ“ All models loaded successfully.\n")

    test_reviews = [
        "The flight was delayed by 5 hours and nobody gave us any information. Terrible experience.",
        "Amazing crew, comfortable seats, and we landed right on time. Loved it!",
        "Average experience. Nothing special but nothing went wrong either.",
        "My bag was lost and customer support has been ignoring my calls for a week.",
        "Got a great deal on the fare. Smooth check-in and boarding.",
    ]

    for review in test_reviews:
        result = predict_single(review, models)
        print(f"  Review : {review[:80]}...")
        print(f"  Sentiment  : {result['sentiment']} ({result['sentiment_confidence']:.2%})")
        print(f"  Category   : {result['category']} ({result['category_confidence']:.2%})")
        print(f"  Priority   : {result['priority']}")
        print()

