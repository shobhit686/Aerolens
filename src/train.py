"""
AeroLens â€“ Model Training Module
=================================
Trains two Logistic Regression classifiers:
  Model 1 : Sentiment Classification  (Positive / Negative / Neutral)
  Model 2 : Complaint Categorisation   (8 issue categories)

Feature extraction uses TF-IDF vectorisation with unigrams + bigrams.
The TF-IDF vectoriser is fitted ONLY on training data to prevent data leakage.

Artifacts saved to  models/ :
  sentiment_model.joblib       â€“ trained sentiment classifier
  category_model.joblib        â€“ trained category classifier
  tfidf_sentiment.joblib       â€“ TF-IDF vectoriser for sentiment
  tfidf_category.joblib        â€“ TF-IDF vectoriser for category

DISCLAIMER: Model performance on synthetic data does not establish
real-world accuracy. These models are trained exclusively on
computer-generated reviews and would require retraining on genuine
customer feedback before production deployment.
"""

import os

import joblib
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.linear_model import LogisticRegression


# ---------------------------------------------------------------------------
# Training helpers
# ---------------------------------------------------------------------------

def train_model(
    train_texts: pd.Series,
    train_labels: pd.Series,
    model_name: str,
    max_features: int = 5000,
    ngram_range: tuple = (1, 2),
    max_iter: int = 1000,
    C: float = 1.0,
    verbose: bool = True,
) -> tuple[TfidfVectorizer, LogisticRegression]:
    """
    Train a TF-IDF + Logistic Regression pipeline.

    Parameters
    ----------
    train_texts   : Series of cleaned review strings.
    train_labels  : Series of target labels (sentiment or category).
    model_name    : Human-readable name for logging.
    max_features  : Maximum vocabulary size for TF-IDF.
    ngram_range   : Tuple (min_n, max_n) for n-gram extraction.
    max_iter      : Maximum iterations for Logistic Regression solver.
    C             : Inverse regularisation strength (higher = less regularisation).
    verbose       : Print training details.

    Returns
    -------
    (tfidf_vectoriser, trained_classifier)

    Technical explanation
    ---------------------
    TF-IDF (Term Frequency â€“ Inverse Document Frequency) converts raw text
    into numerical feature vectors. Each word (or bigram) gets a weight that
    reflects how important it is to a specific document relative to the entire
    corpus. Common words like "the" get low weights; distinctive words like
    "delayed" or "refund" get high weights.

    Logistic Regression then learns a linear decision boundary in this
    high-dimensional feature space to classify reviews.
    """
    if verbose:
        print(f"\n  Training: {model_name}")
        print(f"    Samples       : {len(train_texts)}")
        print(f"    Classes       : {train_labels.nunique()}")
        print(f"    TF-IDF features: max {max_features}, n-grams {ngram_range}")

    # --- TF-IDF Vectorisation (fit on training data ONLY) ---
    tfidf = TfidfVectorizer(
        max_features=max_features,
        ngram_range=ngram_range,
        stop_words="english",
        sublinear_tf=True,      # apply log normalisation to term frequencies
    )
    X_train = tfidf.fit_transform(train_texts)

    if verbose:
        print(f"    Vocabulary size: {len(tfidf.vocabulary_)}")

    # --- Logistic Regression ---
    clf = LogisticRegression(
        C=C,
        max_iter=max_iter,
        solver="lbfgs",
        multi_class="multinomial",
        class_weight="balanced",   # handle class imbalance
        random_state=42,
        n_jobs=-1,
    )
    clf.fit(X_train, train_labels)

    if verbose:
        train_acc = clf.score(X_train, train_labels)
        print(f"    Training accuracy: {train_acc:.4f}")

    return tfidf, clf


def save_artifacts(
    tfidf: TfidfVectorizer,
    clf: LogisticRegression,
    models_dir: str,
    tfidf_name: str,
    model_name: str,
) -> None:
    """Persist the vectoriser and classifier to disk using joblib."""
    os.makedirs(models_dir, exist_ok=True)
    joblib.dump(tfidf, os.path.join(models_dir, tfidf_name))
    joblib.dump(clf, os.path.join(models_dir, model_name))


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    processed_dir = os.path.join(project_root, "data", "processed")
    models_dir = os.path.join(project_root, "models")

    train_path = os.path.join(processed_dir, "train.csv")

    print("=" * 60)
    print("AeroLens â€“ Model Training")
    print("=" * 60)

    if not os.path.exists(train_path):
        print(f"\nâœ— Training data not found at {train_path}")
        print("  Run preprocessing.py first.\n")
        raise SystemExit(1)

    train_df = pd.read_csv(train_path)
    print(f"\n  Training records loaded: {len(train_df)}")

    # ------------------------------------------------------------------
    # Model 1: Sentiment Classification
    # ------------------------------------------------------------------
    tfidf_sent, clf_sent = train_model(
        train_texts=train_df["cleaned_text"],
        train_labels=train_df["sentiment"],
        model_name="Sentiment Classifier",
    )
    save_artifacts(
        tfidf_sent, clf_sent, models_dir,
        tfidf_name="tfidf_sentiment.joblib",
        model_name="sentiment_model.joblib",
    )
    print(f"  âœ“ Sentiment model saved to {models_dir}")

    # ------------------------------------------------------------------
    # Model 2: Complaint Categorisation
    # ------------------------------------------------------------------
    tfidf_cat, clf_cat = train_model(
        train_texts=train_df["cleaned_text"],
        train_labels=train_df["issue_category"],
        model_name="Category Classifier",
    )
    save_artifacts(
        tfidf_cat, clf_cat, models_dir,
        tfidf_name="tfidf_category.joblib",
        model_name="category_model.joblib",
    )
    print(f"  âœ“ Category model saved to {models_dir}")

    print("\n" + "=" * 60)
    print("Training complete. Run evaluate.py to assess model performance.")
    print("=" * 60 + "\n")

