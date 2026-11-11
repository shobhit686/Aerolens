"""
AeroLens â€“ Model Evaluation Module
====================================
Evaluates both trained classifiers (sentiment & category) on the held-out
test set using:
  - Accuracy
  - Precision  (macro & weighted)
  - Recall     (macro & weighted)
  - F1 Score   (macro & weighted)
  - Confusion Matrix

Results are printed to the console and saved as JSON to  outputs/ .

DISCLAIMER: These metrics are computed on synthetic data and do NOT
represent real-world model performance.
"""

import json
import os

import joblib
import numpy as np
import pandas as pd
from sklearn.metrics import (
    accuracy_score,
    classification_report,
    confusion_matrix,
    f1_score,
    precision_score,
    recall_score,
)


def evaluate_model(
    model_path: str,
    tfidf_path: str,
    test_texts: pd.Series,
    test_labels: pd.Series,
    model_name: str,
) -> dict:
    """
    Load a trained model + vectoriser, predict on test data, and compute
    evaluation metrics.

    Returns a dictionary with all metrics and the confusion matrix.
    """
    tfidf = joblib.load(tfidf_path)
    clf = joblib.load(model_path)

    # Transform test data using the ALREADY-FITTED vectoriser (no re-fitting)
    X_test = tfidf.transform(test_texts)
    y_pred = clf.predict(X_test)

    # Compute metrics
    acc = accuracy_score(test_labels, y_pred)
    prec_macro = precision_score(test_labels, y_pred, average="macro", zero_division=0)
    prec_weighted = precision_score(test_labels, y_pred, average="weighted", zero_division=0)
    rec_macro = recall_score(test_labels, y_pred, average="macro", zero_division=0)
    rec_weighted = recall_score(test_labels, y_pred, average="weighted", zero_division=0)
    f1_macro = f1_score(test_labels, y_pred, average="macro", zero_division=0)
    f1_weighted = f1_score(test_labels, y_pred, average="weighted", zero_division=0)

    # Confusion matrix
    labels = sorted(test_labels.unique())
    cm = confusion_matrix(test_labels, y_pred, labels=labels)

    # Classification report (per-class breakdown)
    report = classification_report(test_labels, y_pred, output_dict=True, zero_division=0)

    results = {
        "model": model_name,
        "accuracy": round(acc, 4),
        "precision_macro": round(prec_macro, 4),
        "precision_weighted": round(prec_weighted, 4),
        "recall_macro": round(rec_macro, 4),
        "recall_weighted": round(rec_weighted, 4),
        "f1_macro": round(f1_macro, 4),
        "f1_weighted": round(f1_weighted, 4),
        "confusion_matrix": cm.tolist(),
        "labels": labels,
        "classification_report": report,
    }

    return results


def print_results(results: dict) -> None:
    """Pretty-print evaluation results to the console."""
    print(f"\n{'â”€' * 50}")
    print(f"  {results['model']}")
    print(f"{'â”€' * 50}")
    print(f"  Accuracy          : {results['accuracy']:.4f}")
    print(f"  Precision (macro) : {results['precision_macro']:.4f}")
    print(f"  Precision (wt.)   : {results['precision_weighted']:.4f}")
    print(f"  Recall (macro)    : {results['recall_macro']:.4f}")
    print(f"  Recall (wt.)      : {results['recall_weighted']:.4f}")
    print(f"  F1 Macro          : {results['f1_macro']:.4f}")
    print(f"  F1 Weighted       : {results['f1_weighted']:.4f}")
    print(f"\n  Confusion Matrix (rows=actual, cols=predicted):")
    labels = results["labels"]
    cm = np.array(results["confusion_matrix"])
    # Header
    max_label_len = max(len(str(l)) for l in labels)
    header = " " * (max_label_len + 4) + "  ".join(f"{l[:8]:>8}" for l in labels)
    print(f"  {header}")
    for i, label in enumerate(labels):
        row = "  ".join(f"{v:>8}" for v in cm[i])
        print(f"  {label:<{max_label_len + 4}}{row}")
    print()


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------

if __name__ == "__main__":
    script_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.dirname(script_dir)

    models_dir = os.path.join(project_root, "models")
    processed_dir = os.path.join(project_root, "data", "processed")
    outputs_dir = os.path.join(project_root, "outputs")
    os.makedirs(outputs_dir, exist_ok=True)

    test_path = os.path.join(processed_dir, "test.csv")

    print("=" * 60)
    print("AeroLens â€“ Model Evaluation")
    print("=" * 60)

    if not os.path.exists(test_path):
        print(f"\nâœ— Test data not found at {test_path}")
        print("  Run preprocessing.py first.\n")
        raise SystemExit(1)

    test_df = pd.read_csv(test_path)
    print(f"\n  Test records loaded: {len(test_df)}")

    all_results = {}

    # ------------------------------------------------------------------
    # Evaluate Sentiment Model
    # ------------------------------------------------------------------
    sent_model_path = os.path.join(models_dir, "sentiment_model.joblib")
    sent_tfidf_path = os.path.join(models_dir, "tfidf_sentiment.joblib")

    if os.path.exists(sent_model_path) and os.path.exists(sent_tfidf_path):
        sent_results = evaluate_model(
            model_path=sent_model_path,
            tfidf_path=sent_tfidf_path,
            test_texts=test_df["cleaned_text"],
            test_labels=test_df["sentiment"],
            model_name="Sentiment Classifier",
        )
        print_results(sent_results)
        all_results["sentiment"] = sent_results
    else:
        print("\nâœ— Sentiment model files not found. Run train.py first.")

    # ------------------------------------------------------------------
    # Evaluate Category Model
    # ------------------------------------------------------------------
    cat_model_path = os.path.join(models_dir, "category_model.joblib")
    cat_tfidf_path = os.path.join(models_dir, "tfidf_category.joblib")

    if os.path.exists(cat_model_path) and os.path.exists(cat_tfidf_path):
        cat_results = evaluate_model(
            model_path=cat_model_path,
            tfidf_path=cat_tfidf_path,
            test_texts=test_df["cleaned_text"],
            test_labels=test_df["issue_category"],
            model_name="Category Classifier",
        )
        print_results(cat_results)
        all_results["category"] = cat_results
    else:
        print("\nâœ— Category model files not found. Run train.py first.")

    # ------------------------------------------------------------------
    # Save results to JSON
    # ------------------------------------------------------------------
    output_path = os.path.join(outputs_dir, "evaluation_results.json")
    with open(output_path, "w") as f:
        json.dump(all_results, f, indent=2)
    print(f"âœ“ Evaluation results saved to: {output_path}")

    print("\nDISCLAIMER: These metrics are computed on synthetic data and")
    print("do NOT represent real-world model performance.\n")

