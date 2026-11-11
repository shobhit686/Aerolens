# AeroLens — Airline Customer Experience Intelligence Platform

> **Turning customer conversations into actionable insights.**

AeroLens is a lightweight AI/ML-powered platform that helps airline customer-experience teams consolidate, classify, and analyse customer feedback across channels. It uses NLP to detect sentiment, categorise complaints, and prioritise cases requiring immediate attention.

---

## Product Overview

Airlines receive large volumes of unstructured customer feedback from social media, surveys, support tickets, and review platforms. AeroLens consolidates this feedback and answers questions such as:

- What are customers most unhappy about?
- Which service areas generate the most complaints?
- What proportion of feedback is positive, neutral, or negative?
- Which routes are experiencing recurring service issues?
- Which complaints require immediate attention?
- Are negative-feedback levels increasing over time?

**Target users:** Airline Customer Experience Managers, Service Quality Analysts, and Customer Support Teams.

---

## Airlines Included

The current dataset covers five Indian airlines:

| # | Airline |
|---|---------|
| 1 | IndiGo |
| 2 | Air India |
| 3 | Air India Express |
| 4 | Akasa Air |
| 5 | SpiceJet |

> **Important:** The dataset is **100% synthetic** (computer-generated). Sentiment scores, airline comparisons, and performance indicators do **not** represent actual airline performance and must not be cited as real data.

---

## Features

### Data Ingestion
- Upload customer-feedback CSV files
- Preview and validate uploaded data
- Handle missing values and duplicate records
- Clean and normalise text
- Downloadable CSV template

### NLP/ML Engine

| Model | Task | Algorithm |
|-------|------|-----------|
| Sentiment Classifier | Classify feedback as Positive, Negative, or Neutral | TF-IDF (unigrams + bigrams) + Logistic Regression |
| Category Classifier | Classify into 8 complaint categories | TF-IDF (unigrams + bigrams) + Logistic Regression |
| Priority Assignment | Flag complaints by urgency (High / Medium / Low) | **Rule-based** (not ML) |

**Complaint Categories:**
1. Flight Delays and Cancellations
2. Baggage Handling
3. Refunds and Cancellations
4. Customer Support
5. Staff Behaviour
6. Check-in and Boarding
7. Onboard Experience
8. Fares and Additional Charges

**Priority Rules (transparent business logic, not ML prediction):**
- **High:** Negative feedback involving delays, baggage, refunds, or customer support
- **Medium:** Other negative feedback or neutral feedback in high-impact categories
- **Low:** Positive feedback and remaining neutral feedback

### Analytics Dashboard

- **Executive Overview** — KPIs, sentiment distribution, negative-feedback trends, priority breakdown
- **Complaint Intelligence** — Category volumes, negative rates, route hotspots, airline comparisons
- **Feedback Explorer** — Search, filter, inspect, and export individual records
- **Feedback Analyser** — Single-review prediction, batch CSV upload, model performance metrics

---

## Architecture

```text
AeroLens/
│
├── data/
│   ├── raw/                  # Raw synthetic dataset
│   │   └── feedback_raw.csv
│   └── processed/            # Cleaned + split datasets
│       ├── feedback_cleaned.csv
│       ├── train.csv
│       └── test.csv
│
├── src/
│   ├── __init__.py
│   ├── generate_data.py      # Synthetic data generator
│   ├── preprocessing.py      # Text cleaning + train/test split
│   ├── train.py              # Model training (TF-IDF + LogReg)
│   ├── predict.py            # Prediction utilities
│   └── evaluate.py           # Model evaluation metrics
│
├── models/                   # Trained model artifacts (.joblib)
│   ├── sentiment_model.joblib
│   ├── category_model.joblib
│   ├── tfidf_sentiment.joblib
│   └── tfidf_category.joblib
│
├── outputs/
│   └── evaluation_results.json
│
├── app.py                    # Streamlit application
├── requirements.txt
├── README.md
└── .gitignore
```

---

## Technical Stack

| Component | Technology |
|-----------|------------|
| Language | Python 3.10 |
| UI Framework | Streamlit |
| Data Processing | Pandas, NumPy |
| NLP / Feature Extraction | Scikit-learn TF-IDF Vectorizer |
| ML Algorithm | Logistic Regression (scikit-learn) |
| Visualisation | Matplotlib, Streamlit charts |
| Model Persistence | Joblib |
| Version Control | Git / GitHub |

---

## Setup Instructions

### Prerequisites
- Python 3.10+
- pip

### Installation

```bash
# Clone the repository
git clone https://github.com/<your-username>/AeroLens.git
cd AeroLens

# Create and activate virtual environment
python -m venv .venv
.venv\Scripts\activate       # Windows
# source .venv/bin/activate  # macOS/Linux

# Install dependencies
pip install -r requirements.txt
```

### Running the ML Pipeline

Execute these scripts **in order** from the project root:

```bash
# Step 1: Generate synthetic dataset
python src/generate_data.py

# Step 2: Clean data and create train/test splits
python src/preprocessing.py

# Step 3: Train sentiment and category models
python src/train.py

# Step 4: Evaluate models on the test set
python src/evaluate.py
```

### Launching the Application

```bash
streamlit run app.py
```

The application will open at `http://localhost:8501`.

---

## Model Evaluation

Both models are evaluated using:
- Accuracy
- Precision (macro and weighted)
- Recall (macro and weighted)
- F1 Score (macro and weighted)
- Confusion Matrix

Evaluation results are saved to `outputs/evaluation_results.json` and displayed in the application under **Feedback Analyser → Model Performance**.

### Data Leakage Prevention
- The TF-IDF vectoriser is fitted **only on training data**.
- The test set is transformed using the already-fitted vectoriser (no re-fitting).
- A stratified 80:20 train-test split ensures proportional class representation.

---

## Limitations and Disclaimers

1. **Synthetic data only.** The dataset is entirely computer-generated. All sentiment scores, airline comparisons, and performance indicators are illustrative and do **not** represent actual airline performance.

2. **Model accuracy on synthetic data does not establish real-world performance.** The models achieve high accuracy because the synthetic templates contain distinctive vocabulary per category. Real customer feedback would be noisier and more ambiguous, likely reducing model accuracy.

3. **No live data integrations.** The current version supports CSV uploads only. Social media APIs, feedback-form integrations, and automated ingestion are planned for future iterations.

4. **Priority assignment is rule-based.** The complaint prioritisation uses explicit business rules (not ML), which is clearly documented in the application.

5. **No transformer/BERT models.** The current approach uses TF-IDF + Logistic Regression, which is lightweight and interpretable but less powerful than deep-learning models for nuanced text classification.

---

## Future Enhancements

- Live social media scraping (Twitter/X API)
- API integrations with external feedback platforms
- Automated complaint notifications (Slack, email)
- Advanced transformer-based models (BERT, DistilBERT)
- Agentic complaint-routing workflows
- Automated response generation for unhappy customers

---

## License

This project is developed for educational and demonstration purposes.

---

*Built with Python, Streamlit, and scikit-learn.*
