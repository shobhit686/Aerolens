## AeroLens
An airline customer experience intelligence platform built with Python, Machine Learning and Streamlit.

## Run locally
1. Install Python 3.10 or later.
2. Open a terminal in the AeroLens project folder.
3. Install dependencies:
   python -m pip install -r requirements.txt
4. Generate the synthetic dataset and train the models:
   python -m src.run_pipeline
5. Start the application:
   python -m streamlit run app.py

## Included capabilities
- Individual workspaces for IndiGo, Air India, Air India Express, Akasa Air and SpiceJet
- Sentiment classification: Positive, Neutral and Negative
- Complaint categorisation across eight service categories
- Rule-based complaint prioritisation (High, Medium and Low)
- Individual review analysis and batch CSV upload
- Dashboard with sentiment trends, complaint analysis and route-wise insights
- Feedback search, filtering and CSV export