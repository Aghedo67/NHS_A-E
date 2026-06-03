# NHS A&E Performance Predictor — Streamlit App

## Setup

1. Install dependencies:
   ```bash
   pip install -r requirements.txt
   ```

2. Place these CSV files in the same folder as `app.py`:
   - `Provider_Level_Data.csv`
   - `System_Mapping.csv`
   - `AE_model_predictions.csv`

3. Run the app:
   ```bash
   streamlit run app.py
   ```

## Pages

| Page | Description |
|------|-------------|
| 📊 Overview | National KPIs, regional performance bar chart, worst providers |
| 🔍 EDA | Distributions, scatter plots, correlation heatmap |
| 🤖 Model Results | ROC curves, feature importances, confusion matrix for all 3 models |
| 🔮 Predict Provider | Enter custom values or pick an existing trust to get a live prediction |
| 📋 Provider Explorer | Filter, sort, and download all providers with ML scores |

## Models trained
- Random Forest (Accuracy: 81.6%, ROC-AUC: 0.911)
- Gradient Boosting (Accuracy: 76.3%, ROC-AUC: 0.884)
- Logistic Regression (Accuracy: 86.8%, ROC-AUC: 0.975) ← best

## Deploy to Streamlit Cloud
1. Push this folder to a GitHub repo
2. Go to https://share.streamlit.io
3. Connect your repo and set `app.py` as the entry point
