# 🗣️ Vanishing Voices — Language Extinction Risk Predictor

A machine learning mini project that predicts whether a language is at risk of extinction using the UNESCO Endangered Languages dataset.

## 🎯 Problem Statement
Given a language's number of speakers and geographic location, predict if it is **Endangered** or **Not Endangered**.

## 📊 Dataset
- **Source:** UNESCO Atlas of the World's Languages in Danger (via Kaggle)
- **Size:** 2,722 languages
- **Features Used:** Number of speakers, Latitude, Longitude
- **Target:** Binary classification (1 = Endangered, 0 = Not Endangered)

## 🔧 Tech Stack
- Python
- Pandas, NumPy
- Scikit-learn (RandomForestClassifier)
- Streamlit (Dashboard)
- Jupyter Notebook

## 🧠 ML Workflow
1. **Data Collection** — Loaded UNESCO endangered languages CSV
2. **Preprocessing** — Handled missing values with median/mean imputation
3. **Feature Engineering** — Created binary target from 5-class UNESCO labels
4. **Feature Scaling** — StandardScaler to normalize speaker counts and coordinates
5. **Train-Test Split** — 80/20 split with `random_state=42` for reproducibility
6. **Model Training** — RandomForestClassifier
7. **Evaluation** — Accuracy, Confusion Matrix, Precision, Recall, F1-score
8. **Deployment** — Streamlit web app for interactive predictions

## 📈 Results
- **Accuracy:** ~59% on test set
- **Best Performance:** Extinct languages (96% recall)
- **Model:** Catches endangered languages effectively; struggles with borderline cases

## 🚀 How to Run

### Jupyter Notebook
```bash
jupyter notebook
# Open notebooks/01_ml_pipeline.ipynb