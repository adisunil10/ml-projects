# Credit Card Fraud Detection Engine

Anomaly detection pipeline on 284k+ transactions using XGBoost, SMOTE, and scikit-learn. Addresses severe class imbalance (~0.17% fraud rate) via SMOTE oversampling and cost-sensitive learning.

## Project Structure

```
credit-card-fraud-detection/
├── data/
│   └── creditcard.csv          # Kaggle dataset (not committed)
├── notebooks/
│   └── 01_eda.ipynb            # Exploratory data analysis
├── src/
│   ├── preprocess.py           # Loading, scaling, train/test split
│   ├── features.py             # Feature engineering
│   ├── train.py                # Model training and hyperparameter tuning
│   ├── evaluate.py             # Metrics, plots, threshold selection
│   └── pipeline.py             # End-to-end runner
├── models/                     # Saved model artifacts (not committed)
├── requirements.txt
└── .gitignore
```

## Setup

```bash
pip install -r requirements.txt
```

Place `creditcard.csv` (from [Kaggle](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)) inside the `data/` folder.

## Run the Full Pipeline

```bash
python src/pipeline.py
```

This will:
1. Load and preprocess the data
2. Engineer features
3. Train Logistic Regression, Random Forest, and XGBoost models
4. Apply SMOTE on the training set
5. Tune XGBoost with RandomizedSearchCV
6. Evaluate all models and print/save results

## Key Metrics

Accuracy is misleading with imbalanced data. This project evaluates models using:
- **PR-AUC** (primary) — Precision-Recall Area Under Curve
- **ROC-AUC**
- **F1 Score** at optimal threshold
- Confusion matrix

## Dataset

- Source: [Kaggle Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)
- 284,807 transactions, 492 fraud cases (0.172%)
- Features V1–V28 are PCA-transformed; `Amount` and `Time` are raw
