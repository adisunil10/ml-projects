# Credit Card Fraud Detection

Binary classification pipeline on 284k real transactions (0.17% fraud rate). The main challenge is class imbalance — standard accuracy is useless here, so the focus is on PR-AUC and recall on the minority class.

**Dataset:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Results

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.735 | 0.972 | 0.798 |
| Random Forest | 0.870 | 0.982 | 0.856 |
| XGBoost (tuned) | **0.876** | 0.979 | **0.867** |

XGBoost with SMOTE oversampling and RandomizedSearchCV tuning achieves 0.876 PR-AUC, catching 80% of fraud cases with 95% precision.

## Setup

```bash
pip install -r requirements.txt
```

Place `creditcard.csv` in `data/` then run:

```bash
python src/pipeline.py
```

Plots and confusion matrices are saved to `results/`.

## Structure

```
src/
├── preprocess.py   # scaling, stratified split
├── features.py     # engineered features (log amount, hour, V interactions)
├── train.py        # LR, RF, XGBoost + SMOTE + hyperparameter search
├── evaluate.py     # PR-AUC, ROC-AUC, threshold selection, plots
└── pipeline.py     # runs everything end to end
notebooks/
└── 01_eda.ipynb    # class distribution, feature distributions, correlation
results/
└── summary.md      # findings and model comparison
```
