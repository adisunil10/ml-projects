# Credit Card Fraud Detection

Binary classification pipeline on 284k real transactions (0.17% fraud rate). Built around three core challenges: class imbalance, choosing the right evaluation metric, and optimising for business cost rather than model accuracy.

**Dataset:** [Kaggle — Credit Card Fraud Detection](https://www.kaggle.com/datasets/mlg-ulb/creditcardfraud)

## Results

| Model | PR-AUC | ROC-AUC | F1 |
|---|---|---|---|
| Logistic Regression | 0.735 | 0.972 | 0.798 |
| Random Forest | 0.870 | 0.982 | 0.856 |
| XGBoost (tuned) | **0.876** | 0.979 | **0.867** |

XGBoost with SMOTE + RandomizedSearchCV tuning. Threshold selected by minimising total dollar cost (FN = $88 avg transaction absorbed, FP = $10 declined legit card) rather than maximising F1.

## Setup

```bash
pip install -r requirements.txt
```

Place `creditcard.csv` in `data/`, then run the full pipeline:

```bash
python src/pipeline.py
```

## API

Start the prediction server:

```bash
uvicorn api.main:app --reload
```

Score a transaction:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"Time": 406, "Amount": 149.62, "V1": -1.36, "V2": -0.07, ...}'
```

Returns:
```json
{
  "fraud_probability": 0.9821,
  "prediction": "fraud",
  "threshold": 0.963
}
```

## Structure

```
src/
├── preprocess.py     # scaling, stratified split
├── features.py       # log amount, hour-of-day, V interactions, velocity features
├── train.py          # LR, RF, XGBoost + SMOTE + hyperparameter search
├── evaluate.py       # PR-AUC, ROC-AUC, threshold selection, plots
├── cost_analysis.py  # dollar cost curve, cost-minimising threshold
└── pipeline.py       # runs everything end to end
api/
└── main.py           # FastAPI /predict endpoint
notebooks/
└── 01_eda.ipynb      # class distribution, feature distributions, correlation
results/
└── summary.md        # findings and model comparison
```
