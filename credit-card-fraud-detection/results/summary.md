# Results

## Model Comparison

| Model | PR-AUC | ROC-AUC | F1 | Threshold (F1) |
|---|---|---|---|---|
| Logistic Regression | 0.735 | 0.972 | 0.798 | 1.000 |
| Random Forest | 0.870 | 0.982 | 0.856 | 0.710 |
| XGBoost (tuned) | **0.876** | 0.979 | **0.867** | 0.963 |

All models trained on SMOTE-balanced data (50/50 after oversampling from 0.17% fraud rate).

## Cost-Sensitive Evaluation

Standard F1/PR-AUC don't reflect the real cost asymmetry in fraud detection:
- **False negative** (missed fraud): bank absorbs the charge — ~$88 average transaction
- **False positive** (declined legit card): ~$10 in customer service + churn risk

Optimising threshold for minimum total dollar loss rather than F1 gives a different — and more operationally meaningful — operating point. The cost-minimising threshold for XGBoost is lower than the F1-maximising one, favouring recall over precision (catching more fraud, accepting more false alarms).

## Velocity Features

Rolling 1-hour transaction count and spend added as features. Since this dataset is anonymised (no card IDs, V1–V28 are PCA components), these are population-level rather than per-card — they capture fraud spikes that cluster in time. In a production setting these would be computed per card from a feature store.

## What worked

**SMOTE** gave a clear lift over class_weight alone for tree models — the synthetic minority samples improved boundary learning in high-dimensional PCA space.

**Threshold tuning** — default 0.5 is wrong here. Picking the threshold from the PR curve (either F1-max or cost-min) is necessary for the model to be usable.

**XGBoost tuning** — best params from RandomizedSearchCV (30 iterations, 3-fold CV, scoring=average_precision):
- `n_estimators`: 500, `max_depth`: 8, `learning_rate`: 0.2
- `subsample`: 0.8, `colsample_bytree`: 1.0, `gamma`: 0.1

## What to watch

Logistic Regression returned `threshold=1.000` — its fraud probabilities never confidently exceed the F1-optimal cutoff, meaning it can't cleanly separate classes at this feature set. Not a usable model for this problem.

XGBoost misses ~20% of fraud cases. These are likely transactions where the PCA-transformed behaviour overlaps with legitimate patterns — the kind that per-card velocity features (transaction frequency spikes, rapid location changes) would help flag.

## API

The trained XGBoost model is served via a FastAPI endpoint (`api/main.py`). Send a POST request to `/predict` with a transaction's raw features and get back a fraud probability and binary prediction at the cost-optimised threshold.
