# Results

## Model Comparison

| Model | PR-AUC | ROC-AUC | F1 | Threshold |
|---|---|---|---|---|
| Logistic Regression | 0.735 | 0.972 | 0.798 | 1.000 |
| Random Forest | 0.870 | 0.982 | 0.856 | 0.710 |
| XGBoost (tuned) | **0.876** | 0.979 | **0.867** | 0.963 |

All models trained on SMOTE-balanced data (50/50 split after oversampling from 0.17% fraud rate).

## What worked

**SMOTE before tree models** — training on the balanced set gave Random Forest and XGBoost a strong lift over the baseline. Without it, both models would default to predicting legit on ambiguous cases.

**Threshold tuning** — default 0.5 threshold is wrong for imbalanced data. Picking the F1-maximising threshold from the PR curve pushed F1 from ~0.84 to 0.867 on XGBoost.

**XGBoost tuning** — best params from RandomizedSearchCV (30 iterations, 3-fold CV):
- `n_estimators`: 500, `max_depth`: 8, `learning_rate`: 0.2
- `subsample`: 0.8, `colsample_bytree`: 1.0, `gamma`: 0.1

## What to watch

Logistic Regression returned `threshold=1.000` — the model's fraud probabilities never exceeded the F1-optimal cutoff, indicating it can't separate the classes cleanly at this feature set. Its reported F1 of 0.798 is at the maximum-confidence boundary and not a reliable operating point.

XGBoost catches **80% of fraud cases** (recall=0.80) with **95% precision**. The remaining 20% missed are likely the edge cases where fraud patterns overlap with legitimate behaviour — this is where feature engineering (e.g. velocity features, per-card history) would help most.

## Next steps

- Add velocity features: transaction count and total spend per card in a rolling time window
- Try Isolation Forest as an unsupervised complement to flag anomalies the classifier misses
- Evaluate at multiple operating thresholds to simulate real cost tradeoffs (false negative = missed fraud, false positive = declined legitimate transaction)
