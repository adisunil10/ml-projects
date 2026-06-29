import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path
from sklearn.metrics import precision_recall_curve

RESULTS_DIR = Path(__file__).parent.parent / "results"

# FN: bank absorbs the fraudulent charge (average transaction amount in dataset ~$88)
# FP: declined legitimate transaction — customer service cost + estimated churn risk
FN_COST = 88.0
FP_COST = 10.0


def compute_cost_curve(y_true, y_proba, fn_cost: float = FN_COST, fp_cost: float = FP_COST):
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    n_fraud = y_true.sum()
    n_legit = (y_true == 0).sum()

    costs = []
    for p, r, t in zip(precisions[:-1], recalls[:-1], thresholds):
        fn = n_fraud * (1 - r)
        fp = (r * n_fraud / p) - (r * n_fraud) if p > 0 else 0
        costs.append(fn * fn_cost + fp * fp_cost)

    return thresholds, np.array(costs)


def find_min_cost_threshold(y_true, y_proba) -> tuple[float, float]:
    thresholds, costs = compute_cost_curve(y_true, y_proba)
    idx = np.argmin(costs)
    return thresholds[idx], costs[idx]


def plot_cost_curve(result: dict, y_test):
    thresholds, costs = compute_cost_curve(y_test, result["y_proba"])
    opt_threshold, min_cost = find_min_cost_threshold(y_test, result["y_proba"])

    fig, ax = plt.subplots(figsize=(9, 5))
    ax.plot(thresholds, costs, color="steelblue")
    ax.axvline(opt_threshold, color="red", linewidth=0.8,
               label=f"Min-cost threshold ({opt_threshold:.3f})  —  ${min_cost:,.0f} total loss")
    ax.set_xlabel("Decision threshold")
    ax.set_ylabel(f"Estimated cost  (FN=${FN_COST}, FP=${FP_COST})")
    ax.set_title(f"Cost vs Threshold — {result['name']}")
    ax.legend()
    plt.tight_layout()

    name_slug = result["name"].replace(" ", "_").lower()
    path = RESULTS_DIR / f"cost_{name_slug}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved cost curve → {path}  (optimal threshold={opt_threshold:.3f}, cost=${min_cost:,.0f})")
    plt.close()

    return opt_threshold, min_cost
