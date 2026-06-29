import numpy as np
import matplotlib.pyplot as plt
from pathlib import Path

from sklearn.metrics import (
    average_precision_score,
    roc_auc_score,
    precision_recall_curve,
    roc_curve,
    confusion_matrix,
    f1_score,
    classification_report,
)

RESULTS_DIR = Path(__file__).parent.parent / "results"
RESULTS_DIR.mkdir(exist_ok=True)


def find_optimal_threshold(y_true, y_proba) -> float:
    """F1-maximising threshold on the PR curve."""
    precisions, recalls, thresholds = precision_recall_curve(y_true, y_proba)
    f1s = 2 * precisions * recalls / (precisions + recalls + 1e-9)
    return thresholds[np.argmax(f1s[:-1])]


def evaluate_model(model, X_test, y_test, name: str) -> dict:
    y_proba = model.predict_proba(X_test)[:, 1]
    threshold = find_optimal_threshold(y_test, y_proba)
    y_pred = (y_proba >= threshold).astype(int)

    pr_auc = average_precision_score(y_test, y_proba)
    roc_auc = roc_auc_score(y_test, y_proba)
    f1 = f1_score(y_test, y_pred)

    print(f"\n{'='*50}")
    print(f"Model: {name}  |  threshold={threshold:.3f}")
    print(f"  PR-AUC : {pr_auc:.4f}")
    print(f"  ROC-AUC: {roc_auc:.4f}")
    print(f"  F1     : {f1:.4f}")
    print(classification_report(y_test, y_pred, target_names=["Legit", "Fraud"]))

    return {"name": name, "pr_auc": pr_auc, "roc_auc": roc_auc, "f1": f1,
            "threshold": threshold, "y_proba": y_proba, "y_pred": y_pred}


def plot_pr_curves(results: list, y_test):
    plt.figure(figsize=(8, 6))
    for r in results:
        p, rec, _ = precision_recall_curve(y_test, r["y_proba"])
        plt.plot(rec, p, label=f"{r['name']} (PR-AUC={r['pr_auc']:.3f})")
    plt.xlabel("Recall")
    plt.ylabel("Precision")
    plt.title("Precision-Recall Curves")
    plt.legend()
    plt.tight_layout()
    path = RESULTS_DIR / "pr_curves.png"
    plt.savefig(path, dpi=150)
    print(f"Saved PR curves → {path}")
    plt.close()


def plot_roc_curves(results: list, y_test):
    plt.figure(figsize=(8, 6))
    for r in results:
        fpr, tpr, _ = roc_curve(y_test, r["y_proba"])
        plt.plot(fpr, tpr, label=f"{r['name']} (ROC-AUC={r['roc_auc']:.3f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("ROC Curves")
    plt.legend()
    plt.tight_layout()
    path = RESULTS_DIR / "roc_curves.png"
    plt.savefig(path, dpi=150)
    print(f"Saved ROC curves → {path}")
    plt.close()


def plot_confusion_matrix(y_test, y_pred, name: str):
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(5, 4))
    im = ax.imshow(cm, cmap="Blues")
    ax.set_xticks([0, 1]); ax.set_yticks([0, 1])
    ax.set_xticklabels(["Legit", "Fraud"]); ax.set_yticklabels(["Legit", "Fraud"])
    ax.set_xlabel("Predicted"); ax.set_ylabel("Actual")
    ax.set_title(f"Confusion Matrix — {name}")
    for i in range(2):
        for j in range(2):
            ax.text(j, i, cm[i, j], ha="center", va="center", fontsize=14)
    plt.tight_layout()
    path = RESULTS_DIR / f"cm_{name.replace(' ', '_').lower()}.png"
    plt.savefig(path, dpi=150)
    print(f"Saved confusion matrix → {path}")
    plt.close()
