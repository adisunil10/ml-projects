import pandas as pd
from pathlib import Path

from preprocess import load_and_prepare
from features import engineer_features
from train import apply_smote, train_logistic_regression, train_random_forest, train_xgboost, save_model
from evaluate import evaluate_model, plot_pr_curves, plot_roc_curves, plot_confusion_matrix, plot_threshold_analysis

DATA_PATH = Path(__file__).parent.parent / "data" / "creditcard.csv"


def main():
    print("Loading data...")
    X_train, X_test, y_train, y_test = load_and_prepare(str(DATA_PATH))

    print("Engineering features...")
    X_train = engineer_features(X_train)
    X_test = engineer_features(X_test)

    print(f"Train: {X_train.shape}  fraud rate={y_train.mean():.4f}")
    print(f"Test : {X_test.shape}  fraud rate={y_test.mean():.4f}")

    print("\nApplying SMOTE...")
    X_train_sm, y_train_sm = apply_smote(X_train, y_train)
    print(f"After SMOTE: {X_train_sm.shape}  fraud rate={y_train_sm.mean():.4f}")

    results = []

    print("\nTraining Logistic Regression...")
    lr = train_logistic_regression(X_train_sm, y_train_sm)
    save_model(lr, "logistic_regression")
    results.append(evaluate_model(lr, X_test, y_test, "Logistic Regression"))

    print("\nTraining Random Forest...")
    rf = train_random_forest(X_train_sm, y_train_sm)
    save_model(rf, "random_forest")
    results.append(evaluate_model(rf, X_test, y_test, "Random Forest"))

    print("\nTraining XGBoost...")
    xgb = train_xgboost(X_train_sm, y_train_sm, tune=True)
    save_model(xgb, "xgboost")
    results.append(evaluate_model(xgb, X_test, y_test, "XGBoost"))

    print("\nGenerating plots...")
    plot_pr_curves(results, y_test)
    plot_roc_curves(results, y_test)
    for r in results:
        plot_confusion_matrix(y_test, r["y_pred"], r["name"])
        plot_threshold_analysis(r, y_test)

    best = max(results, key=lambda r: r["pr_auc"])
    print(f"\nBest model: {best['name']}  PR-AUC={best['pr_auc']:.4f}")


if __name__ == "__main__":
    main()
