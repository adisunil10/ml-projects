import joblib
from pathlib import Path

from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import RandomizedSearchCV
from xgboost import XGBClassifier
from imblearn.over_sampling import SMOTE


MODELS_DIR = Path(__file__).parent.parent / "models"
MODELS_DIR.mkdir(exist_ok=True)


def apply_smote(X_train, y_train, random_state: int = 42):
    smote = SMOTE(random_state=random_state)
    return smote.fit_resample(X_train, y_train)


def train_logistic_regression(X_train, y_train) -> LogisticRegression:
    model = LogisticRegression(class_weight="balanced", max_iter=2000, solver="saga", random_state=42)
    model.fit(X_train, y_train)
    return model


def train_random_forest(X_train, y_train) -> RandomForestClassifier:
    model = RandomForestClassifier(
        n_estimators=100,
        class_weight="balanced",
        n_jobs=-1,
        random_state=42,
    )
    model.fit(X_train, y_train)
    return model


def train_xgboost(X_train, y_train, tune: bool = True) -> XGBClassifier:
    neg = (y_train == 0).sum()
    pos = (y_train == 1).sum()

    base_params = dict(
        scale_pos_weight=neg / pos,
        eval_metric="aucpr",
        random_state=42,
        n_jobs=-1,
    )

    if not tune:
        model = XGBClassifier(**base_params, n_estimators=200, max_depth=6, learning_rate=0.1)
        model.fit(X_train, y_train)
        return model

    param_dist = {
        "n_estimators": [100, 200, 300, 500],
        "max_depth": [3, 4, 5, 6, 8],
        "learning_rate": [0.01, 0.05, 0.1, 0.2],
        "subsample": [0.6, 0.8, 1.0],
        "colsample_bytree": [0.6, 0.8, 1.0],
        "min_child_weight": [1, 3, 5],
        "gamma": [0, 0.1, 0.3],
    }

    search = RandomizedSearchCV(
        XGBClassifier(**base_params),
        param_distributions=param_dist,
        n_iter=30,
        scoring="average_precision",
        cv=3,
        n_jobs=-1,
        random_state=42,
        verbose=1,
    )
    search.fit(X_train, y_train)
    print(f"Best XGBoost params: {search.best_params_}")
    return search.best_estimator_


def save_model(model, name: str):
    path = MODELS_DIR / f"{name}.joblib"
    joblib.dump(model, path)
    print(f"Saved {name} → {path}")


def load_model(name: str):
    return joblib.load(MODELS_DIR / f"{name}.joblib")
