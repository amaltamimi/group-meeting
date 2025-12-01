import os
from typing import Any, Dict, List

import joblib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.linear_model import Ridge
from sklearn.model_selection import KFold, cross_val_score

from . import featurize, utils


TARGET_COLUMN = "target"


def synthetic_target(df: pd.DataFrame, random_state: int) -> pd.Series:
    rng = np.random.default_rng(random_state)
    base = 0.1 * df.get("desc_mol_wt", 0) + 0.5 * df.get("desc_logp", 0) + 0.2 * df.get("desc_tpsa", 0)
    noise = rng.normal(0, 0.1, size=len(df))
    return base + noise


def train_models(df: pd.DataFrame, settings: Dict[str, Any]):
    train_cfg = settings.get("training", {})
    models_dir = settings.get("paths", {}).get("models_dir", "models")
    reports_dir = settings.get("paths", {}).get("reports_dir", "reports")
    utils.ensure_directories([models_dir, reports_dir])

    random_state = int(train_cfg.get("random_state", 42))
    cv_folds = int(train_cfg.get("cv_folds", 5))

    if TARGET_COLUMN not in df.columns:
        df[TARGET_COLUMN] = synthetic_target(df, random_state=random_state)

    X = featurize.get_feature_matrix(df)
    y = df[TARGET_COLUMN]

    ridge = Ridge(alpha=float(train_cfg.get("ridge", {}).get("alpha", 1.0)), random_state=random_state)
    rf_cfg = train_cfg.get("random_forest", {})
    rf = RandomForestRegressor(
        n_estimators=int(rf_cfg.get("n_estimators", 200)),
        max_depth=rf_cfg.get("max_depth"),
        random_state=random_state,
        n_jobs=-1,
    )

    kf = KFold(n_splits=cv_folds, shuffle=True, random_state=random_state)
    ridge_scores = cross_val_score(ridge, X, y, cv=kf, scoring="r2")
    rf_scores = cross_val_score(rf, X, y, cv=kf, scoring="r2")

    ridge.fit(X, y)
    rf.fit(X, y)

    joblib.dump(ridge, os.path.join(models_dir, "ridge.pkl"))
    joblib.dump(rf, os.path.join(models_dir, "random_forest.pkl"))

    plot_feature_importance(rf, X.columns, os.path.join(reports_dir, "feature_importance.png"))

    metrics = {
        "ridge_cv_mean_r2": float(np.mean(ridge_scores)),
        "rf_cv_mean_r2": float(np.mean(rf_scores)),
    }
    return metrics


def plot_feature_importance(model, feature_names: List[str], output_path: str) -> None:
    importances = model.feature_importances_
    indices = np.argsort(importances)[-20:]
    top_features = np.array(feature_names)[indices]
    top_importances = importances[indices]

    plt.figure(figsize=(8, 6))
    plt.barh(range(len(top_features)), top_importances, color="steelblue")
    plt.yticks(range(len(top_features)), top_features)
    plt.xlabel("Importance")
    plt.title("Top 20 Feature Importances (Random Forest)")
    plt.tight_layout()
    plt.savefig(output_path)
    plt.close()


def main() -> None:
    args = utils.parse_args()
    settings = utils.load_settings(args.config)
    logger = utils.get_logger(__name__)

    processed_path = settings.get("paths", {}).get("processed_data", "data/processed/processed.csv")
    if not os.path.exists(processed_path):
        raise FileNotFoundError(f"Processed data not found at {processed_path}. Run featurize first.")

    df = pd.read_csv(processed_path)
    metrics = train_models(df, settings)

    logger.info("Training complete. Metrics: %s", metrics)


if __name__ == "__main__":
    main()
