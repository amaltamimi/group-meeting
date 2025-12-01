import os

import joblib
import matplotlib.pyplot as plt
import pandas as pd

from . import data_download, featurize, utils


def predict_new_molecules(config_path: str) -> pd.DataFrame:
    settings = utils.load_settings(config_path)
    logger = utils.get_logger(__name__)

    models_dir = settings.get("paths", {}).get("models_dir", "models")
    reports_dir = settings.get("paths", {}).get("reports_dir", "reports")
    utils.ensure_directories([reports_dir])

    ridge_path = os.path.join(models_dir, "ridge.pkl")
    rf_path = os.path.join(models_dir, "random_forest.pkl")

    if not os.path.exists(ridge_path) or not os.path.exists(rf_path):
        raise FileNotFoundError("Trained models not found. Run train.py first.")

    ridge = joblib.load(ridge_path)
    rf = joblib.load(rf_path)

    # Fetch molecules for evaluation
    eval_cfg = settings.get("evaluation", {})
    pubchem_cfg = settings.get("pubchem", {}).copy()
    pubchem_cfg["identifiers"] = eval_cfg.get("molecules", [])
    temp_settings = settings.copy()
    temp_settings["pubchem"] = pubchem_cfg

    # Write temporary settings to reuse fetcher
    temp_config_path = os.path.join("/tmp", "temp_settings.yaml")
    utils.save_json(temp_settings, temp_config_path)
    df_raw = data_download.fetch_compounds(temp_config_path)

    feat_df = featurize.featurize_dataframe(df_raw, settings=settings, logger=logger)
    X = featurize.get_feature_matrix(feat_df)

    feat_df["ridge_pred"] = ridge.predict(X)
    feat_df["rf_pred"] = rf.predict(X)
    return feat_df


def update_report(df_preds: pd.DataFrame, report_path: str) -> None:
    lines = ["# Model Performance\n", "\n", "## Latest Predictions\n", "\n"]
    lines.append(df_preds[["query", "ridge_pred", "rf_pred"]].to_markdown(index=False))
    lines.append("\n\n")
    lines.append("![Prediction Comparison](prediction_comparison.png)\n")
    with open(report_path, "w", encoding="utf-8") as f:
        f.writelines(lines)


def plot_predictions(df_preds: pd.DataFrame, output_dir: str) -> str:
    path = os.path.join(output_dir, "prediction_comparison.png")
    plt.figure(figsize=(6, 6))
    plt.scatter(df_preds["ridge_pred"], df_preds["rf_pred"], color="purple")
    plt.xlabel("Ridge Prediction")
    plt.ylabel("Random Forest Prediction")
    plt.title("Model Prediction Comparison")
    plt.grid(True)
    plt.tight_layout()
    plt.savefig(path)
    plt.close()
    return path


def main() -> None:
    args = utils.parse_args()
    settings = utils.load_settings(args.config)
    logger = utils.get_logger(__name__)

    df_preds = predict_new_molecules(args.config)
    reports_dir = settings.get("paths", {}).get("reports_dir", "reports")
    report_path = settings.get("evaluation", {}).get("report_file", os.path.join(reports_dir, "model_performance.md"))

    plot_predictions(df_preds, reports_dir)
    update_report(df_preds, report_path)
    logger.info("Evaluation complete. Report saved to %s", report_path)


if __name__ == "__main__":
    main()
