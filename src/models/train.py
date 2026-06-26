import os
import json
import warnings
from pathlib import Path

import numpy as np
import pandas as pd
from dotenv import load_dotenv
from sklearn.dummy import DummyClassifier
from sklearn.metrics import roc_auc_score, average_precision_score
import mlflow
from mlflow.tracking import MlflowClient

# Nouveaux imports
from const import DATA_DIR, MODEL_DIR, CONFIG_DIR, SHARPE_THRESHOLD, MAX_DD_THRESHOLD
from src.utils.metrics import calculate_financial_metrics
from src.features.alpha_features import add_all_features
from src.models.ensemble import AlphaEdgeEnsemble, FEATURE_GROUPS
from src.utils.logger import setup_logger

load_dotenv()
warnings.filterwarnings("ignore")
logger = setup_logger("train")

HF_TOKEN = os.getenv("HF_TOKEN")
USE_MLFLOW = bool(HF_TOKEN)

if USE_MLFLOW:
    os.environ["MLFLOW_TRACKING_USERNAME"] = "SORADATA"
    os.environ["MLFLOW_TRACKING_PASSWORD"] = HF_TOKEN
    mlflow.set_tracking_uri("https://soradata-alphaedge-registry.hf.space")
    mlflow.set_experiment("AlphaEdge_Ensemble_Production")


def walk_forward_eval(df: pd.DataFrame, n_windows: int = 4, test_months: int = 3, n_optuna_trials: int = 20) -> pd.DataFrame:
    dates = df.index.get_level_values("date").unique().sort_values()
    results = []

    for i in range(n_windows):
        test_end = dates[-(i * test_months + 1)]
        test_start = dates[-(i * test_months + test_months)]
        train_end = test_start - pd.DateOffset(months=1)

        df_tr = df[df.index.get_level_values("date") <= train_end]
        df_te = df[
            (df.index.get_level_values("date") >= test_start) &
            (df.index.get_level_values("date") <= test_end)
        ]

        if len(df_tr) < 50 or len(df_te) < 5:
            continue
        if len(df_te["target"].unique()) < 2:
            continue

        model = AlphaEdgeEnsemble(n_optuna_trials=n_optuna_trials)
        model.fit(df_tr, df_tr["target"])
        proba = model.predict_proba(df_te)[:, 1]

        results.append({
            "window":     i + 1,
            "test_start": str(test_start.date()),
            "test_end":   str(test_end.date()),
            "auc":        round(roc_auc_score(df_te["target"], proba), 4),
            "apr":        round(average_precision_score(df_te["target"], proba), 4),
            "n_test":     len(df_te),
        })
        logger.info(
            f"Window {i+1} | AUC: {results[-1]['auc']:.4f} | "
            f"APR: {results[-1]['apr']:.4f} | n={results[-1]['n_test']}"
        )

    return pd.DataFrame(results)


def train_pipeline(market_name: str = "CAC40") -> tuple[AlphaEdgeEnsemble, dict]:
    logger.info(f"Début du cycle d'entraînement AlphaEdge — {market_name}")

    data_path = DATA_DIR / "processed" / market_name / "monthly_features.parquet"
    if not data_path.exists():
        raise FileNotFoundError(f"Fichier source introuvable : {data_path}")

    df = pd.read_parquet(data_path)
    df = add_all_features(df)

    df["future_return"] = df.groupby(level="ticker")["adj close"].pct_change(1).shift(-1)
    df["target"] = df["future_return"].gt(0).astype(int)
    df = df.dropna(subset=["target", "future_return"])

    dates = df.index.get_level_values("date")
    split_date = dates.max() - pd.DateOffset(months=6)

    df_train = df[dates <= split_date].copy()
    df_test = df[dates > split_date].copy()

    if len(df_train) < 100:
        raise ValueError(f"Volume de données insuffisant : {len(df_train)}")

    all_feats = [f for g in FEATURE_GROUPS.values() for f in g]
    available = [f for f in all_feats if f in df_train.columns]

    dummy = DummyClassifier(strategy="most_frequent").fit(df_train[available], df_train["target"])
    baseline_auc = roc_auc_score(df_test["target"], dummy.predict_proba(df_test[available])[:, 1])

    model = AlphaEdgeEnsemble(n_optuna_trials=50)
    model.fit(df_train, df_train["target"])

    proba = model.predict_proba(df_test)[:, 1]
    final_auc = roc_auc_score(df_test["target"], proba)
    final_apr = average_precision_score(df_test["target"], proba)
    lift = final_auc - baseline_auc
    fin_metrics = calculate_financial_metrics(df_test, probas=proba, threshold=0.5)

    logger.info(f"ML -> AUC : {final_auc:.4f} | APR : {final_apr:.4f} | Lift : +{lift:.4f}")
    logger.info(f"Finances -> Sharpe : {fin_metrics['sharpe']} | Max DD : {fin_metrics['max_drawdown']}")

    wf_results = walk_forward_eval(df, n_windows=4, test_months=3, n_optuna_trials=20)

    MODEL_DIR.mkdir(parents=True, exist_ok=True)
    model.save(MODEL_DIR / "ensemble_model.pkl")

    model_card = {
        "market":        market_name,
        "trained_at":    pd.Timestamp.now().isoformat(),
        "architecture":  "XGBoost + LightGBM + Ridge → LogisticRegression",
        "metrics_ml": {
            "auc_test":      round(final_auc, 4),
            "apr_test":      round(final_apr, 4),
            "baseline_auc":  round(baseline_auc, 4),
            "lift":          round(lift, 4)
        },
        "metrics_fin": fin_metrics,
        "model_weights": model.get_model_weights(),
        "walk_forward":  wf_results.to_dict(orient="records") if not wf_results.empty else [],
        "n_features":    len(model.features_),
        "train_size":    len(df_train),
        "test_size":     len(df_test),
    }
    with open(MODEL_DIR / "model_card.json", "w") as f:
        json.dump(model_card, f, indent=2)

    if USE_MLFLOW:
        registered_model_name = f"AlphaEdge_Ensemble_{market_name}"
        with mlflow.start_run(run_name=f"Ensemble_{market_name}") as run:
            mlflow.log_params({
                "architecture": "XGB+LGB+Ridge->LR",
                "market":       market_name,
                "n_features":   len(model.features_)
            })
            mlflow.log_metrics({
                "AUC_Test":     final_auc,
                "APR_Test":     final_apr,
                "Sharpe_Ratio": fin_metrics["sharpe"],
                "Max_Drawdown": fin_metrics["max_drawdown"],
                "Total_Return": fin_metrics["total_return"],
                "WF_AUC_mean":  wf_results["auc"].mean() if not wf_results.empty else 0.0
            })
            mlflow.log_dict(model_card, "model_card.json")
            mlflow.sklearn.log_model(model, name="ensemble_model")
            mv = mlflow.register_model(
                model_uri=f"runs:/{run.info.run_id}/ensemble_model",
                name=registered_model_name,
            )
            client = MlflowClient()
            if fin_metrics["sharpe"] >= SHARPE_THRESHOLD and fin_metrics["max_drawdown"] >= MAX_DD_THRESHOLD:
                client.set_registered_model_alias(registered_model_name, "champion", mv.version)
                logger.info(f"Promotion : Version {mv.version} passe 'champion'.")
            else:
                logger.warning(
                    "Promotion refusée : seuils de risque non atteints. Ancien champion maintenu."
                    )

    return model, model_card


if __name__ == "__main__":
    config_path = CONFIG_DIR / "markets" / "cac40.json"
    market = json.load(open(config_path))["market_name"] if config_path.exists() else "CAC40"
    train_pipeline(market)
