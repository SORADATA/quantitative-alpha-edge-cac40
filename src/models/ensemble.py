"""
AlphaEdgeEnsemble
==================
XGBoost + LightGBM + Ridge → LogisticRegression (stacking).
Optimisation Bayésienne (Optuna) par modèle.
Interface sklearn : fit / predict_proba / get_model_weights.
"""

import warnings
import numpy as np
import pandas as pd
import optuna
import pickle
from typing import Dict, List, Optional

import xgboost as xgb
import lightgbm as lgb
from sklearn.linear_model import LogisticRegression, RidgeClassifier
from sklearn.calibration import CalibratedClassifierCV
from sklearn.preprocessing import StandardScaler
from sklearn.base import BaseEstimator, ClassifierMixin

from src.models.cv import PurgedTimeSeriesSplit
from src.utils.logger import setup_logger
from const import FEATURE_GROUPS

optuna.logging.set_verbosity(optuna.logging.WARNING)
warnings.filterwarnings("ignore")
logger = setup_logger("ensemble")


def _all_features() -> List[str]:
    seen, result = set(), []
    for feats in FEATURE_GROUPS.values():
        for f in feats:
            if f not in seen:
                seen.add(f)
                result.append(f)
    return result


def _available(df: pd.DataFrame) -> List[str]:
    all_f = _all_features()
    return [f for f in all_f if f in df.columns]


def _prepare_X(df: pd.DataFrame, features: List[str]) -> pd.DataFrame:
    return df[features].fillna(0).replace([np.inf, -np.inf], 0)


# ══════════════════════════════════════════════════════════════════
# OPTUNA OBJECTIVES
# ══════════════════════════════════════════════════════════════════

def _xgb_objective(X: pd.DataFrame, y: pd.Series, n_trials: int) -> xgb.XGBClassifier:
    from sklearn.metrics import roc_auc_score
    cv = PurgedTimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 6),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_weight": trial.suggest_int("min_child_weight", 1, 10),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "gamma":            trial.suggest_float("gamma", 0.0, 1.0),
            "eval_metric": "auc", "random_state": 42, "n_jobs": -1,
        }
        scores = []
        for tr_idx, val_idx in cv.split(X):
            m = xgb.XGBClassifier(**params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx], verbose=False)
            p = m.predict_proba(X.iloc[val_idx])[:, 1]
            if len(np.unique(y.iloc[val_idx])) > 1:
                scores.append(roc_auc_score(y.iloc[val_idx], p))
        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=42),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = xgb.XGBClassifier(**study.best_params, eval_metric="auc", random_state=42, n_jobs=-1)
    best.fit(X, y, verbose=False)
    logger.info(f"   XGB best AUC (CV): {study.best_value:.4f} | params: {study.best_params}")
    return best


def _lgb_objective(X: pd.DataFrame, y: pd.Series, n_trials: int) -> lgb.LGBMClassifier:
    from sklearn.metrics import roc_auc_score
    cv = PurgedTimeSeriesSplit(n_splits=5)

    def objective(trial):
        params = {
            "n_estimators":     trial.suggest_int("n_estimators", 100, 500, step=50),
            "max_depth":        trial.suggest_int("max_depth", 3, 7),
            "learning_rate":    trial.suggest_float("learning_rate", 0.01, 0.15, log=True),
            "subsample":        trial.suggest_float("subsample", 0.6, 1.0),
            "colsample_bytree": trial.suggest_float("colsample_bytree", 0.5, 1.0),
            "min_child_samples":trial.suggest_int("min_child_samples", 5, 50),
            "reg_alpha":        trial.suggest_float("reg_alpha", 1e-3, 10.0, log=True),
            "reg_lambda":       trial.suggest_float("reg_lambda", 1e-3, 10.0, log=True),
            "random_state": 42, "n_jobs": -1, "verbose": -1,
        }
        scores = []
        for tr_idx, val_idx in cv.split(X):
            m = lgb.LGBMClassifier(**params)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            p = m.predict_proba(X.iloc[val_idx])[:, 1]
            if len(np.unique(y.iloc[val_idx])) > 1:
                scores.append(roc_auc_score(y.iloc[val_idx], p))
        return np.mean(scores) if scores else 0.0

    study = optuna.create_study(
        direction="maximize",
        sampler=optuna.samplers.TPESampler(seed=0),
        pruner=optuna.pruners.MedianPruner(n_startup_trials=10),
    )
    study.optimize(objective, n_trials=n_trials, show_progress_bar=False)

    best = lgb.LGBMClassifier(**study.best_params, random_state=42, n_jobs=-1, verbose=-1)
    best.fit(X, y)
    logger.info(f"   LGB best AUC (CV): {study.best_value:.4f} | params: {study.best_params}")
    return best


# ══════════════════════════════════════════════════════════════════
# ALPHAEDGE ENSEMBLE
# ══════════════════════════════════════════════════════════════════

class AlphaEdgeEnsemble(BaseEstimator, ClassifierMixin):
    """
    Ensemble à 2 niveaux :
      Niveau 0 : XGBoost + LightGBM + Ridge (calibré)
      Niveau 1 : LogisticRegression (méta-learner)

    Le méta-learner est entraîné sur les probabilités out-of-fold
    des modèles de base via PurgedTimeSeriesSplit.

    Parameters
    ----------
    n_optuna_trials : int — nombre d'essais Optuna par modèle de base
    """

    def __init__(self, n_optuna_trials: int = 50):
        self.n_optuna_trials = n_optuna_trials
        self.xgb_model_: Optional[xgb.XGBClassifier] = None
        self.lgb_model_: Optional[lgb.LGBMClassifier] = None
        self.ridge_model_: Optional[CalibratedClassifierCV] = None
        self.meta_model_:  Optional[LogisticRegression] = None
        self.scaler_:      Optional[StandardScaler] = None
        self.features_:    Optional[List[str]] = None
        self.classes_ = np.array([0, 1])

    def _oof_probas(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        model_cls,
        fit_kwargs: dict,
    ) -> np.ndarray:
        """Génère les probabilités out-of-fold pour le méta-learner."""
        oof = np.zeros(len(X))
        cv = PurgedTimeSeriesSplit(n_splits=5)

        for tr_idx, val_idx in cv.split(X):
            m = model_cls(**fit_kwargs)
            m.fit(X.iloc[tr_idx], y.iloc[tr_idx])
            oof[val_idx] = m.predict_proba(X.iloc[val_idx])[:, 1]

        return oof

    def fit(self, df: pd.DataFrame, y: pd.Series) -> "AlphaEdgeEnsemble":
        logger.info("Training AlphaEdgeEnsemble...")

        self.features_ = _available(df)
        if not self.features_:
            raise ValueError("Aucune feature disponible dans df.")

        X = _prepare_X(df, self.features_)

        trials_each = max(10, self.n_optuna_trials // 2)

        logger.info("  [1/3] XGBoost...")
        self.xgb_model_ = _xgb_objective(X, y, trials_each)

        logger.info("  [2/3] LightGBM...")
        self.lgb_model_ = _lgb_objective(X, y, trials_each)

        logger.info("  [3/3] Ridge (calibré)...")
        self.scaler_ = StandardScaler()
        X_scaled = pd.DataFrame(self.scaler_.fit_transform(X), columns=X.columns, index=X.index)
        ridge_base = RidgeClassifier(alpha=1.0, random_state=42)
        self.ridge_model_ = CalibratedClassifierCV(ridge_base, cv=5, method="sigmoid")
        self.ridge_model_.fit(X_scaled, y)

        # ── Niveau 1 : méta-learner sur probas OOF
        logger.info("  [Meta] Stacking LogisticRegression...")
        oof_xgb = self._oof_probas(X, y, xgb.XGBClassifier,
                                     {**self.xgb_model_.get_params(), "eval_metric": "auc"})
        oof_lgb = self._oof_probas(X, y, lgb.LGBMClassifier,
                                     {**self.lgb_model_.get_params(), "verbose": -1})
        oof_ridge = self.ridge_model_.predict_proba(X_scaled)[:, 1]

        meta_X = np.column_stack([oof_xgb, oof_lgb, oof_ridge])
        self.meta_model_ = LogisticRegression(C=1.0, random_state=42, max_iter=500)
        self.meta_model_.fit(meta_X, y)

        weights = self.get_model_weights()
        logger.info(
            f"  Weights → XGB: {weights['xgb']:.3f} | "
            f"LGB: {weights['lgb']:.3f} | Ridge: {weights['ridge']:.3f}"
        )
        return self

    def predict_proba(self, df: pd.DataFrame) -> np.ndarray:
        X = _prepare_X(df, self.features_)
        X_scaled = pd.DataFrame(self.scaler_.transform(X), columns=X.columns, index=X.index)

        p_xgb = self.xgb_model_.predict_proba(X)[:, 1]
        p_lgb = self.lgb_model_.predict_proba(X)[:, 1]
        p_ridge = self.ridge_model_.predict_proba(X_scaled)[:, 1]

        meta_X = np.column_stack([p_xgb, p_lgb, p_ridge])
        return self.meta_model_.predict_proba(meta_X)

    def predict(self, df: pd.DataFrame) -> np.ndarray:
        return (self.predict_proba(df)[:, 1] >= 0.5).astype(int)

    def get_model_weights(self) -> Dict[str, float]:
        if self.meta_model_ is None:
            return {"xgb": 1/3, "lgb": 1/3, "ridge": 1/3}
        coefs = np.abs(self.meta_model_.coef_[0])
        total = coefs.sum() or 1.0
        return {
            "xgb":   round(coefs[0] / total, 4),
            "lgb":   round(coefs[1] / total, 4),
            "ridge": round(coefs[2] / total, 4),
        }

    def save(self, path) -> None:
        with open(path, "wb") as f:
            pickle.dump(self, f)

    @staticmethod
    def load(path) -> "AlphaEdgeEnsemble":
        with open(path, "rb") as f:
            return pickle.load(f)